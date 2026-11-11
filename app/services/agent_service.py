import json
from openai import AsyncOpenAI
from app.config import settings
from app.schemas.agent import AgentActionStep, AgentResponse


# =====================================================================
# 1. HERRAMIENTAS REALES (Funciones de Python que el agente puede invocar)
# =====================================================================

def check_system_health(system_name: str) -> dict:
    """Consulta métricas de telemetría de un microservicio específico."""
    system = system_name.lower().strip()
    simulated_systems = {
        "payments-gateway": {
            "status": "CRITICAL",
            "cpu_usage": "98%",
            "memory": "94%",
            "error_rate": "42%",
            "active_connections": 1420,
            "root_cause": "Pool de conexiones agotado en la base de datos de pagos"
        },
        "auth-service": {
            "status": "HEALTHY",
            "cpu_usage": "22%",
            "memory": "35%",
            "error_rate": "0.1%",
            "active_connections": 210
        },
        "database-primary": {
            "status": "WARNING",
            "cpu_usage": "78%",
            "memory": "82%",
            "error_rate": "4.5%",
            "active_connections": 980
        }
    }
    return simulated_systems.get(
        system,
        {"status": "UNKNOWN", "message": f"El sistema '{system_name}' no existe en el catálogo de infraestructura."}
    )


def restart_service(service_name: str) -> dict:
    """Reinicia un microservicio en el clúster de Kubernetes."""
    return {
        "service": service_name,
        "action": "pod_restart",
        "status": "SUCCESS",
        "message": f"Contenedor de '{service_name}' reiniciado exitosamente. CPU normalizado al 12%."
    }


def notify_slack_channel(channel: str, message: str) -> dict:
    """Envía una alerta a un canal de Slack del equipo de ingeniería."""
    return {
        "status": "DELIVERED",
        "channel": channel,
        "payload_sent": message
    }


# Mapeo de nombres a funciones ejecutables de Python
TOOL_DISPATCHER = {
    "check_system_health": check_system_health,
    "restart_service": restart_service,
    "notify_slack_channel": notify_slack_channel,
}


# =====================================================================
# 2. ESPECIFICACIÓN DE HERRAMIENTAS PARA EL LLM (JSON Schema)
# =====================================================================

TOOLS_SPECIFICATION = [
    {
        "type": "function",
        "function": {
            "name": "check_system_health",
            "description": "Consulta el estado operativo, CPU, memoria y errores de un microservicio. Sistemas disponibles: 'payments-gateway', 'auth-service', 'database-primary'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "system_name": {
                        "type": "string",
                        "description": "Nombre del servicio a inspeccionar"
                    }
                },
                "required": ["system_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "restart_service",
            "description": "Reinicia un microservicio que esté en estado CRITICAL o fallando en Kubernetes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_name": {
                        "type": "string",
                        "description": "Nombre exacto del microservicio a reiniciar"
                    }
                },
                "required": ["service_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "notify_slack_channel",
            "description": "Envía una notificación urgente o reporte al canal de Slack correspondiente (ej. '#devops-alerts').",
            "parameters": {
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "description": "Canal de destino, por ejemplo '#devops-alerts'"
                    },
                    "message": {
                        "type": "string",
                        "description": "Texto del mensaje detallando el incidente y la acción tomada"
                    }
                },
                "required": ["channel", "message"]
            }
        }
    }
]


# =====================================================================
# 3. EL AGENTE Y SU BUCLE DE RAZONAMIENTO (ReAct Loop)
# =====================================================================

class AgentService:
    def _get_ai_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )

    async def run_agent(self, user_prompt: str, max_iterations: int = 5) -> AgentResponse:
        client = self._get_ai_client()

        # Historial de conversación que se alimentará en cada ciclo
        messages = [
            {
                "role": "system",
                "content": (
                    "Eres un Agente Autónomo SRE (Site Reliability Engineer) de élite. "
                    "Tienes acceso a herramientas para diagnosticar servidores, reiniciarlos y enviar alertas. "
                    "Cuando se te asigne un problema, analiza el estado primero, toma las acciones necesarias para mitigar "
                    "y notifica al equipo por Slack si hubo una resolución de incidente. "
                    "Usa tus herramientas de forma lógica y paso a paso."
                )
            },
            {"role": "user", "content": user_prompt}
        ]

        steps_log: list[AgentActionStep] = []

        # Bucle ReAct (Reason + Act): máximo N ciclos para evitar bucles infinitos
        for _ in range(max_iterations):
            # Le pasamos el historial y las herramientas disponibles al LLM
            response = await client.chat.completions.create(
                model=settings.model_name,
                messages=messages,
                tools=TOOLS_SPECIFICATION,
                tool_choice="auto"  # El modelo decide libremente si llamar a una tool o responder
            )

            assistant_message = response.choices[0].message
            messages.append(assistant_message)

            # Si el modelo no llamó herramientas, significa que terminó y tiene la respuesta final
            if not assistant_message.tool_calls:
                return AgentResponse(
                    prompt=user_prompt,
                    final_answer=assistant_message.content or "Operación completada sin mensaje.",
                    steps_executed=steps_log
                )

            # Si el modelo decidió llamar a una o varias herramientas:
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)

                # Ejecutamos la función de Python en nuestro servidor
                function_to_call = TOOL_DISPATCHER.get(tool_name)
                if function_to_call:
                    tool_result = function_to_call(**arguments)
                else:
                    tool_result = {"error": f"Herramienta '{tool_name}' no reconocida."}

                # Registramos el paso para la auditoría y respuesta al cliente
                steps_log.append(
                    AgentActionStep(
                        tool_name=tool_name,
                        arguments=arguments,
                        output=tool_result
                    )
                )

                # Le devolvemos el resultado de la función al LLM con el rol 'tool'
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_result)
                })

        # Si agotó las iteraciones sin terminar
        return AgentResponse(
            prompt=user_prompt,
            final_answer="El agente alcanzó el límite máximo de pasos sin concluir la tarea.",
            steps_executed=steps_log
        )


agent_service = AgentService()