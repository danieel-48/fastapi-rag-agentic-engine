from pydantic import BaseModel, Field
from typing import Any


class AgentPromptRequest(BaseModel):
    """Instrucción o requerimiento que el usuario le da al agente."""
    prompt: str = Field(..., min_length=3, description="Objetivo o consulta para el agente")


class AgentActionStep(BaseModel):
    """Registro de cada acción o herramienta que el agente decidió invocar."""
    tool_name: str
    arguments: dict[str, Any]
    output: Any


class AgentResponse(BaseModel):
    """Respuesta final del agente junto con la bitácora de ejecución."""
    prompt: str
    final_answer: str
    steps_executed: list[AgentActionStep] = Field(
        default_factory=list,
        description="Bitácora de herramientas y decisiones tomadas por el agente"
    )