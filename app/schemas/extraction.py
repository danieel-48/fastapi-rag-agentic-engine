from typing import Literal
from pydantic import BaseModel, Field

class ChatPromptRequest(BaseModel):
    """Payload de entrada que recibe nuestra API desde el cliente"""
    prompt: str = Field(..., min_length=2, description="Texto o requerimento del usuario")
    system_instruction: str | None = Field(
        default="Eres un asistente tecnico especializado en desarrollo de software e inteligencia artificial.", 
        description="Instrucción de sistema para condicionar el rol del modelo"
    )

class TicketAnalysis(BaseModel):
    """Contrato estricto que obligamos al LLM a devolver."""
    categoria: Literal["bug", "feature_request", "soporte", "infraestructura"] = Field(
        description="Clasificación principal del requerimiento"
    )
    prioridad: Literal["baja", "media", "alta", "critica"] = Field(
        description="Nivel de urgencia evaluado a partir del impacto técnico reportado"
    )
    resumen_tecnico: str = Field(
        description="Resumen de máximo dos oraciones en lenguaje técnico claro y conciso"
    )
    servicios_afectados: list[str] = Field(
        default_factory=list,
        description="Lista de microservicios o tecnologías involucradas (ej. Auth, Pagos, Redis, PostgreSQL)"
    )
    accion_recomendada: str = Field(
        description="Primer paso concreto que debe ejecutar el equipo de ingeniería para mitigar o resolver"
    )