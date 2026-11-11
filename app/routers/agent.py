from fastapi import APIRouter, HTTPException, status
from app.schemas.agent import AgentPromptRequest, AgentResponse
from app.services.agent_service import agent_service

router = APIRouter(prefix="/agent", tags=["Autonomous Agents & Tool Calling"])

@router.post(
    "/run",
    response_model=AgentResponse,
    summary="Ejecutar un Agente Autónomo con ReAct y Tool Calling",
    description="El modelo analiza el objetivo, decide qué herramientas ejecutar dinámicamente y devuelve el resultado con su bitácora de pasos."
)

async def run_autonomous_agent(payload: AgentPromptRequest) -> AgentResponse:
    try:
        response = await agent_service.run_agent(user_prompt=payload.prompt)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error durante la ejecución del agente: {str(e)}"
        )