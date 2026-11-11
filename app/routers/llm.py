from collections.abc import AsyncGenerator
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI, APIError
from app.config import settings
from app.schemas.extraction import ChatPromptRequest, TicketAnalysis

router = APIRouter(prefix="/llm", tags=["LLM Operations"])


def get_ai_client() -> AsyncOpenAI:
    """Instancia del cliente asíncrono configurado con nuestras variables de entorno."""
    return AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )


@router.post(
    "/analyze-ticket",
    response_model=TicketAnalysis,
    summary="Extracción estructurada con garantía de esquema (Structured Outputs)",
    description="Convierte texto libre en un objeto JSON fuertemente tipado según el modelo Pydantic."
)
async def analyze_ticket(request: ChatPromptRequest) -> TicketAnalysis:
    client = get_ai_client()
    try:
        # beta.chat.completions.parse convierte automáticamente el schema de Pydantic
        # a JSON Schema y obliga al modelo a responder bajo esa estructura
        completion = await client.beta.chat.completions.parse(
            model=settings.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "Eres un analista de incidentes de software. Analiza el reporte y clasifícalo estrictamente bajo el esquema solicitado."
                },
                {"role": "user", "content": request.prompt}
            ],
            response_format=TicketAnalysis,
        )
        
        parsed_result = completion.choices[0].message.parsed
        if not parsed_result:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="El modelo no devolvió una estructura válida."
            )
        return parsed_result

    except APIError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Error en el proveedor de IA: {e.message}"
        )


@router.post(
    "/stream-chat",
    summary="Streaming de tokens en tiempo real con SSE (Server-Sent Events)",
    description="Retorna tokens progresivos a través de text/event-stream para respuestas interactivas sin latencia percibida."
)
async def stream_chat(request: ChatPromptRequest):
    client = get_ai_client()

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            stream = await client.chat.completions.create(
                model=settings.model_name,
                messages=[
                    {"role": "system", "content": request.system_instruction or "Eres un asistente útil."},
                    {"role": "user", "content": request.prompt}
                ],
                stream=True,
            )
            async for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                if token:
                    # Formato estándar de Server-Sent Events (SSE): 'data: <contenido>\n\n'
                    yield f"data: {token}\n\n"
            
            # Marcador estándar para avisar al cliente que terminó la transmisión
            yield "data: [DONE]\n\n"

        except APIError as e:
            yield f"event: error\ndata: {e.message}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Evita que Nginx u otros proxies almacenen en búfer los tokens
        }
    )