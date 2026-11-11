from pydantic import BaseModel, Field

class DocumentIngestRequest(BaseModel):
     """Payload para enviar un documento o política interna a la base de datos vectorial."""
     document_id: str = Field(..., description="Identificador único del documento (ej. 'manual-vacaciones-2026')")
     title: str = Field(..., description="Titulo")
     content: str = Field(..., min_length=20, description="Texto completo del documento a indexar")


class IngestResponse(BaseModel):
    """Confirmación del proceso de fragmentación (chunking) e indexación."""
    document_id: str
    total_chunks: int = Field(description="Cantidad de fragmentos en los que se dividió el texto")
    message: str

class RAGQueryRequest(BaseModel):
    """Consulta que hace el usuario sobre la documentación indexada."""
    question: str = Field(..., min_length=3, description="Pregunta del usuario")
    top_k: int = Field(default=3, ge=1, le=10, description="Número de fragmentos más cercanos a recuperar")

class RAGSourceChunk(BaseModel):
    """Evidencia recuperada de la base vectorial para dar trazabilidad (grounding)."""
    document_id: str
    chunk_index: int
    content: str
    distance: float = Field(description="Distancia matemática (a menor distancia, mayor relevancia)")
    
class RAGQueryResponse(BaseModel):
    """Respuesta generada por el LLM respaldada por las fuentes consultadas."""
    question: str
    answer: str
    sources: list[RAGSourceChunk] = Field(description="Fragmentos exactos usados como evidencia")