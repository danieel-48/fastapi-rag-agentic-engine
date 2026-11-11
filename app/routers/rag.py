from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.orm import Session
from app.schemas.rag import (
    DocumentIngestRequest,
    IngestResponse,
    RAGQueryRequest,
    RAGQueryResponse
)
from app.services.rag_service import rag_service
from app.services.pgvector_service import pgvector_service
from app.database import get_db

router = APIRouter(prefix="/rag", tags=["RAG & Knowledge Base"])


# --- ENDPOINTS CHROMADB (LOCAL PROTOTYPING) ---

@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Indexar documento en ChromaDB"
)
async def ingest_document(payload: DocumentIngestRequest) -> IngestResponse:
    try:
        total_chunks = await rag_service.ingest_document(
            document_id=payload.document_id,
            title=payload.title,
            content=payload.content
        )
        return IngestResponse(
            document_id=payload.document_id,
            total_chunks=total_chunks,
            message=f"Documento '{payload.title}' indexado en ChromaDB ({total_chunks} chunks)."
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en ChromaDB: {str(e)}"
        )


@router.post(
    "/query",
    response_model=RAGQueryResponse,
    summary="Consultar base vectorial ChromaDB"
)
async def query_knowledge(payload: RAGQueryRequest) -> RAGQueryResponse:
    try:
        return await rag_service.query_knowledge(
            question=payload.question,
            top_k=payload.top_k
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error consultando ChromaDB: {str(e)}"
        )


# --- ENDPOINTS POSTGRESQL + PGVECTOR (ENTERPRISE) ---

@router.post(
    "/pg/ingest",
    response_model=IngestResponse,
    summary="Indexar documento en PostgreSQL con pgvector",
    description="Inserta filas en la tabla document_chunks guardando metadatos y vectores nativos de 3072 dimensiones."
)
async def pg_ingest_document(
    payload: DocumentIngestRequest,
    db: Session = Depends(get_db)
) -> IngestResponse:
    try:
        total_chunks = await pgvector_service.ingest_document(
            db=db,
            document_id=payload.document_id,
            title=payload.title,
            content=payload.content
        )
        return IngestResponse(
            document_id=payload.document_id,
            total_chunks=total_chunks,
            message=f"Documento '{payload.title}' indexado en PostgreSQL ({total_chunks} chunks)."
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en PostgreSQL pgvector: {str(e)}"
        )


@router.post(
    "/pg/query",
    response_model=RAGQueryResponse,
    summary="Consultar base relacional con búsqueda vectorial nativa (pgvector)",
    description="Ejecuta consulta SQL con el operador <=> de distancia coseno directamente en la base de datos."
)
async def pg_query_knowledge(
    payload: RAGQueryRequest,
    db: Session = Depends(get_db)
) -> RAGQueryResponse:
    try:
        return await pgvector_service.query_knowledge(
            db=db,
            question=payload.question,
            top_k=payload.top_k
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error consultando pgvector: {str(e)}"
        )