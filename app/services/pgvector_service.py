from sqlalchemy.orm import Session
from app.database import DocumentChunkModel
from app.services.rag_service import rag_service
from app.schemas.rag import RAGSourceChunk, RAGQueryResponse
from app.config import settings


class PgVectorService:
    async def ingest_document(
        self,
        db: Session,
        document_id: str,
        title: str,
        content: str
    ) -> int:
        """Divide el documento, calcula los embeddings y los guarda en PostgreSQL."""
        chunks = rag_service.chunk_text(content)

        # Si ya existían fragmentos previos de este documento, los limpiamos (idempotencia)

        db.query(DocumentChunkModel).filter(
            DocumentChunkModel.document_id == document_id
        ).delete()
        for i, chunk in enumerate(chunks):
            embedding = await rag_service.get_embedding(chunk)
            chunk_record = DocumentChunkModel(
                document_id=document_id,
                title=title,
                chunk_index=i,
                content=chunk,
                embedding=embedding
            )
            db.add(chunk_record)
        db.commit()
        return len(chunks)
    
    async def query_knowledge(
        self,
        db: Session,
        question: str,
        top_k: int = 3
    ) -> RAGQueryResponse:
        """
        Búsqueda por similitud coseno directa en el motor SQL de PostgreSQL
        usando el operador nativo <=> de pgvector.
        """
        # 1. Calculamos el embedding de la pregunta
        question_embedding = await rag_service.get_embedding(question)
        # 2. Consulta SQL con SQLAlchemy calculando la distancia coseno
        # En SQL se traduce a: ORDER BY embedding <=> '[...]' LIMIT top_k
        results = (
            db.query(
                DocumentChunkModel,
                DocumentChunkModel.embedding.cosine_distance(question_embedding).label("distance")
            )
            .order_by("distance")
            .limit(top_k)
            .all()
        )
        sources: list[RAGSourceChunk] = []
        retrieved_texts = []
        for record, distance in results:
            sources.append(
                RAGSourceChunk(
                    document_id=record.document_id,
                    chunk_index=record.chunk_index,
                    content=record.content,
                    distance=round(float(distance), 4)
                )
            )
            retrieved_texts.append(record.content)
        context_block = "\n---\n".join(retrieved_texts)
        # 3. Prompt aumentado anti-alucinaciones
        system_prompt = (
            "Eres un asistente corporativo de alta precisión conectado a PostgreSQL. "
            "Responde a la pregunta del usuario utilizando EXCLUSIVAMENTE la información provista en el CONTEXTO. "
            "Si la respuesta no se encuentra en el contexto, di claramente: 'No cuento con información suficiente en los documentos disponibles para responder esa pregunta.' "
            "No inventes ni asumas información."
        )
        user_prompt = f"CONTEXTO DE POSTGRESQL:\n{context_block}\n\nPREGUNTA DEL USUARIO:\n{question}"
        client = rag_service._get_ai_client()
        completion = await client.chat.completions.create(
            model=settings.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        answer_text = completion.choices[0].message.content or ""
        return RAGQueryResponse(
            question=question,
            answer=answer_text,
            sources=sources
        )
pgvector_service = PgVectorService()