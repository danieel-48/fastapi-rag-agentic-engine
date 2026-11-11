import chromadb
from openai import AsyncOpenAI
from app.config import settings
from app.schemas.rag import RAGSourceChunk, RAGQueryResponse


class RAGService:
    def __init__(self):
        # Base de datos vectorial persistida en la carpeta local './chroma_db'
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        # Usamos distancia coseno (la métrica estándar para comparar significado de texto)
        self.collection = self.chroma_client.get_or_create_collection(
            name="company_knowledge",
            metadata={"hnsw:space": "cosine"}
        )

    def _get_ai_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
    
    async def get_embedding(self, text: str) -> list[float]:
        """Convierte texto en un vector numérico usando el modelo de embeddings."""
        client = self._get_ai_client()
        response = await client.embeddings.create(
            model=settings.embedding_model_name,
            input=text
        )
        return response.data[0].embedding
    
    def chunk_text(self, text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
        """
        Divide un texto largo en fragmentos más pequeños con solapamiento (overlap)
        para preservar el contexto entre frases consecutivas.
        """
        chunks = []
        start = 0
        text_length = len(text)
        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            # Avanzamos restando el solapamiento para no cortar ideas por la mitad
            start += chunk_size - overlap
        return chunks
    async def ingest_document(self, document_id: str, title: str, content: str) -> int:
        """Divide el documento, calcula sus vectores y los indexa en ChromaDB."""
        chunks = self.chunk_text(content)
        
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        for i, chunk in enumerate(chunks):
            embedding = await self.get_embedding(chunk)
            ids.append(f"{document_id}_chunk_{i}")
            embeddings.append(embedding)
            documents.append(chunk)
            metadatas.append({
                "document_id": document_id,
                "title": title,
                "chunk_index": i
            })
        # Guardamos en la base vectorial en batch
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        return len(chunks)
        
    async def query_knowledge(self, question: str, top_k: int = 3) -> RAGQueryResponse:
        """Busca los fragmentos más cercanos y le pide al LLM responder con base en ellos."""
        # 1. Calculamos el embedding de la pregunta
        question_embedding = await self.get_embedding(question)
        # 2. Búsqueda por similitud coseno en ChromaDB
        results = self.collection.query(
            query_embeddings=[question_embedding],
            n_results=top_k
        )
        # 3. Formateamos las fuentes recuperadas
        sources: list[RAGSourceChunk] = []
        retrieved_texts = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                sources.append(
                    RAGSourceChunk(
                        document_id=meta["document_id"],
                        chunk_index=meta["chunk_index"],
                        content=doc,
                        distance=round(dist, 4)
                    )
                )
                retrieved_texts.append(doc)
        context_block = "\n---\n".join(retrieved_texts)
        # 4. Prompt aumentado (Augmented Generation) con instrucciones de grounding
        system_prompt = (
            "Eres un asistente corporativo de alta precisión. "
            "Responde a la pregunta del usuario utilizando EXCLUSIVAMENTE la información provista en el CONTEXTO. "
            "Si la respuesta no se encuentra en el contexto, di claramente: 'No cuento con información suficiente en los documentos disponibles para responder esa pregunta.' "
            "No inventes ni asumas información."
        )
        user_prompt = f"CONTEXTO RECUPERADO:\n{context_block}\n\nPREGUNTA DEL USUARIO:\n{question}"
        # 5. Generación de respuesta con el LLM
        client = self._get_ai_client()
        completion = await client.chat.completions.create(
            model=settings.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0  # Temperatura 0 para respuestas deterministas y sin creatividad/alucinación
        )
        answer_text = completion.choices[0].message.content or ""
        return RAGQueryResponse(
            question=question,
            answer=answer_text,
            sources=sources
        )
# Singleton del servicio para inyectar en los routers
rag_service = RAGService()