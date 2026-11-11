from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers.llm import router as llm_router
from app.routers.rag import router as rag_router
from app.routers.agent import router as agent_router 
from app.database import init_db

@asynccontextmanager

async def lifespan(app: FastAPI):
    """Inicializa la extensión pgvector y las tablas SQL al arrancar el servidor."""
    init_db()
    yield

app = FastAPI(
    title="AI Core API - Spec-Driven LLM Backend",
    description="Backend de servicios de IA con FastAPI: Structured Outputs, Streaming SSE y arquitectura desacoplada.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS: Fundamental para cuando conectes frontends en React, Next.js o extensiones
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registramos el router bajo el prefijo /api/v1
app.include_router(llm_router, prefix="/api/v1")
app.include_router(rag_router, prefix="/api/v1") 
app.include_router(agent_router, prefix="/api/v1")


@app.get("/health", tags=["System"])
async def health_check():
    """Endpoint simple para verificar que el servicio está activo (ideal para health probes en Kubernetes/AWS)."""
    return {"status": "ok", "service": "AI Core API"}