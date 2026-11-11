Caracteristicas Principales:

1. Structured Outputs Deterministas (Pydantic v2):

Eliminación del vibe coding mediante contratos estrictos (response_format: json_schema) validados en tipos Literal. Extracción infalible de incidentes técnicos sin errores de parseo JSON.

2. Streaming en Tiempo Real con Server-Sent Events (SSE):
Respuestas token por token mediante StreamingResponse(text/event-stream) asíncronas para una experiencia interactiva sin latencia percibida.

3. Inferencia Híbrida Desacoplada (Cloud & Edge):
Soporta Google Gemini API (gemini-2.5-flash / gemini-embedding-001) y ejecución local 100% privada sin internet mediante Ollama (llama3.2:3b) cambiando únicamente variables de entorno.

4. Arquitecturas RAG & Bases Vectoriales:

ChromaDB: Prototipado y colecciones vectoriales locales.
PostgreSQL + pgvector: Estándar empresarial con cálculo de distancia coseno nativo en SQL (<=>), compatible con filtrado relacional y persistencia en volúmenes Docker.

5. Agentes Autónomos con Tool Calling (Patrón ReAct):
Agente SRE con loop de razonamiento multi-paso capaz de diagnosticar microservicios caídos, reiniciarlos de forma autónoma y notificar alertas a Slack con bitácora auditable de pasos.

6. Empaquetado y Despliegue con Docker y uv:
Dockerfile ultrarrápido multi-stage optimizado con uv.
Orquestación con docker-compose.yml que enlaza FastAPI y PostgreSQL con verificación de salud (healthcheck).

Estructura:

ai-dev-path/
├── app/
│   ├── config.py              # Configuración tipada con pydantic-settings
│   ├── database.py            # Conexión SQLAlchemy y modelos pgvector
│   ├── routers/
│   │   ├── llm.py             # Endpoints de Streaming y Structured Outputs
│   │   ├── rag.py             # Endpoints de RAG (ChromaDB y PostgreSQL)
│   │   └── agent.py           # Endpoints de Agentes Autónomos
│   ├── schemas/
│   │   ├── extraction.py      # Esquemas de entrada y salidas estrictas
│   │   ├── rag.py             # Esquemas de ingestión y trazabilidad (grounding)
│   │   └── agent.py           # Esquemas de comandos y bitácoras del agente
│   └── services/
│       ├── rag_service.py     # Lógica de chunking, embeddings y ChromaDB
│       ├── pgvector_service.py# Lógica SQL vectorial con pgvector
│       └── agent_service.py   # Herramientas y loop de razonamiento ReAct
├── docker-compose.yml         # Orquestación de API + PostgreSQL pgvector
├── Dockerfile                 # Imagen de producción construida con uv
├── pyproject.toml             # Manifiesto moderno de dependencias (PEP 621)
└── uv.lock                    # Árbol de dependencias determinista