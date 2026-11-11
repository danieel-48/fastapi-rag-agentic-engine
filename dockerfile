FROM python:3.12-slim

# 1. Copiamos el binario oficial de uv (ultrarrápido y ligero)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 2. Copiamos solo los manifiestos de dependencias para aprovechar la caché de Docker
COPY pyproject.toml uv.lock ./

# 3. Instalamos las dependencias exactas usando el lockfile
RUN uv sync --frozen --no-dev --no-cache

# 4. Copiamos el código de la aplicación
COPY main.py ./
COPY app/ ./app/

# Exponemos el puerto estándar de FastAPI
EXPOSE 8000

# 5. Comando de arranque: en producción vinculamos al host 0.0.0.0 (accesible desde fuera del contenedor)
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]