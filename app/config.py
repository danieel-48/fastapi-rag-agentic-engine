from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Configuración de conexión al LLM
    openai_api_key: str = "sk-placeholder"
    openai_base_url: str = "https://api.openai.com/v1"
    model_name: str = "gpt-4o-mini"
    embedding_model_name: str = "gemini-embedding-001"
    database_url: str = "postgresql://postgres:postgres@localhost:5432/rag_enterprise"

# Instancia única que importaremos en el resto del proyecto
settings = Settings()