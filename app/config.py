from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    env: Literal["dev", "prod"] = "dev"
    port: int = 8090  # Distinct from python-adtech-mcp-client (8080)

    # MCP Server — real-python-adtech-mcp-server SSE endpoint
    mcp_server_url: str = "http://localhost:8085/sse"

    # LLM provider
    llm_provider: Literal["groq", "vertex"] = "groq"
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    vertex_project_id: str = ""
    vertex_location: str = "us-central1"
    vertex_model: str = "gemini-2.0-flash-001"

    # CrewAI settings
    crew_verbose: bool = False
    crew_max_iter: int = 10

    # Database
    database_url: str = "sqlite+aiosqlite:///./crew_tickets.db"

    # Knowledge Base
    kb_provider: Literal["local", "gcp"] = "local"
    kb_docs_dir: str = "./kb-docs"
    chroma_db_path: str = "./chroma_db"
    gcp_project_id: str = ""
    kb_datastore_id: str = "adtech-kb"
    kb_serving_config: str = "default_search"

    # Security
    api_key: str = ""
    cors_origins: str = "*"


settings = Settings()
