import os
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseModel):
    zetaris_ui_url: str = os.getenv("ZETARIS_UI_URL", "http://localhost:3000")
    zetaris_api_url: str = os.getenv("ZETARIS_API_URL", "http://localhost:8888/api/v1.0")
    zetaris_admin_email: str = os.getenv("ZETARIS_ADMIN_EMAIL", "admin@zetaris.com")
    zetaris_admin_password: str = os.getenv("ZETARIS_ADMIN_PASSWORD", "Zetaris1@")
    zetaris_query_timeout: int = int(os.getenv("ZETARIS_QUERY_TIMEOUT", "45"))
    
    meterless_nightshift_url: str = os.getenv("METERLESS_NIGHTSHIFT_URL", "http://127.0.0.1:8088/v1")
    meterless_api_key: str = os.getenv("METERLESS_API_KEY", "meterless-local-hotspot")
    
    nvidia_api_key: Optional[str] = os.getenv("NVIDIA_API_KEY", None)
    nvidia_api_url: str = os.getenv("NVIDIA_API_URL", "https://integrate.api.nvidia.com/v1")
    nemotron_model: str = os.getenv("NEMOTRON_MODEL", "nvidia/llama-3.1-nemotron-70b-instruct")
    
    max_chunks: int = int(os.getenv("MAX_CHUNKS", "12"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "8000"))
    carryover_tokens: int = int(os.getenv("CARRYOVER_TOKENS", "256"))


def get_settings() -> Settings:
    return Settings()
