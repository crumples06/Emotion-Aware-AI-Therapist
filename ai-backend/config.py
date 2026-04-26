import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    class Settings(BaseSettings):
        app_name: str = "AI Therapist Backend"
        debug: bool = False
        port: int = 8001
        host: str = "0.0.0.0"

        groq_api_key: str
        groq_model: str = "llama-3.1-8b-instant"
        max_tokens: int = 300

        elevenlabs_api_key: str = ""
        elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"

        did_api_key: str = ""

        secret_key: str
        allowed_origins: list = ["http://localhost:3000"]

        model_config = SettingsConfigDict(env_file=".env")

settings = Settings()
