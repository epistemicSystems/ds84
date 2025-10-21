"""Configuration management for Realtor AI Copilot"""
import os
from pydantic_settings import BaseSettings
from typing import Optional


class LLMConfig(BaseSettings):
    """LLM API configuration"""
    openai_api_key: str
    anthropic_api_key: Optional[str] = None
    default_model: str = "gpt-4"
    timeout: int = 30

    class Config:
        env_file = ".env"
        env_prefix = "LLM_"


class DatabaseConfig(BaseSettings):
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    user: str = "postgres"
    password: str = "postgres"
    database: str = "realtor_ai"

    class Config:
        env_file = ".env"
        env_prefix = "DB_"

    @property
    def connection_string(self) -> str:
        """Get PostgreSQL connection string"""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class VectorDBConfig(BaseSettings):
    """Vector database configuration"""
    host: str = "localhost"
    port: int = 5433
    user: str = "postgres"
    password: str = "postgres"
    database: str = "vector_store"

    class Config:
        env_file = ".env"
        env_prefix = "VECTORDB_"

    @property
    def connection_string(self) -> str:
        """Get PostgreSQL connection string for vector DB"""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class Settings(BaseSettings):
    """Application settings"""
    app_name: str = "Realtor AI Copilot"
    debug: bool = True
    environment: str = "development"

    # Nested configs
    llm: LLMConfig = None
    database: DatabaseConfig = None
    vector_db: VectorDBConfig = None

    class Config:
        env_file = ".env"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize nested configs
        self.llm = LLMConfig()
        self.database = DatabaseConfig()
        self.vector_db = VectorDBConfig()


# Global settings instance
settings = Settings()
