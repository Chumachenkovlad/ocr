"""Service configuration loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8080
    device: str = "cpu"
    max_file_size_mb: int = 50
    log_level: str = "INFO"

    model_config = {"env_prefix": "OCR_"}
