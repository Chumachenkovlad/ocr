"""Gateway configuration via environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8080
    paddle_url: str = "http://paddle:8080"
    tesseract_url: str = "http://tesseract:8080"
    surya_url: str = "http://surya:8080"
    doctr_url: str = "http://doctr:8080"
    request_timeout: float = 60.0
    log_level: str = "INFO"

    model_config = {"env_prefix": "GATEWAY_"}
