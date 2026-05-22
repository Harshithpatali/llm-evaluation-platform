from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):

    # =========================================
    # ENVIRONMENT
    # =========================================

    ENVIRONMENT: str = Field(default="development")

    # =========================================
    # GROQ
    # =========================================

    GROQ_API_KEY: str

    # =========================================
    # REDIS
    # =========================================

    REDIS_HOST: str

    REDIS_PORT: int = 6379

    REDIS_PASSWORD: str

    # =========================================
    # APP
    # =========================================

    APP_NAME: str = "LLM Reliability Platform"

    LOG_LEVEL: str = "INFO"

    # =========================================
    # REDIS TLS URL
    # =========================================

    @property
    def REDIS_URL(self) -> str:

        return (
            f"rediss://:"
            f"{self.REDIS_PASSWORD}"
            f"@{self.REDIS_HOST}:"
            f"{self.REDIS_PORT}"
            f"?ssl_cert_reqs=required"
        )

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
print("GROQ:", settings.GROQ_API_KEY[:10])