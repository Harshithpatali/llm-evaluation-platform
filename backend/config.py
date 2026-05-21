import os
from dotenv import load_dotenv

load_dotenv()


class Settings:

    APP_NAME: str = (
        "AI Reliability Platform"
    )

    ENVIRONMENT: str = os.getenv(
        "ENVIRONMENT",
        "development"
    )

    GROQ_API_KEY: str = os.getenv(
        "GROQ_API_KEY",
        ""
    )

    REDIS_HOST: str = os.getenv(
        "REDIS_HOST",
        ""
    )

    REDIS_PORT: int = int(
        os.getenv("REDIS_PORT", 6379)
    )

    REDIS_PASSWORD: str = os.getenv(
        "REDIS_PASSWORD",
        ""
    )

    GROQ_MODEL: str = (
        "llama-3.3-70b-versatile"
    )

    REQUEST_TIMEOUT: int = 30

    @property
    def redis_url(self) -> str:

        return (
            f"rediss://:"
            f"{self.REDIS_PASSWORD}"
            f"@"
            f"{self.REDIS_HOST}"
            f":"
            f"{self.REDIS_PORT}"
            f"/0"
        )


settings = Settings()