"""
Centralized configuration management.

Production-grade settings module.
"""

from dotenv import load_dotenv
import os


# =====================================================
# LOAD ENV VARIABLES
# =====================================================

load_dotenv()


# =====================================================
# SETTINGS CLASS
# =====================================================

class Settings:
    """
    Application settings container.
    """

    # ==========================================
    # APP SETTINGS
    # ==========================================

    APP_NAME: str = (
        "LLM Evaluation Platform"
    )

    API_VERSION: str = "v1"

    ENVIRONMENT: str = os.getenv(
        "ENVIRONMENT",
        "development"
    )

    # ==========================================
    # GROQ SETTINGS
    # ==========================================

    GROQ_API_KEY: str = os.getenv(
        "GROQ_API_KEY",
        ""
    )

    GROQ_MODEL: str = os.getenv(
        "GROQ_MODEL",
        "llama-3.3-70b-versatile"
    )

    # ==========================================
    # REDIS SETTINGS
    # ==========================================

    REDIS_HOST: str = os.getenv(
        "REDIS_HOST",
        "localhost"
    )

    REDIS_PORT: int = int(
        os.getenv(
            "REDIS_PORT",
            6379
        )
    )

    REDIS_PASSWORD: str = os.getenv(
        "REDIS_PASSWORD",
        ""
    )

    # ==========================================
    # API SETTINGS
    # ==========================================

    REQUEST_TIMEOUT: int = 30


# =====================================================
# GLOBAL SETTINGS INSTANCE
# =====================================================

settings = Settings()