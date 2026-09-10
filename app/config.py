from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    MODEL_PATH: str
    METADATA_PATH: str
    LOG_LEVEL: str
    MAX_BATCH_SIZE: int
    API_TITLE: str
    API_KEY: str
    CORS_ORIGINS: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True
    )


settings = Settings()