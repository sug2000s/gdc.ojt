from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Bulletin Board API"
    db_url: str = Field(
        default="sqlite:///./bulletin.db",
        validation_alias=AliasChoices("DB_URL", "DATABASE_URL", "database_url"),
    )
    secret_key: str = "change-me"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
