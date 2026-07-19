from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    sh_client_id: str
    sh_client_secret: str
    sh_base_url: str = "https://sh.dataspace.copernicus.eu"
    sh_token_url: str = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"


settings = Settings()
