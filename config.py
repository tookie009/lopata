from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    sh_client_id: str
    sh_client_secret: str
    sh_base_url: str = "https://sh.dataspace.copernicus.eu"
    sh_token_url: str = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

    # How long a raw NDVI fetch is reused for an identical request (same bbox/resolution/date
    # range/cloud-cover) instead of re-hitting Copernicus - see ndvi._RAW_NDVI_CACHE. Set to 0 to
    # effectively disable caching (e.g. while testing) without a code change.
    ndvi_cache_ttl_seconds: float = 3600


settings = Settings()
