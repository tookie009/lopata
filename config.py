from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    sh_client_id: str
    sh_client_secret: str
    sh_base_url: str = "https://sh.dataspace.copernicus.eu"
    sh_token_url: str = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

    # How long a raw NDVI fetch is reused for an identical request (same bbox/resolution/date
    # range/cloud-cover) instead of re-hitting Copernicus - see ndvi._RAW_NDVI_CACHE. Was 3600
    # (1h) when this was memory-only; now that db_cache.py makes it survive a process restart,
    # it's safe to raise substantially - a Sentinel-2 revisit is ~5 days, so a few hours of
    # staleness is never meaningful. Set to 0 to effectively disable caching (e.g. while testing)
    # without a code change.
    ndvi_cache_ttl_seconds: float = 21600

    # Persistent (Postgres) L2 cache for the raw NDVI raster + acquisition metadata - see
    # db_cache.py. Disabled by default (memory-only _RAW_NDVI_CACHE, as before) until explicitly
    # configured; a missing/unreachable DB never blocks a request, it just falls back to
    # memory-only behavior. Deliberately its own schema/credentials in kret's existing Postgres
    # instance, not kret's own farming_db tables/user.
    #
    # A single connection string (postgresql://user:password@host:port/dbname), not separate
    # host/port/dbname/user/password fields - matches kret's own SPRING_DATASOURCE_URL convention
    # and Railway's native per-plugin DATABASE_URL, and psycopg2 accepts this directly as a DSN
    # with no parsing needed on this end. lopata_db_schema stays separate since a schema isn't
    # part of a standard Postgres URI.
    lopata_db_enabled: bool = False
    lopata_db_url: str = ""
    lopata_db_schema: str = "lopata"


settings = Settings()
