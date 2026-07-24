import contextlib
import json
import logging
import re
import threading
import time
import zlib
from datetime import datetime, timedelta, timezone

import numpy as np
import psycopg2
from psycopg2.pool import ThreadedConnectionPool

from config import settings

logger = logging.getLogger("lopata_db_cache")

# Only used to interpolate settings.lopata_db_schema into DDL/DML - psycopg2 can't parameterize
# identifiers, and this comes from an env var (deploy-time config, not user input), but validated
# anyway since it's cheap insurance against a typo'd schema name breaking in a confusing way.
_SCHEMA_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

_pool: ThreadedConnectionPool | None = None
_pool_lock = threading.Lock()

# Circuit breaker for pool creation - see _get_pool's docstring for why this exists (a real
# incident: a downed local Postgres turned every single NDVI request into ~14 failed connection
# attempts at several seconds each, since a fresh TCP connect to a refusing/unreachable host isn't
# instant).
_POOL_RETRY_COOLDOWN_SECONDS = 30.0
_last_pool_failure_at: float | None = None


def _enabled() -> bool:
    return settings.lopata_db_enabled


def _schema() -> str:
    schema = settings.lopata_db_schema
    if not _SCHEMA_NAME_RE.match(schema):
        raise ValueError(f"Niepoprawna nazwa schematu lopata_db_schema: {schema!r}")
    return schema


def _get_pool() -> ThreadedConnectionPool | None:
    """FastAPI runs each sync route handler in a threadpool, so concurrent requests can call
    into this module from different threads at once - a single shared psycopg2 connection (the
    original design here) is NOT safe under that: two threads issuing queries on the same
    connection/socket concurrently can block each other indefinitely, which is exactly what
    caused a real hang in production testing. ThreadedConnectionPool hands each caller its own
    connection for the duration of a call (see _connection() below) and is safe for this.

    Also a circuit breaker: returns None without attempting a connection for
    _POOL_RETRY_COOLDOWN_SECONDS after a failed attempt, rather than retrying on every single
    get()/set() call - a downed/unreachable DB otherwise silently adds a multi-second connection
    timeout to every single NDVI request (verified: ~4s per attempt, ~14 attempts in one
    /field-zones call, adding up to a full minute), defeating the point of a cache meant to make
    things faster. Callers already treat a None/exception here as "cache unavailable" and fall
    through to fetching from Copernicus directly.
    """
    global _pool, _last_pool_failure_at
    if _pool is not None:
        return _pool
    if _last_pool_failure_at is not None and time.monotonic() - _last_pool_failure_at < _POOL_RETRY_COOLDOWN_SECONDS:
        return None
    with _pool_lock:
        if _pool is not None:
            return _pool
        if _last_pool_failure_at is not None and time.monotonic() - _last_pool_failure_at < _POOL_RETRY_COOLDOWN_SECONDS:
            return None
        try:
            _pool = ThreadedConnectionPool(
                minconn=1,
                maxconn=8,
                dsn=settings.lopata_db_url,
            )
            _last_pool_failure_at = None
        except Exception:
            _last_pool_failure_at = time.monotonic()
            logger.warning(
                "lopata DB cache pool init failed - falling back to memory-only cache for %.0fs",
                _POOL_RETRY_COOLDOWN_SECONDS, exc_info=True,
            )
            return None
        return _pool


@contextlib.contextmanager
def _connection():
    pool = _get_pool()
    if pool is None:
        raise RuntimeError("lopata DB cache pool unavailable (see recent warning log)")
    conn = pool.getconn()
    try:
        conn.autocommit = True
        yield conn
    finally:
        pool.putconn(conn)


def init_schema() -> None:
    """Idempotent - safe to call on every startup. A no-op (not an error) when the DB cache
    isn't configured, so a missing/misconfigured DB never blocks the service from starting - it
    just means the process falls back to the in-memory-only cache, same as before this existed."""
    if not _enabled():
        logger.info("lopata DB cache disabled (LOPATA_DB_ENABLED not set) - using in-memory cache only")
        return
    schema = _schema()
    try:
        with _connection() as conn, conn.cursor() as cur:
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {schema}.ndvi_cache (
                    id BIGSERIAL PRIMARY KEY,
                    field_id BIGINT,
                    min_x DOUBLE PRECISION NOT NULL,
                    min_y DOUBLE PRECISION NOT NULL,
                    max_x DOUBLE PRECISION NOT NULL,
                    max_y DOUBLE PRECISION NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    max_cloud_cover DOUBLE PRECISION NOT NULL,
                    time_from TIMESTAMPTZ NOT NULL,
                    time_to TIMESTAMPTZ NOT NULL,
                    mosaicking_order TEXT NOT NULL,
                    ndvi_array BYTEA NOT NULL,
                    ndvi_metadata JSONB,
                    -- Denormalized out of ndvi_metadata purely for convenient SQL querying/
                    -- debugging (e.g. "which cached entries had few candidates considered")
                    -- without parsing JSON - the application itself always reads the full
                    -- ndvi_metadata dict, never these columns directly. NULL together with
                    -- ndvi_metadata for the metadata-less scoring-only entries (see get()/set()).
                    season_from TIMESTAMPTZ,
                    season_to TIMESTAMPTZ,
                    cloud_cover_pct DOUBLE PRECISION,
                    candidates_considered INTEGER,
                    ndvi_mean_at_selection DOUBLE PRECISION,
                    fetched_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    expires_at TIMESTAMPTZ NOT NULL
                )
            """)
            # Two separate partial unique indexes rather than one over all columns: field_id is
            # nullable (a plain UNIQUE constraint treats every NULL as distinct, so it would never
            # dedupe/ON-CONFLICT-match rows from callers that don't send a field_id, letting them
            # accumulate forever) - one indexed path per caller shape instead.
            cur.execute(f"""
                CREATE UNIQUE INDEX IF NOT EXISTS ndvi_cache_field_key
                ON {schema}.ndvi_cache (field_id, width, height, max_cloud_cover, time_from, time_to, mosaicking_order)
                WHERE field_id IS NOT NULL
            """)
            cur.execute(f"""
                CREATE UNIQUE INDEX IF NOT EXISTS ndvi_cache_bbox_key
                ON {schema}.ndvi_cache (min_x, min_y, max_x, max_y, width, height, max_cloud_cover, time_from, time_to, mosaicking_order)
                WHERE field_id IS NULL
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS ndvi_cache_expires_at_idx ON {schema}.ndvi_cache (expires_at)
            """)
        logger.info("lopata DB cache schema ready (%s.ndvi_cache)", schema)
    except Exception:
        logger.exception("Failed to initialize lopata DB cache schema - falling back to in-memory cache only")


def _mosaicking_str(mosaicking_order) -> str:
    return mosaicking_order.value if hasattr(mosaicking_order, "value") else str(mosaicking_order)


def _bbox_tuple(bbox) -> tuple[float, float, float, float]:
    return (
        round(bbox.min_x, 6), round(bbox.min_y, 6),
        round(bbox.max_x, 6), round(bbox.max_y, 6),
    )


def _encode_array(arr: np.ndarray) -> bytes:
    return zlib.compress(np.asarray(arr, dtype=np.float32).tobytes())


def _decode_array(blob, height: int, width: int) -> np.ndarray:
    raw = zlib.decompress(bytes(blob))
    return np.frombuffer(raw, dtype=np.float32).reshape(height, width, -1)


def get(field_id, bbox, width, height, max_cloud_cover, time_interval, mosaicking_order):
    """Returns (ndvi_array, metadata) for a still-valid cache entry, or None on a miss. metadata
    may itself be None (an entry cached without acquisition metadata, e.g. from a small
    scoring-only fetch in fetch_best_vegetation_ndvi_array). Never raises - any DB error is
    logged and treated as a miss so an unreachable/misbehaving DB never blocks a request."""
    if not _enabled():
        return None
    schema = _schema()
    mosaicking_str = _mosaicking_str(mosaicking_order)
    cloud_cover = round(max_cloud_cover, 1)
    min_x, min_y, max_x, max_y = _bbox_tuple(bbox)
    try:
        with _connection() as conn, conn.cursor() as cur:
            if field_id is not None:
                cur.execute(
                    f"""
                    SELECT min_x, min_y, max_x, max_y, ndvi_array, ndvi_metadata
                    FROM {schema}.ndvi_cache
                    WHERE field_id = %s AND width = %s AND height = %s
                      AND max_cloud_cover = %s AND time_from = %s AND time_to = %s
                      AND mosaicking_order = %s AND expires_at > now()
                    """,
                    (field_id, width, height, cloud_cover,
                     time_interval[0], time_interval[1], mosaicking_str),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                row_min_x, row_min_y, row_max_x, row_max_y, blob, metadata = row
                if (round(row_min_x, 6), round(row_min_y, 6), round(row_max_x, 6), round(row_max_y, 6)) != (
                    min_x, min_y, max_x, max_y
                ):
                    # This field's geometry no longer matches what was cached under its id (e.g.
                    # a future boundary-edit feature) - don't serve a raster for the wrong shape,
                    # treat as a miss so the caller re-fetches and overwrites this row.
                    return None
            else:
                cur.execute(
                    f"""
                    SELECT ndvi_array, ndvi_metadata
                    FROM {schema}.ndvi_cache
                    WHERE field_id IS NULL
                      AND min_x = %s AND min_y = %s AND max_x = %s AND max_y = %s
                      AND width = %s AND height = %s AND max_cloud_cover = %s
                      AND time_from = %s AND time_to = %s AND mosaicking_order = %s
                      AND expires_at > now()
                    """,
                    (min_x, min_y, max_x, max_y, width, height, cloud_cover,
                     time_interval[0], time_interval[1], mosaicking_str),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                blob, metadata = row
            return _decode_array(blob, height, width), metadata
    except Exception:
        logger.exception("lopata DB cache read failed - treating as a miss")
        return None


def set(field_id, bbox, width, height, max_cloud_cover, time_interval, mosaicking_order,
        ndvi_array: np.ndarray, metadata: dict | None, ttl_seconds: float) -> None:
    """Upserts the raster (and metadata, if already known) for this key. metadata may be None
    (see get()'s docstring) - COALESCE in the UPDATE branch keeps a previously-stored metadata
    value in place rather than clobbering it back to NULL when a later scoring-only write for the
    same key happens to race a metadata-carrying one (shouldn't normally happen given how the
    keys are constructed, but costs nothing to guard against)."""
    if not _enabled():
        return
    schema = _schema()
    mosaicking_str = _mosaicking_str(mosaicking_order)
    cloud_cover = round(max_cloud_cover, 1)
    min_x, min_y, max_x, max_y = _bbox_tuple(bbox)
    blob = psycopg2.Binary(_encode_array(ndvi_array))
    metadata_json = json.dumps(metadata) if metadata is not None else None
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)

    # Denormalized straight out of metadata (see init_schema's comment on these columns) - stay
    # None/NULL together with ndvi_metadata for the metadata-less scoring-only writes.
    season_from = season_to = None
    cloud_cover_pct = None
    candidates_considered = None
    ndvi_mean_at_selection = None
    if metadata is not None:
        window = metadata.get("time_window_searched") or {}
        if window.get("from"):
            season_from = datetime.fromisoformat(window["from"])
        if window.get("to"):
            season_to = datetime.fromisoformat(window["to"])
        cloud_cover_pct = metadata.get("cloud_cover")
        candidates_considered = metadata.get("candidates_considered")
        ndvi_mean_at_selection = metadata.get("ndvi_mean_at_selection")

    try:
        with _connection() as conn, conn.cursor() as cur:
            if field_id is not None:
                cur.execute(
                    f"""
                    INSERT INTO {schema}.ndvi_cache
                        (field_id, min_x, min_y, max_x, max_y, width, height, max_cloud_cover,
                         time_from, time_to, mosaicking_order, ndvi_array, ndvi_metadata,
                         season_from, season_to, cloud_cover_pct, candidates_considered,
                         ndvi_mean_at_selection, fetched_at, expires_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, now(), %s)
                    ON CONFLICT (field_id, width, height, max_cloud_cover, time_from, time_to, mosaicking_order)
                        WHERE field_id IS NOT NULL
                    DO UPDATE SET
                        min_x = EXCLUDED.min_x, min_y = EXCLUDED.min_y,
                        max_x = EXCLUDED.max_x, max_y = EXCLUDED.max_y,
                        ndvi_array = EXCLUDED.ndvi_array,
                        ndvi_metadata = COALESCE(EXCLUDED.ndvi_metadata, {schema}.ndvi_cache.ndvi_metadata),
                        season_from = COALESCE(EXCLUDED.season_from, {schema}.ndvi_cache.season_from),
                        season_to = COALESCE(EXCLUDED.season_to, {schema}.ndvi_cache.season_to),
                        cloud_cover_pct = COALESCE(EXCLUDED.cloud_cover_pct, {schema}.ndvi_cache.cloud_cover_pct),
                        candidates_considered = COALESCE(EXCLUDED.candidates_considered, {schema}.ndvi_cache.candidates_considered),
                        ndvi_mean_at_selection = COALESCE(EXCLUDED.ndvi_mean_at_selection, {schema}.ndvi_cache.ndvi_mean_at_selection),
                        fetched_at = now(), expires_at = EXCLUDED.expires_at
                    """,
                    (field_id, min_x, min_y, max_x, max_y, width, height, cloud_cover,
                     time_interval[0], time_interval[1], mosaicking_str, blob, metadata_json,
                     season_from, season_to, cloud_cover_pct, candidates_considered,
                     ndvi_mean_at_selection, expires_at),
                )
            else:
                cur.execute(
                    f"""
                    INSERT INTO {schema}.ndvi_cache
                        (field_id, min_x, min_y, max_x, max_y, width, height, max_cloud_cover,
                         time_from, time_to, mosaicking_order, ndvi_array, ndvi_metadata,
                         season_from, season_to, cloud_cover_pct, candidates_considered,
                         ndvi_mean_at_selection, fetched_at, expires_at)
                    VALUES (NULL, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, now(), %s)
                    ON CONFLICT (min_x, min_y, max_x, max_y, width, height, max_cloud_cover, time_from, time_to, mosaicking_order)
                        WHERE field_id IS NULL
                    DO UPDATE SET
                        ndvi_array = EXCLUDED.ndvi_array,
                        ndvi_metadata = COALESCE(EXCLUDED.ndvi_metadata, {schema}.ndvi_cache.ndvi_metadata),
                        season_from = COALESCE(EXCLUDED.season_from, {schema}.ndvi_cache.season_from),
                        season_to = COALESCE(EXCLUDED.season_to, {schema}.ndvi_cache.season_to),
                        cloud_cover_pct = COALESCE(EXCLUDED.cloud_cover_pct, {schema}.ndvi_cache.cloud_cover_pct),
                        candidates_considered = COALESCE(EXCLUDED.candidates_considered, {schema}.ndvi_cache.candidates_considered),
                        ndvi_mean_at_selection = COALESCE(EXCLUDED.ndvi_mean_at_selection, {schema}.ndvi_cache.ndvi_mean_at_selection),
                        fetched_at = now(), expires_at = EXCLUDED.expires_at
                    """,
                    (min_x, min_y, max_x, max_y, width, height, cloud_cover,
                     time_interval[0], time_interval[1], mosaicking_str, blob, metadata_json,
                     season_from, season_to, cloud_cover_pct, candidates_considered,
                     ndvi_mean_at_selection, expires_at),
                )
    except Exception:
        logger.exception("lopata DB cache write failed - continuing without persisting")
