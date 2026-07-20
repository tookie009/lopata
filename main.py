import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

from field_zones import compute_field_zones
from ndvi import fetch_ndvi_png
from schemas import FieldZonesRequest, NdviRequest

app = FastAPI(
    title="NDVI API",
    description=(
        "Endpoint generujacy standardowy obraz NDVI (Sentinel-2 L2A, "
        "Copernicus Data Space Ecosystem) przyciety do podanego obszaru (bbox)."
    ),
    version="1.0.0",
)

# Full request/response JSON for /field-zones, one plain-text block per call - separate from
# uvicorn's own access log (which only has the status line, not the bodies) so a request that
# produced a bad geometry can be copy-pasted straight out of this file and handed over as-is,
# instead of re-typing/re-fetching it from the browser dev tools every time.
_field_zones_logger = logging.getLogger("field_zones_requests")
_field_zones_logger.setLevel(logging.INFO)
_field_zones_logger.propagate = False
if not _field_zones_logger.handlers:
    _log_path = Path(__file__).parent / "logs" / "field_zones_requests.log"
    _log_path.parent.mkdir(parents=True, exist_ok=True)
    _handler = logging.FileHandler(_log_path, encoding="utf-8")
    _handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    _field_zones_logger.addHandler(_handler)


def _ndvi_metadata_headers(metadata: dict) -> dict[str, str]:
    """Flattens NDVI metadata (acquisition date, search window, ...) into response headers,
    for endpoints (like /ndvi) whose body is a raw image and can't carry it as JSON."""
    headers = {
        "X-NDVI-Acquisition-Dates": ",".join(metadata["acquisition_dates"]),
        "X-NDVI-Time-From": metadata["time_window_searched"]["from"],
        "X-NDVI-Time-To": metadata["time_window_searched"]["to"],
        "X-NDVI-Max-Cloud-Cover": str(metadata["max_cloud_cover"]),
        "X-NDVI-Mosaicking-Order": metadata["mosaicking_order"],
        "X-NDVI-Data-Collection": metadata["data_collection"],
    }
    if metadata["acquired"] is not None:
        headers["X-NDVI-Acquired"] = metadata["acquired"]
    if metadata.get("cloud_cover") is not None:
        headers["X-NDVI-Cloud-Cover"] = str(metadata["cloud_cover"])
    if metadata.get("candidates_considered") is not None:
        headers["X-NDVI-Candidates-Considered"] = str(metadata["candidates_considered"])
    if metadata.get("ndvi_mean_at_selection") is not None:
        headers["X-NDVI-Mean-At-Selection"] = str(metadata["ndvi_mean_at_selection"])
    return headers


@app.post(
    "/ndvi",
    summary="Wygeneruj obraz NDVI dla podanego wielokata pola",
    response_class=Response,
    responses={
        200: {
            "content": {"image/png": {}},
            "description": (
                "Obraz NDVI w formacie PNG, przyciety do dokladnych krawedzi podanego "
                "wielokata pola (pozostale piksele przezroczyste), dla terminu z ostatniego "
                "sezonu wegetacyjnego o najlepszej roslinnosci w obrebie pola. Metadane (data "
                "zdjecia satelitarnego, ile terminow rozwazono, ...) sa dolaczone jako naglowki "
                "odpowiedzi X-NDVI-*."
            ),
        }
    },
)
def get_ndvi(payload: NdviRequest):
    try:
        png_bytes, metadata = fetch_ndvi_png(
            polygon_lonlat=payload.polygon,
            width=payload.width,
            height=payload.height,
            max_cloud_cover=payload.max_cloud_cover,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Blad pobierania danych z Copernicus: {exc}") from exc

    return Response(content=png_bytes, media_type="image/png", headers=_ndvi_metadata_headers(metadata))


@app.post(
    "/field-zones",
    summary="Podziel pole na strefy/dzialki na podstawie poziomow NDVI",
    description=(
        "Przyjmuje wielokat pola i docelowa wielkosc dzialki (ha). Pobiera NDVI dla "
        "obszaru pola, grupuje piksele o podobnym poziomie NDVI w klastry (1D k-means) "
        "tak, by liczba stref odpowiadala field_area_ha / target_plot_size_ha, a nastepnie "
        "zamienia kazdy klaster na geometrie (przycieta do granic pola). Zwraca GeoJSON "
        "FeatureCollection ze strefami posortowanymi rosnaco wg sredniego NDVI."
    ),
)
def post_field_zones(payload: FieldZonesRequest):
    _field_zones_logger.info(
        "REQUEST /field-zones\n%s", json.dumps(payload.model_dump(), ensure_ascii=False, indent=2)
    )
    try:
        result = compute_field_zones(
            polygon_lonlat=payload.polygon,
            target_plot_size_ha=payload.target_plot_size_ha,
            max_cloud_cover=payload.max_cloud_cover,
            resolution_m=payload.resolution_m,
            strategy=payload.strategy,
            line_smoothing=payload.line_smoothing,
        )
        _field_zones_logger.info(
            "RESPONSE /field-zones\n%s", json.dumps(result, ensure_ascii=False, indent=2)
        )
        return result
    except LookupError as exc:
        _field_zones_logger.info("ERROR /field-zones 404: %s", exc)
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        _field_zones_logger.info("ERROR /field-zones 400: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        _field_zones_logger.info("ERROR /field-zones 502: %s", exc)
        raise HTTPException(status_code=502, detail=f"Blad przetwarzania: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
