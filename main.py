from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response

from field_zones import compute_field_zones
from ndvi import fetch_ndvi_png
from schemas import FieldZonesRequest

app = FastAPI(
    title="NDVI API",
    description=(
        "Endpoint generujacy standardowy obraz NDVI (Sentinel-2 L2A, "
        "Copernicus Data Space Ecosystem) przyciety do podanego obszaru (bbox)."
    ),
    version="1.0.0",
)


@app.get(
    "/ndvi",
    summary="Wygeneruj obraz NDVI dla podanego obszaru",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}, "description": "Obraz NDVI w formacie PNG"}},
)
def get_ndvi(
    min_lon: float = Query(..., ge=-180, le=180, description="Minimalna dlugosc geograficzna (WGS84)"),
    min_lat: float = Query(..., ge=-90, le=90, description="Minimalna szerokosc geograficzna (WGS84)"),
    max_lon: float = Query(..., ge=-180, le=180, description="Maksymalna dlugosc geograficzna (WGS84)"),
    max_lat: float = Query(..., ge=-90, le=90, description="Maksymalna szerokosc geograficzna (WGS84)"),
    width: int = Query(512, gt=0, le=2500, description="Szerokosc obrazu w pikselach"),
    height: int = Query(512, gt=0, le=2500, description="Wysokosc obrazu w pikselach"),
    max_cloud_cover: float = Query(30.0, ge=0, le=100, description="Maksymalne dopuszczalne zachmurzenie sceny w %"),
    search_days: int = Query(30, gt=0, le=365, description="Ile dni wstecz szukac najnowszego bezchmurnego zdjecia"),
):
    if min_lon >= max_lon or min_lat >= max_lat:
        raise HTTPException(status_code=400, detail="min_lon/min_lat musza byc mniejsze niz max_lon/max_lat")

    try:
        png_bytes = fetch_ndvi_png(
            min_lon=min_lon,
            min_lat=min_lat,
            max_lon=max_lon,
            max_lat=max_lat,
            width=width,
            height=height,
            max_cloud_cover=max_cloud_cover,
            search_days=search_days,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Blad pobierania danych z Copernicus: {exc}") from exc

    return Response(content=png_bytes, media_type="image/png")


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
    try:
        return compute_field_zones(
            polygon_lonlat=payload.polygon,
            target_plot_size_ha=payload.target_plot_size_ha,
            max_cloud_cover=payload.max_cloud_cover,
            search_days=payload.search_days,
            resolution_m=payload.resolution_m,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Blad przetwarzania: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
