from typing import Literal

from pydantic import BaseModel, Field, field_validator

from field_zones import DEFAULT_LINE_SMOOTHING, DEFAULT_MAX_SAMPLE_POINTS_PER_ZONE


def _validate_lonlat_polygon(value: list[tuple[float, float]]) -> list[tuple[float, float]]:
    for lon, lat in value:
        if not (-180 <= lon <= 180) or not (-90 <= lat <= 90):
            raise ValueError(f"Niepoprawny punkt [{lon}, {lat}]: lon musi byc w [-180,180], lat w [-90,90]")
    return value


class NdviRequest(BaseModel):
    polygon: list[tuple[float, float]] = Field(
        ...,
        min_length=3,
        description="Wierzcholki wielokata pola jako pary [lon, lat] (WGS84), min. 3 punkty",
        examples=[[[20.90, 52.15], [21.00, 52.15], [21.00, 52.20], [20.90, 52.20]]],
    )
    width: int = Field(512, gt=0, le=2500, description="Szerokosc obrazu w pikselach")
    height: int = Field(512, gt=0, le=2500, description="Wysokosc obrazu w pikselach")
    max_cloud_cover: float = Field(30.0, ge=0, le=100, description="Maksymalne dopuszczalne zachmurzenie sceny w %")
    field_id: int | None = Field(
        None,
        description=(
            "Opcjonalny identyfikator pola (z kreta) - gdy podany, uzywany jako prostszy "
            "klucz cache NDVI zamiast wspolrzednych bbox (patrz ndvi.py/db_cache.py). "
            "Nieobowiazkowy - inne wywolujace moga go pominac."
        ),
    )

    @field_validator("polygon")
    @classmethod
    def _validate_polygon(cls, value: list[tuple[float, float]]) -> list[tuple[float, float]]:
        return _validate_lonlat_polygon(value)


class FieldZonesRequest(BaseModel):
    polygon: list[tuple[float, float]] = Field(
        ...,
        min_length=3,
        description="Wierzcholki wielokata pola jako pary [lon, lat] (WGS84), min. 3 punkty",
        examples=[[[20.90, 52.15], [21.00, 52.15], [21.00, 52.20], [20.90, 52.20]]],
    )
    target_plot_size_ha: float = Field(
        ..., gt=0, description="Docelowa powierzchnia jednej dzialki/strefy w hektarach"
    )
    max_cloud_cover: float = Field(30.0, ge=0, le=100, description="Maksymalne dopuszczalne zachmurzenie sceny w %")
    resolution_m: float = Field(
        10.0, gt=0, le=100, description="Rozdzielczosc analizy NDVI w metrach na piksel (natywna Sentinel-2 to 10m)"
    )
    strategy: Literal["smooth", "contiguous"] = Field(
        "smooth",
        description=(
            "'smooth': prosty podzial wg wartosci NDVI (k-means) z wygladzaniem malych wysp - "
            "szybki, ale strefy moga wyjsc nierownej wielkosci i nadal jako kilka rozlacznych "
            "plam. "
            "'contiguous': strefy budowane od ziarna (region growing) do rownej liczby pikseli - "
            "kazdy zwrocony wielokat jest jednym spojnym kawalkiem I ma powierzchnie w granicach "
            "MAX_ZONE_SIZE_RATIO wzgledem pozostalych stref (patrz field_zones.py), kosztem "
            "nieco dluzszego przetwarzania."
        ),
    )
    line_smoothing: float = Field(
        DEFAULT_LINE_SMOOTHING,
        gt=0,
        le=10,
        description=(
            "Jak mocno prostowac postrzepione, rastrowe granice stref (Douglas-Peucker w metrach: "
            "resolution_m * line_smoothing) - patrz _simplify_zone_boundaries w field_zones.py. "
            "Wieksze = prostsze/mniej wierzcholkow; powyzej ok. 2.5 dalsze zwiekszanie zwykle nic "
            "juz nie daje, bo liczbe wierzcholkow ogranicza wtedy liczba wezlow sieci (miejsc, "
            "gdzie stykaja sie 3+ strefy), nie ta tolerancja."
        ),
    )
    max_sample_points_per_zone: int = Field(
        DEFAULT_MAX_SAMPLE_POINTS_PER_ZONE,
        ge=0,
        le=15,
        description=(
            "Maks. liczba kandydatow na punkty probne na strefe, dobranych z pikseli poza "
            "skrajnymi percentylami NDVI danej strefy (odrzuca anomalie: kaluze, ugory, sciezki) "
            "i rozlozonych przestrzennie (farthest-point sampling) - front-end wybiera z nich "
            "tyle, ile faktycznie potrzeba (patrz field_zones.py's _select_sample_points)."
        ),
    )
    field_id: int | None = Field(
        None,
        description=(
            "Opcjonalny identyfikator pola (z kreta) - gdy podany, uzywany jako prostszy "
            "klucz cache NDVI zamiast wspolrzednych bbox (patrz ndvi.py/db_cache.py). "
            "Nieobowiazkowy - inne wywolujace moga go pominac."
        ),
    )

    @field_validator("polygon")
    @classmethod
    def _validate_polygon(cls, value: list[tuple[float, float]]) -> list[tuple[float, float]]:
        return _validate_lonlat_polygon(value)
