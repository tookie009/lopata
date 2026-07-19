from typing import Literal

from pydantic import BaseModel, Field, field_validator


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

    @field_validator("polygon")
    @classmethod
    def _validate_polygon(cls, value: list[tuple[float, float]]) -> list[tuple[float, float]]:
        return _validate_lonlat_polygon(value)
