from pydantic import BaseModel, Field, field_validator


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
    search_days: int = Field(30, gt=0, le=365, description="Ile dni wstecz szukac najnowszego bezchmurnego zdjecia")
    resolution_m: float = Field(
        10.0, gt=0, le=100, description="Rozdzielczosc analizy NDVI w metrach na piksel (natywna Sentinel-2 to 10m)"
    )

    @field_validator("polygon")
    @classmethod
    def _validate_polygon(cls, value: list[tuple[float, float]]) -> list[tuple[float, float]]:
        for lon, lat in value:
            if not (-180 <= lon <= 180) or not (-90 <= lat <= 90):
                raise ValueError(f"Niepoprawny punkt [{lon}, {lat}]: lon musi byc w [-180,180], lat w [-90,90]")
        return value
