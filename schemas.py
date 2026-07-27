from pydantic import BaseModel, Field, field_validator, model_validator

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
    zone_polygon: list[tuple[float, float]] | None = Field(
        None,
        min_length=3,
        description=(
            "Opcjonalny wielokat podpola do faktycznego podzialu (podzbior polygon) - gdy podany, "
            "polygon sluzy tylko do pobrania/rozmiaru rastra NDVI (co pozwala wielu wywolaniom dla "
            "roznych podpol tego samego pola dzielic sie jednym pobranym rastrem, patrz field_id), "
            "a n_zones/geometrie stref sa liczone wzgledem zone_polygon. Brak = dzielimy caly "
            "polygon (dotychczasowe zachowanie)."
        ),
    )
    target_plot_size_ha: float = Field(
        ..., gt=0, description="Docelowa powierzchnia jednej dzialki/strefy w hektarach"
    )
    max_cloud_cover: float = Field(30.0, ge=0, le=100, description="Maksymalne dopuszczalne zachmurzenie sceny w %")
    resolution_m: float = Field(
        10.0, gt=0, le=100, description="Rozdzielczosc analizy NDVI w metrach na piksel (natywna Sentinel-2 to 10m)"
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
    single_zone_override: bool = Field(
        False,
        description=(
            "Jawna zgoda na pominiecie podzialu i zwrocenie polygon (lub zone_polygon, gdy podany) "
            "jako JEDNEJ strefy, niezaleznie od jej powierzchni - jedyny sposob na otrzymanie "
            "strefy wiekszej niz MAX_SUBFIELD_AREA_HA. Dozwolone tylko gdy zone_polygon jest puste "
            "(czyli chodzi o cale zarejestrowane pole, nie recznie narysowany kawalek) - patrz "
            "walidacja ponizej i docstring single_zone_override w field_zones.compute_field_zones."
        ),
    )
    exclusion_polygons: list[list[tuple[float, float]]] | None = Field(
        None,
        description=(
            "Opcjonalna lista wielokatow (kazdy jako pojedynczy pierscien [lon, lat], WGS84) "
            "wylaczonych z probkowania w obrebie tego pola (staw/budynek/linia energetyczna...) - "
            "odejmowane od zone_polygon (lub od calego polygon, gdy zone_polygon nie podano) PRZED "
            "budowa stref, wiec zadna strefa ani sample_point nie powstanie w ich obrebie. Patrz "
            "compute_field_zones's docstring."
        ),
    )

    @field_validator("polygon")
    @classmethod
    def _validate_polygon(cls, value: list[tuple[float, float]]) -> list[tuple[float, float]]:
        return _validate_lonlat_polygon(value)

    @field_validator("zone_polygon")
    @classmethod
    def _validate_zone_polygon(
        cls, value: list[tuple[float, float]] | None
    ) -> list[tuple[float, float]] | None:
        return _validate_lonlat_polygon(value) if value is not None else None

    @field_validator("exclusion_polygons")
    @classmethod
    def _validate_exclusion_polygons(
        cls, value: list[list[tuple[float, float]]] | None
    ) -> list[list[tuple[float, float]]] | None:
        if value is None:
            return None
        for polygon in value:
            _validate_lonlat_polygon(polygon)
        return value

    @model_validator(mode="after")
    def _validate_single_zone_override(self) -> "FieldZonesRequest":
        # "tylko jezeli jest to jedno pole" (only when it's one whole field) - a manually-drawn
        # subfield (zone_polygon set) is exactly the case this is NOT meant for, since that's
        # already the user choosing to divide something smaller than the full field. Enforced
        # server-side (not just in the frontend) since this bypasses the one hard, otherwise
        # unconditional cap in field_zones.py - see MAX_SUBFIELD_AREA_HA's own docstring.
        if self.single_zone_override and self.zone_polygon is not None:
            raise ValueError(
                "single_zone_override jest dozwolone tylko dla calego pola (bez zone_polygon)"
            )
        return self
