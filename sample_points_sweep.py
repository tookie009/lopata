# sample_points_sweep.py - alternatywny algorytm doboru punktow naklec ("boustrophedon"/kosiarka)
#
# Zobacz field_zones.py's SAMPLE_POINT_ALGORITHM (flaga wyboru algorytmu) i D:\lopata\CLAUDE.md
# ("Alternatywny algorytm: boustrophedon sweep") po pelny opis powodu powstania tego pliku.
#
# W skrocie: zamiast jednej najdluzszej cieciwy/sciezki geodezyjnej + zachlannego marszu z
# limitem katu skretu (field_zones._compute_zone_sample_points_legacy - ~725 linii, wiele warstw
# lat poprawek), ten modul dzieli strefe na rownolegle pasy (rzedy) prostopadle do jej krotszej
# osi (z minimum_rotated_rectangle samego wielokata strefy, nie z chmury kandydatow - to samo
# podejscie "ksztalt strefy dyktuje kierunek" co stary algorytm), i w kazdym rzedzie rozklada
# cele rownomiernie wzdluz dluzszej osi, dobierajac najblizszy prawdziwy kandydat NDVI do kazdego
# celu. Rzedy odwiedzane sa na przemian (serpentyna) - polaczenie miedzy kolejnymi rzedami to
# krotki, niemal prostopadly skok, co daje zakrety ~90 stopni na koncach rzedow zamiast dlugich
# zygzakow.
#
# Kluczowa roznica wzgledem starego algorytmu: KAZDY rzad ma wlasny, niezalezny zakres "t" (pozycji
# wzdluz dluzszej osi), wyliczony z przeciecia PASA z prawdziwym wielokatem strefy - dla strefy
# niewypuklej/wygietej rozne rzedy naturalnie obejmuja rozne fragmenty ksztaltu, zamiast zakladac
# jeden sztywny prostokat. To eliminuje systemowa slabosc starego podejscia (patrz pamiec
# ndvi_zone_boundary_invalid_data_gap: sciezka oparta o JEDNA linie moze pokrywac swoj wlasny
# t_coverage w 100%, a mimo to obejmowac tylko ulamek prawdziwego 2D ksztaltu strefy) - tutaj
# pokrycie calej strefy wynika wprost z konstrukcji (rzedy razem obejmuja caly zakres krotszej
# osi), nie jest efektem ubocznym doboru punktow na jednej linii.
#
# Zamierzenie wydajnosciowe: bez grafu widocznosci/Dijkstry (geodezyjne sciezki dla wygietych
# stref), bez wielostartowego zachlannego-najblizszy-sasiad + wspinaczki po kacie skretu (stary
# fallback, ~2.3s/wywolanie zmierzone w poprzednich sesjach) - dobor kandydata do celu to
# przeszukanie liniowe po maks. kilkuset kandydatach w strefie, budowa pasow to stala liczba
# operacji geometrycznych (przeciecie prostokat x wielokat) proporcjonalna do liczby rzedow, nie
# do liczby wierzcholkow/kandydatow w kwadracie.

import logging
import math

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import transform as shp_transform
from shapely.vectorized import contains as _shapely_contains

from field_zones import (
    MIN_PIXELS_FOR_PERCENTILE_FILTER,
    SAMPLE_POINT_MIN_DISTANCE_FROM_BOUNDARY_M,
    SAMPLE_POINT_WORST_PERCENTILE,
    _remove_path_crossings,
    _repair_and_smooth_order,
    _safe_buffer0,
)

logger = logging.getLogger(__name__)


def _principal_axes(utm_geom: Polygon):
    """(major_dir, minor_dir, center) - jednostkowe wektory osi z minimum_rotated_rectangle
    samego wielokata strefy (nie z chmury kandydatow), zeby ulozenie pasow nie bylo skrzywione
    przez to, gdzie akurat sa/nie ma prawdziwych danych NDVI. Dziala dla kazdego wielokata,
    rowniez niewypuklego - MRR zawsze istnieje, nawet jesli nie jest "ciasny" dla mocno
    nieregularnych ksztaltow (a te i tak sa dalej obslugiwane per-rzad przez przeciecie z
    prawdziwym wielokatem, patrz _t_ranges_from_shape)."""
    mrr = utm_geom.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)[:-1]
    if len(coords) != 4:
        minx, miny, maxx, maxy = utm_geom.bounds
        coords = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]
    pts = np.array(coords, dtype=float)
    edges = np.diff(np.vstack([pts, pts[:1]]), axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    len_a, len_b = float(lengths[0]), float(lengths[1])
    dir_a = edges[0] / max(len_a, 1e-9)
    dir_b = edges[1] / max(len_b, 1e-9)
    if len_a >= len_b:
        major_dir, minor_dir = dir_a, dir_b
    else:
        major_dir, minor_dir = dir_b, dir_a
    center = pts.mean(axis=0)
    return major_dir, minor_dir, center


def _row_count(max_points: int, major_extent: float, minor_extent: float) -> int:
    """Liczba rzedow skalowana wspolczynnikiem proporcji strefy - ta sama formula co stara
    (juz zaakceptowana) boustrophedon-banding poprawka w field_zones.py, ale BEZ ograniczenia do
    maks. 2 rzedow: tam to byl ostatni-deska-ratunku reorder juz wybranych 15 punktow, tutaj to
    GLOWNY mechanizm doboru, wiec wiecej rzedow dla naprawde 2D-rozlozonej strefy jest poprawne,
    nie przypadkiem brzegowym do unikania."""
    if minor_extent < 1e-6 or major_extent < 1e-6 or max_points <= 1:
        return 1
    aspect_ratio = major_extent / minor_extent
    n_rows = max(1, round(math.sqrt(max_points / max(aspect_ratio, 1e-6))))
    return int(min(n_rows, max_points))


def _band_strip_polygon(center, major_dir, minor_dir, s_lo, s_hi, t_lo, t_hi) -> Polygon:
    corners_ts = [(t_lo, s_lo), (t_hi, s_lo), (t_hi, s_hi), (t_lo, s_hi)]
    coords = [tuple(center + t * major_dir + s * minor_dir) for t, s in corners_ts]
    return Polygon(coords)


def _t_ranges_from_shape(shape_geom, center, major_dir) -> list[tuple[float, float]]:
    """Zakresy t (pozycji wzdluz dluzszej osi) rzeczywiscie zajmowane przez wielokat strefy w
    obrebie jednego pasa - moze byc wiecej niz jeden zakres (rozlaczne kawalki), gdy strefa jest
    na tyle niewypukla, ze pas przecina ja w kilku miejscach. Posortowane rosnaco po t_lo."""
    if shape_geom is None or shape_geom.is_empty:
        return []
    if shape_geom.geom_type == "Polygon":
        polys = [shape_geom]
    elif shape_geom.geom_type == "MultiPolygon":
        polys = list(shape_geom.geoms)
    elif shape_geom.geom_type == "GeometryCollection":
        polys = []
        for g in shape_geom.geoms:
            if g.geom_type == "Polygon":
                polys.append(g)
            elif g.geom_type == "MultiPolygon":
                polys.extend(g.geoms)
    else:
        return []
    ranges = []
    for p in polys:
        if p.is_empty or p.area <= 1e-9:
            continue
        coords = np.array(p.exterior.coords, dtype=float)
        t_vals = (coords - center) @ major_dir
        ranges.append((float(t_vals.min()), float(t_vals.max())))
    ranges.sort(key=lambda r: r[0])
    return ranges


def _evenly_spaced_targets_over_ranges(t_ranges: list[tuple[float, float]], n: int) -> list[float]:
    """Rozklada n celow rownomiernie wzdluz SUMY dlugosci (moze rozlacznych) zakresow t, pomijajac
    przerwy miedzy nimi proporcjonalnie (parametryzacja po dlugosci luku nad suma odcinkow) -
    zamiast rownomiernie w calym [min(t), max(t)], co uklada by czesc celow w realnej przerwie
    (gdzie strefa w tym pasie w ogole nie istnieje)."""
    if not t_ranges or n <= 0:
        return []
    lengths = [max(0.0, hi - lo) for lo, hi in t_ranges]
    total = sum(lengths)
    if total <= 1e-9:
        mid = (t_ranges[0][0] + t_ranges[0][1]) / 2.0
        return [mid] * n
    positions = [total / 2.0] if n == 1 else [i * total / (n - 1) for i in range(n)]
    targets = []
    for pos in positions:
        acc = 0.0
        placed = False
        for (lo, hi), length in zip(t_ranges, lengths):
            if pos <= acc + length + 1e-9:
                local = min(max(pos - acc, 0.0), length)
                targets.append(lo + local)
                placed = True
                break
            acc += length
        if not placed:
            targets.append(t_ranges[-1][1])
    return targets


def _best_candidate_for_target(
    t_all: np.ndarray, s_all: np.ndarray, t_target: float, s_center: float,
    ndvi_safe: np.ndarray, used: set[int], in_band: np.ndarray,
) -> int | None:
    """Najblizszy (w plaszczyznie t/s) nieuzyty jeszcze kandydat do zadanego celu - najpierw
    "bezpieczny" (poza najgorszym SAMPLE_POINT_WORST_PERCENTILE% NDVI) kandydat W TYM PASIE,
    potem dowolny w tym pasie, potem bezpieczny gdziekolwiek w strefie, na koncu dowolny
    gdziekolwiek - ta sama zasada preferencji co stary algorytm (unikaj najgorszego NDVI, ale nie
    kosztem calkowitego braku punktu), tylko bez nieograniczonego "podazania" za bezpiecznym
    kandydatem (SAFE_PREFERENCE_MAX_REACH_MULTIPLE w starym kodzie istnialo wlasnie zeby to
    ograniczyc) - tutaj kazdy pas ma z gory wlasny, wazki zakres kandydatow, wiec nie ma
    mechanizmu ktory moglby "odjechac" daleko w bok."""

    def _search(pool_mask: np.ndarray, prefer_safe: bool) -> int | None:
        idxs = np.nonzero(pool_mask)[0]
        best_idx = None
        best_d2 = None
        for i in idxs:
            if i in used:
                continue
            if prefer_safe and not ndvi_safe[i]:
                continue
            d2 = (t_all[i] - t_target) ** 2 + (s_all[i] - s_center) ** 2
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best_idx = int(i)
        return best_idx

    idx = _search(in_band, True)
    if idx is not None:
        return idx
    idx = _search(in_band, False)
    if idx is not None:
        return idx
    full_pool = np.ones_like(in_band, dtype=bool)
    idx = _search(full_pool, True)
    if idx is not None:
        return idx
    return _search(full_pool, False)


def _reorder_into_rows(chosen: list[int], t_all: np.ndarray, s_all: np.ndarray, row_edges: np.ndarray) -> list[int]:
    """Po ewentualnym backfillu (patrz compute_zone_sample_points_sweep) przydziela KAZDY wybrany
    punkt (rowniez dolozony w backfillu) do rzedu na podstawie jego wlasnej wspolrzednej s, po
    czym sortuje kazdy rzad po t (na przemian rosnaco/malejaco) - gwarantuje spojna, serpentynowa
    kolejnosc niezaleznie od tego, w jakiej kolejnosci punkty trafily do `chosen`."""
    n_rows = len(row_edges) - 1
    buckets: list[list[int]] = [[] for _ in range(n_rows)]
    for idx in chosen:
        row_i = int(np.searchsorted(row_edges, s_all[idx], side="right") - 1)
        row_i = min(max(row_i, 0), n_rows - 1)
        buckets[row_i].append(idx)
    ordered: list[int] = []
    for row_i in range(n_rows):
        row_pts = buckets[row_i]
        if not row_pts:
            continue
        row_pts.sort(key=lambda i: t_all[i], reverse=(row_i % 2 == 1))
        ordered.extend(row_pts)
    return ordered


def _order_by_major_axis(points_m: np.ndarray, geom, transformer) -> list[int]:
    """Kolejnosc awaryjna, gdy jest za malo kandydatow zeby w ogole ukladac rzedy (< max_points)
    albo geometria strefy jest nieuzywalna - sortuje po rzucie na os glowna (z ksztaltu strefy,
    jesli dostepny, w przeciwnym razie PCA samej chmury kandydatow), potem usuwa ewentualne
    samoprzeciecia."""
    n = len(points_m)
    if n <= 1:
        return list(range(n))
    direction = None
    if geom is not None and not geom.is_empty:
        try:
            utm_geom = _safe_buffer0(shp_transform(transformer.transform, geom))
            if utm_geom.geom_type == "MultiPolygon" and len(utm_geom.geoms):
                utm_geom = max(utm_geom.geoms, key=lambda p: p.area)
            if utm_geom.geom_type == "Polygon" and not utm_geom.is_empty:
                direction, _minor, _center = _principal_axes(utm_geom)
        except Exception as e:
            logger.warning("sweep fallback: could not derive axis from geometry, using PCA: %s", e)
            direction = None
    if direction is None:
        centered = points_m - points_m.mean(axis=0)
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        direction = eigvecs[:, int(np.argmax(eigvals))]
    t = (points_m - points_m.mean(axis=0)) @ direction
    order = list(np.argsort(t))
    return _remove_path_crossings(points_m, order)


def compute_zone_sample_points_sweep(
    ndvi: np.ndarray,
    grid_lon: np.ndarray,
    grid_lat: np.ndarray,
    transformer,
    mask: np.ndarray,
    geom,
    max_points: int,
) -> list[list[float]]:
    """Alternatywa dla field_zones._compute_zone_sample_points_legacy - patrz naglowek tego
    pliku i field_zones.SAMPLE_POINT_ALGORITHM po pelny kontekst. Ten sam kontrakt (sygnatura,
    filtrowanie NDVI, margines od granicy, docelowa liczba punktow) - inny mechanizm doboru
    pozycji/kolejnosci."""
    if max_points <= 0 or not mask.any():
        return []
    values = ndvi[mask]
    lons = grid_lon[mask]
    lats = grid_lat[mask]

    # Margines od granicy strefy - identyczny mechanizm co w starym algorytmie (erozja w UTM,
    # fallback do pelnej geometrii jesli erozja zostawilaby pustke).
    containment_geom = geom
    if geom is not None and not geom.is_empty:
        try:
            utm_geom_raw = shp_transform(transformer.transform, geom)
            eroded_utm = utm_geom_raw.buffer(-SAMPLE_POINT_MIN_DISTANCE_FROM_BOUNDARY_M)
            if not eroded_utm.is_empty:
                containment_geom = shp_transform(
                    lambda x, y: transformer.transform(x, y, direction="INVERSE"), eroded_utm
                )
        except Exception as e:
            logger.warning("sweep: boundary erosion failed, using uneroded zone geometry: %s", e)

    if containment_geom is not None and not containment_geom.is_empty and len(lons):
        inside = _shapely_contains(containment_geom, lons, lats)
        if inside.any():
            values, lons, lats = values[inside], lons[inside], lats[inside]

    if len(lons) == 0:
        return []

    if len(values) >= MIN_PIXELS_FOR_PERCENTILE_FILTER:
        worst_cutoff = np.percentile(values, SAMPLE_POINT_WORST_PERCENTILE)
        ndvi_safe = values >= worst_cutoff
        if not ndvi_safe.any():
            ndvi_safe = np.ones(len(values), dtype=bool)
    else:
        ndvi_safe = np.ones(len(values), dtype=bool)

    xs, ys = transformer.transform(lons, lats)
    points_m = np.column_stack([xs, ys])

    if len(points_m) < 2:
        return [[float(lons[0]), float(lats[0])]]

    if len(points_m) <= max_points:
        # Za malo prawdziwych kandydatow zeby w ogole wypelnic max_points - zwroc wszystkie,
        # uporzadkowane wzdluz osi glownej zamiast w surowej (mozliwie przypadkowej) kolejnosci.
        # Ten sam zaakceptowany wyjatek co stary algorytm ("mniej punktow tylko gdy naprawde za
        # malo kandydatow", patrz pamiec ndvi_sample_point_reach_cap_investigation).
        order = _order_by_major_axis(points_m, geom, transformer)
        return [[float(lons[i]), float(lats[i])] for i in order]

    utm_geom = None
    if geom is not None and not geom.is_empty:
        try:
            utm_geom = _safe_buffer0(shp_transform(transformer.transform, geom))
        except Exception as e:
            logger.warning("sweep: UTM reprojection/buffer0 failed: %s", e)
            utm_geom = None

    if utm_geom is None or utm_geom.is_empty or utm_geom.geom_type not in ("Polygon", "MultiPolygon"):
        order = _order_by_major_axis(points_m, geom, transformer)
        return [[float(lons[i]), float(lats[i])] for i in order]

    utm_geom_main = utm_geom if utm_geom.geom_type == "Polygon" else max(utm_geom.geoms, key=lambda p: p.area)
    major_dir, minor_dir, center = _principal_axes(utm_geom_main)

    rel = points_m - center
    t_all = rel @ major_dir
    s_all = rel @ minor_dir

    poly_coords = np.array(utm_geom_main.exterior.coords, dtype=float)
    poly_rel = poly_coords - center
    poly_t = poly_rel @ major_dir
    poly_s = poly_rel @ minor_dir
    major_extent = float(poly_t.max() - poly_t.min())
    minor_extent = float(poly_s.max() - poly_s.min())

    n_rows = _row_count(max_points, major_extent, minor_extent)
    base = max_points // n_rows
    remainder = max_points - base * n_rows
    row_point_counts = [base + (1 if i < remainder else 0) for i in range(n_rows)]

    # Krancowe krancow rzedow wg KWANTYLI prawdziwych kandydatow (s_all), nie rownego podzialu
    # geometrycznego zasiegu strefy - realne piksele NDVI leza na siatce rastra (10m), wiec
    # rownomierny podzial geometryczny czesto zostawial rzad z zerem/garstka kandydatow (kazdy
    # pas lapal fragment innego wiersza rastra niedopasowany do jego wlasnej siatki), zmuszajac
    # dobor kandydata do "szukania w calej puli" (patrz _best_candidate_for_target) - a to
    # rozbija zalozenie o monotonicznosci w obrebie rzedu. Podzial po kwantylach gwarantuje z
    # grubsza rowna liczbe prawdziwych kandydatow w kazdym rzedzie, wiec fallback poza pas jest
    # rzadki.
    row_edges = np.quantile(s_all, np.linspace(0.0, 1.0, n_rows + 1))
    row_edges = np.unique(row_edges)
    if len(row_edges) < 2:
        row_edges = np.array([float(poly_s.min()), float(poly_s.max())])
    n_rows = len(row_edges) - 1
    if n_rows != len(row_point_counts):
        base = max_points // n_rows
        remainder = max_points - base * n_rows
        row_point_counts = [base + (1 if i < remainder else 0) for i in range(n_rows)]

    chosen: list[int] = []
    used: set[int] = set()

    t_lo_search, t_hi_search = float(poly_t.min()) - 1.0, float(poly_t.max()) + 1.0
    for row_i in range(n_rows):
        n_pts_row = row_point_counts[row_i]
        if n_pts_row <= 0:
            continue
        s_lo, s_hi = float(row_edges[row_i]), float(row_edges[row_i + 1])
        s_center = (s_lo + s_hi) / 2.0
        pad = max((s_hi - s_lo) * 1e-6, 1e-6)
        in_band = (s_all >= s_lo - pad) & (s_all <= s_hi + pad)
        if row_i == n_rows - 1:
            in_band |= s_all >= s_hi  # ostatni rzad zbiera tez skrajnych kandydatow >= gornego kranca

        strip = _band_strip_polygon(center, major_dir, minor_dir, s_lo, s_hi, t_lo_search, t_hi_search)
        try:
            row_shape = utm_geom.intersection(strip)
        except Exception as e:
            logger.warning("sweep: row/zone intersection failed for row %d: %s", row_i, e)
            row_shape = None
        t_ranges = _t_ranges_from_shape(row_shape, center, major_dir)
        if not t_ranges:
            # Brak realnej geometrii w tym pasie (moglo sie zdarzyc dla cienkiego rogu MRR przy
            # mocno nieregularnej strefie) - lepszy przyblizony cel niz brak celu w ogole.
            t_ranges = [(float(poly_t.min()), float(poly_t.max()))]

        targets = _evenly_spaced_targets_over_ranges(t_ranges, n_pts_row)
        if row_i % 2 == 1:
            targets = targets[::-1]

        row_chosen: list[int] = []
        for t_target in targets:
            idx = _best_candidate_for_target(t_all, s_all, t_target, s_center, ndvi_safe, used, in_band)
            if idx is not None:
                used.add(idx)
                row_chosen.append(idx)

        # Dobor per-cel jest zachlanny i NIEZALEZNY miedzy celami w tym samym rzedzie - kolejnosc
        # w jakiej cele zostaly przetworzone (rosnaco/malejaco po t_target) NIE gwarantuje, ze
        # faktycznie WYBRANE kandydaty wyjda w tej samej kolejnosci po ich WLASNYM t (jeden cel
        # moze "przegrac" swojego najblizszego kandydata na rzecz sasiedniego celu i dostac kogos
        # dalszego, w niewlasciwym miejscu sekwencji) - potwierdzone bezposrednio jako zrodlo
        # samoprzeciec. Ten sam wzorzec co stary algorytm: sortuj WYNIK po jego wlasnym t przed
        # potraktowaniem go jako sciezke, nie ufaj kolejnosci przetwarzania celow.
        row_chosen.sort(key=lambda i: t_all[i], reverse=(row_i % 2 == 1))
        chosen.extend(row_chosen)

    if len(chosen) < max_points:
        # Gwarancja dokladnie max_points, kiedy fizycznie istnieje tyle kandydatow (twardy
        # wymog produktowy - patrz pamiec ndvi_sample_point_reach_cap_investigation, "final
        # resolution": "backend zawsze musi zwracac zadana liczbe punktow") - dwuprzebiegowo,
        # tak jak stary algorytm: pierwszy przebieg (powyzej) trzyma sie wlasnego pasa, drugi
        # (tu) dobiera z calej reszty puli bez ograniczenia, preferujac bezpieczne piksele.
        remaining = [i for i in range(len(points_m)) if i not in used]
        remaining.sort(key=lambda i: (0 if ndvi_safe[i] else 1))
        for idx in remaining:
            if len(chosen) >= max_points:
                break
            chosen.append(idx)
            used.add(idx)
        chosen = _reorder_into_rows(chosen, t_all, s_all, row_edges)

    if len(chosen) < 2:
        return [[float(lons[i]), float(lats[i])] for i in chosen]

    # Siatka bezpieczenstwa (nie glowny mechanizm porzadkowania) - kolejnosc serpentynowa jest
    # nie-przecinajaca sie z konstrukcji w typowym przypadku (rozlaczne pasy wzdluz s,
    # monotoniczna kolejnosc w kazdym pasie), ale cel lezacy blisko granicy dwoch pasow moze
    # sporadycznie trafic najblizszego kandydata z sasiedniego pasa, i punkt dolozony w
    # backfillu (patrz wyzej) moze wypasc z porzadku wzgledem swoich bezposrednich sasiadow.
    # `_repair_and_smooth_order` to gotowy, juz sprawdzony w tym pliku wspolny rurociag (usuwanie
    # skrzyzowan + Or-opt na "kolcach" + wygladzanie katow skretu, caly czas w OBU ukladach
    # wspolrzednych, bo reprojekcja moze odwrocic wynik testu prostoty dla niemal-wspolliniowych
    # punktow - patrz pamiec ndvi_sample_point_path_algorithm, poprawka #7) - reuzyty tutaj
    # zamiast pisania drugiej kopii tej samej logiki. Bez tego pelny korpus regresyjny pokazywal
    # realne skrzyzowania w pojedynczych przypadkach ORAZ systematycznie wiecej ostrych zakretow
    # niz stary algorytm (wlasny sweep sortuje kazdy rzad osobno, ale nie ma nic co ogranicza kat
    # NA GRANICY dwoch rzedow - to jest dokladnie luka, ktora _smooth_path_turns wypelnia).
    lonlat_all = np.column_stack([lons, lats])
    smoothed = _repair_and_smooth_order(points_m, lonlat_all, chosen)
    if smoothed is not None:
        chosen = smoothed
    else:
        # _repair_and_smooth_order zwraca None tylko gdy zadna z jego wlasnych napraw nie
        # zostawila sciezki prostej w OBU ukladach - w takim wypadku uzyj przynajmniej
        # zwyklego usuwania skrzyzowan (nie idealne, ale lepsze niz nieuporzadkowana kolejnosc).
        chosen = _remove_path_crossings(points_m, chosen)
        chosen = _remove_path_crossings(lonlat_all, chosen)

    return [[float(lons[i]), float(lats[i])] for i in chosen]
