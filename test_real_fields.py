"""Regression corpus of real fields reported by the user during live debugging sessions.

Not a pytest suite (no test framework is in requirements.txt) - a plain script, run directly
with this project's own .venv interpreter:

    D:\\lopata\\.venv\\Scripts\\python.exe test_real_fields.py

Every time the user pastes a new field's coordinates while debugging a zone-division issue,
add it to FIELDS below (field_id, name, and the WKT exactly as given - usually EPSG:2180, the
Polish CS92 system the krecik/kret backend stores field geometry in). This script reprojects to
lon/lat, calls compute_field_zones at 1.0/2.0/3.0/4.0ha for each field, and reports anything that
would look wrong on the map: a zone over its target_max_ha cap, a zone under 1 pixel's worth of
area (a degenerate sliver), or more zones than the ideal ceil(field_area/target).

Run this after any change to field_zones.py, before pushing - it's the fastest way to catch a
regression against every real field already known to have exposed a bug this session, instead of
re-deriving reproduction steps from scratch each time.

Every field/target combination runs through BOTH of compute_field_zones' independent internal
code paths (whole-field-only, and subfield-scoped with the whole field passed as its own
"subfield") - see run()'s own docstring for why testing only one path (as this script did until
2026-07-29) let a real bug reach production despite looking clean here: krecik's actual frontend
ALWAYS uses the subfield-scoped path, which has its own separate hard-cap-enforcement logic a
whole-field-only test can never exercise.
"""

import math

import numpy as np
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform as shp_transform

import field_zones as fz

FIELDS = [
    {
        "field_id": 125,
        "name": "Lubów 30/3",
        "wkt_2180": (
            "POLYGON ((324361.027212608 415598.909521444, 324516.974389966 415748.871773928, "
            "324449.092149101 415764.806196622, 324402.19872247 415784.752205055, "
            "324307.257265592 415746.10277961, 324284.39177841 415726.951263852, "
            "324265.325678229 415718.064776124, 324237.849447694 415686.340000262, "
            "324229.930653238 415667.993687357, 324210.395696859 415661.37295783, "
            "324089.013784277 415534.450891463, 324163.999114529 415478.700795551, "
            "324076.906411428 415413.09168748, 324084.439280872 415371.810535668, "
            "324103.848393267 415355.349674872, 324135.277834972 415336.974866134, "
            "324140.318357813 415326.159013866, 324129.418966475 415306.24393725, "
            "324065.954262364 415205.181697998, 324056.847139028 415163.548369451, "
            "324065.211879298 415150.187760979, 324068.512546986 415147.383402143, "
            "324070.857351154 415145.591831485, 324111.393044177 415109.597510321, "
            "324150.002976556 415071.680104425, 324230.367338692 415152.616896095, "
            "324467.744200052 415382.47175187, 324447.781459233 415395.00131937, "
            "324437.044690986 415393.548678086, 324414.8847024 415394.541660135, "
            "324409.725012121 415392.262933229, 324400.782946518 415394.11477297, "
            "324357.803705373 415467.791620911, 324355.540395132 415484.307811209, "
            "324359.3231444 415498.471938739, 324376.689899673 415508.101473608, "
            "324367.21220461 415574.561928357, 324363.322768805 415586.211783377, "
            "324361.027212608 415598.909521444))"
        ),
    },
    {
        "field_id": 126,
        "name": "Lubów 457",
        "wkt_2180": (
            "POLYGON ((324038.982255835 415504.054397494, 324034.162587985 415514.937211227, "
            "324003.133237717 415573.514935143, 323970.848867741 415632.429743838, "
            "323945.446289207 415666.777474468, 323940.447572097 415658.728212661, "
            "323939.079777144 415655.93773858, 323856.935430403 415488.020459867, "
            "323942.781455921 415451.54614282, 323895.358744578 415353.163572248, "
            "323907.89894402 415344.674407212, 323922.463565349 415333.628276381, "
            "323944.890055337 415306.809111757, 324041.045668015 415176.990727092, "
            "324045.215511215 415174.144485347, 324046.60086741 415182.60308079, "
            "324051.922833777 415215.010877996, 324061.649709139 415277.829598241, "
            "324069.110282782 415340.139478711, 324069.447621731 415342.144282323, "
            "324070.883426155 415371.096259996, 324068.875250918 415400.415263901, "
            "324068.436069338 415404.849991824, 324056.14786802 415441.987422803, "
            "324047.602188357 415459.659265225, 324043.702442367 415473.478639644, "
            "324032.406059478 415496.886469563, 324038.982255835 415504.054397494))"
        ),
    },
    {
        "field_id": 127,
        "name": "Tworzanice 60",
        "wkt_2180": (
            "POLYGON ((341225.543510484 442832.811918455, 341231.096977784 442817.900409995, "
            "341250.562778395 442619.236414266, 341270.721514557 442379.286074133, "
            "341280.150881308 442273.130430798, 341295.622789465 442096.654235405, "
            "341310.173398224 441928.797951159, 341319.741456965 441723.492023662, "
            "341326.19082706 441582.298483375, 341328.393536598 441551.228145353, "
            "341332.642681766 441512.66210021, 341472.322635413 441526.100053805, "
            "341485.229101552 441355.707215568, 341496.026626458 441214.474034641, "
            "341597.316619148 441251.061727862, 341734.96455396 441302.185596202, "
            "341843.433959253 441341.813719014, 341899.200278858 441361.781697576, "
            "341880.838407588 441651.101294219, 341874.280132456 441652.53088162, "
            "341825.53138669 442313.51892626, 341831.140362025 442315.031425844, "
            "341750.76502391 443372.347240209, 341742.082686158 443371.24682473, "
            "341638.885823719 443337.03493268, 341611.439616753 443327.934750226, "
            "341592.114884649 443322.771780018, 341589.041365795 443322.444092439, "
            "341483.963789018 443246.63128351, 341384.426902973 442996.228035319, "
            "341290.547448624 443013.021852322, 341251.205673206 443019.9898814, "
            "341231.503592529 442963.308495143, 341227.892887183 442951.521833314, "
            "341226.919692514 442933.061079325, 341225.543510484 442832.811918455))"
        ),
    },
    {
        "field_id": 320,
        "name": "Borszyn Wielki 276/4",
        "wkt_2180": (
            "POLYGON ((336762.395278413 424652.30324554, 336768.063243123 424669.800148364, "
            "336771.218084636 424683.362706756, 336796.490243841 424703.070321157, "
            "336816.193804816 424725.673329958, 336783.076098094 424794.425705411, "
            "336735.576531334 424770.024097961, 336718.639012216 424816.53170819, "
            "336645.007121492 424792.098157343, 336735.953562467 424561.803657694, "
            "336771.321932073 424505.556565735, 336764.844786568 424488.100735062, "
            "336796.111129922 424417.014465221, 336892.221247116 424432.342921555, "
            "336918.07253738 424460.759898232, 336929.419868642 424471.041231821, "
            "336899.269088326 424571.752986823, 336866.444909078 424646.609423524, "
            "336840.601554909 424691.029689359, 336820.289990213 424720.378853802, "
            "336799.098955644 424695.736859144, 336817.170025985 424649.513639394, "
            "336809.109325392 424609.136635973, 336792.47407401 424601.077061904, "
            "336787.837187213 424612.14714978, 336777.954176126 424608.783602919, "
            "336762.545907962 424641.404570106, 336762.395278413 424652.30324554))"
        ),
    },
    {
        "field_id": 318,
        "name": "Lubów 155",
        "wkt_2180": (
            "POLYGON ((324490.458073172 413583.312477297, 324495.557573403 413562.188827652, "
            "324491.353949457 413522.367886729, 324485.025900299 413514.336796892, "
            "324471.23458694 413505.178147962, 324468.558239369 413482.731272624, "
            "324473.213021885 413475.139785353, 324477.713991099 413476.767730317, "
            "324483.539757842 413487.544896983, 324485.295910897 413487.30093963, "
            "324495.680451763 413478.351450486, 324505.543712455 413483.914886442, "
            "324506.825369753 413493.564552958, 324492.86783555 413497.834285383, "
            "324489.845723919 413504.193791102, 324491.3201589 413507.47266701, "
            "324493.770964135 413504.659949858, 324497.424725273 413510.128372905, "
            "324501.230956518 413525.281901982, 324506.823995577 413540.970827986, "
            "324506.157289381 413563.123527688, 324507.343740461 413581.891853008, "
            "324507.127866623 413599.729635215, 324513.736075602 413613.625190288, "
            "324513.578055621 413621.075192423, 324508.576416206 413623.772844986, "
            "324492.106608948 413616.860196229, 324485.762453433 413626.654160676, "
            "324486.986099859 413634.255214777, 324494.011829386 413645.785742816, "
            "324493.774731704 413656.955748781, 324485.028332514 413665.073068411, "
            "324486.765432738 413679.515114004, 324494.951835362 413684.541645998, "
            "324502.891252376 413681.743825608, 324512.848705117 413681.037786967, "
            "324513.193899323 413673.385282503, 324509.773108191 413664.484668079, "
            "324513.249793776 413644.582868337, 324516.317670961 413636.453249189, "
            "324528.651834527 413661.137330101, 324534.866642232 413690.123890831, "
            "324541.761980279 413703.815574665, 324535.610900779 413719.685135706, "
            "324544.258153895 413732.823008945, 324528.886280049 413768.902875155, "
            "324520.879333329 413793.805211555, 324512.035067017 413862.736240349, "
            "324503.750776132 413884.903166622, 324457.887273379 413936.295816707, "
            "324439.897618245 413937.401648397, 324225.555907215 414105.945699635, "
            "324109.699837871 414202.443078797, 323952.185233449 414278.43646042, "
            "323836.851665472 414307.896128411, 323841.711028001 414198.331276054, "
            "323891.33499205 414180.667352269, 323921.179709753 414164.83354962, "
            "324007.018973758 414106.665792982, 324019.473084654 414104.306062073, "
            "324026.256066167 414101.74398937, 324043.792015135 414091.58697695, "
            "324093.288620509 414060.228771679, 324144.330591178 414024.910571897, "
            "324231.951431937 413939.856262965, 324267.652654055 413889.542329904, "
            "324293.257625672 413851.712933172, 324302.69454615 413828.300588815, "
            "324309.454256456 413797.726950619, 324318.158649037 413733.186541266, "
            "324320.545402535 413701.573039443, 324314.620771631 413629.48497963, "
            "324303.062665862 413584.796063494, 324285.394640862 413524.635223093, "
            "324256.994713458 413459.722590461, 324214.538280979 413384.535247168, "
            "324280.61297957 413342.423402592, 324300.931892228 413323.770800828, "
            "324314.529481192 413312.188113567, 324360.559446168 413272.229909566, "
            "324394.362778489 413357.972575925, 324433.849789857 413506.959139967, "
            "324442.504490668 413714.320629501, 324397.527505554 413884.606694539, "
            "324396.733018828 413909.110465894, 324406.136187958 413916.849560522, "
            "324410.241767704 413916.613452486, 324444.979604404 413901.112628734, "
            "324454.663508764 413885.524642481, 324461.501859806 413845.352717535, "
            "324466.616893884 413827.557887234, 324517.726708948 413732.886010791, "
            "324515.246454897 413722.582934396, 324511.590065057 413718.384180712, "
            "324501.227607681 413718.715870035, 324499.160855189 413715.255151887, "
            "324504.802960218 413708.219992772, 324501.231195378 413703.6301944, "
            "324478.794423519 413687.311917652, 324477.588701417 413680.290450308, "
            "324476.572030905 413620.581668428, 324484.053036964 413597.42602226, "
            "324490.458073172 413583.312477297))"
        ),
    },
    {
        "field_id": 346,
        "name": "Luboszyce Małe 23",
        "wkt_2180": (
            "POLYGON ((322865.440519103 417673.922003122, 322861.389000997 417675.187102562, "
            "322752.865149537 417614.190231157, 322740.7566008 417607.467917402, "
            "322682.416593051 417577.755072403, 322517.492491266 417495.106438644, "
            "322582.018769102 417432.631044804, 322617.319712635 417411.523863893, "
            "322787.488288525 417382.112880778, 322954.250043809 417496.047364214, "
            "322955.115184978 417500.824148754, 322954.014583046 417502.948604573, "
            "322894.416134821 417617.961418313, 322865.440519103 417673.922003122))"
        ),
    },
    {
        "field_id": 369,
        "name": "Bełcz Wielki 288",
        "wkt_2180": (
            "POLYGON ((317734.037248305 421672.764327361, 317675.732645355 421650.069200154, "
            "317646.952161015 421637.766808175, 317620.026028503 421623.629528131, "
            "317640.30026766 421603.907338141, 317734.612258964 421446.799382376, "
            "317755.048295998 421454.95722094, 317809.562423461 421462.208581606, "
            "317881.413698832 421464.49387734, 317995.665294163 421450.143102633, "
            "318130.319966961 421432.783736277, 318171.107409146 421426.126976953, "
            "318241.417863191 421426.863811485, 318348.61965217 421435.753109713, "
            "318389.43411876 421439.832980609, 318505.355106704 421446.653436433, "
            "318543.866587376 421525.673873149, 318365.234978209 421658.443291507, "
            "318223.489187282 421760.266230517, 318085.900362888 421859.033135546, "
            "318054.561944289 421840.98739975, 317953.581732094 421783.426530764, "
            "317913.163466091 421760.096621441, 317858.355520951 421728.466108583, "
            "317815.934204398 421706.333292574, 317777.829874377 421690.639553031, "
            "317734.037248305 421672.764327361))"
        ),
    },
    {
        "field_id": 383,
        "name": "Luboszyce Małe 45",
        "wkt_2180": (
            "POLYGON ((323028.400660164 417758.43688748, 322872.002245551 417680.760213381, "
            "322870.747278146 417676.718553806, 322902.576543379 417615.240489781, "
            "322960.529017704 417503.409295592, 322963.964102726 417503.122354888, "
            "323042.244512751 417570.541387062, 323043.555141903 417579.380904459, "
            "323049.662144709 417586.59523041, 323066.438402949 417608.589252912, "
            "323077.023516626 417639.135567488, 323078.874714558 417668.481787747, "
            "323081.179396162 417708.318783357, 323083.748915011 417787.96070737, "
            "323047.384797404 417768.634072354, 323028.400660164 417758.43688748))"
        ),
    },
    {
        # Added 2026-07-29 from krecik dev DB farmer 1 ("www www") - see
        # ndvi_stray_multipolygon_sliver memory for the investigation this came from.
        "field_id": 447,
        "name": "Lipno 447",
        "wkt_2180": (
            "POLYGON ((332342.360448816 453401.486351387, 332307.658446121 453310.391158079, "
            "332180.541667742 453339.820048833, 332151.323613704 453346.589736638, "
            "332154.80111032 453355.939085558, 332165.939061084 453385.127085839, "
            "332178.694223693 453418.51157528, 332185.984689535 453437.615535599, "
            "332191.832635605 453452.140719374, 332199.163907008 453471.30409974, "
            "332207.107008451 453492.068581093, 332223.256279697 453534.553634739, "
            "332241.334398605 453581.360852232, 332397.590467685 453545.253272739, "
            "332379.483607924 453498.536434195, 332342.360448816 453401.486351387))"
        ),
    },
    {
        "field_id": 4502,
        "name": "Studzionki 150/2",
        "wkt_2180": (
            "POLYGON ((315379.647073833 412488.615083794, 315371.91148867 412533.491220749, "
            "315302.732023583 412551.645118632, 315303.78224947 412557.845050378, "
            "315195.947359565 412611.124132128, 314999.598081247 412694.778941912, "
            "314983.924805656 412648.420807689, 315164.971805481 412549.875030572, "
            "315168.716071195 412548.408573295, 315218.393433168 412528.809220759, "
            "315236.63492759 412521.673407818, 315287.485517231 412501.559315822, "
            "315379.647073833 412488.615083794))"
        ),
    },
]

TARGET_SIZES_HA = [0.5, 1.0, 2.0, 3.0, 4.0]

# krecik's routes wizard always requests 15 (RoutesComponent.pointsPerSubfield) - the
# DEFAULT_MAX_SAMPLE_POINTS_PER_ZONE=8 this corpus used to run against was never what real
# traffic actually sends, and denser target spacing (chord_len/15 vs chord_len/8) interacts
# differently with the reach-cap/backfill/sanity-check machinery - confirmed 2026-07-27: two
# real bugs on field 127 "Tworzanice 60" @4ha only surfaced at 15 points/zone, not 8.
MAX_SAMPLE_POINTS_PER_ZONE = 15


def _wkt_to_lonlat(wkt_2180: str) -> list[tuple[float, float]]:
    coords_str = wkt_2180.replace("POLYGON ((", "").replace("))", "")
    pairs = [tuple(map(float, p.strip().split())) for p in coords_str.split(",")]
    transformer = Transformer.from_crs("EPSG:2180", "EPSG:4326", always_xy=True)
    return [transformer.transform(x, y) for x, y in pairs]


# A step-to-step ratio check was tried first and rejected: it false-positived on every field at
# small target sizes because of the endpoint-extension pass's normal longer-than-average reach at
# line ends. This diagonal-relative version only flags a step that's a large fraction of the WHOLE
# zone's own size, which a genuine disconnected-cluster jump is and a normal endpoint reach isn't -
# clean on the existing corpus except one legitimate borderline flag on field 127 "Tworzanice 60"
# @2.0ha (a separately-known non-convex field).
SAMPLE_POINT_MAX_STEP_FRACTION_OF_DIAGONAL = 0.5


def _check_sample_point_continuity(result: dict) -> list[str]:
    """Flags a zone whose sample_points contain a jump too large relative to that zone's own
    bounding-box diagonal - see SAMPLE_POINT_MAX_STEP_FRACTION_OF_DIAGONAL's docstring for why
    this is diagonal-relative rather than a plain step-to-step ratio."""
    issues = []
    for feature in result["features"]:
        points = feature["properties"].get("sample_points")
        if not points or len(points) < 2:
            continue
        lons = np.array([p[0] for p in points])
        lats = np.array([p[1] for p in points])
        lat0 = math.radians(float(np.mean(lats)))
        m_per_deg_lat = 111320.0
        m_per_deg_lon = 111320.0 * math.cos(lat0)
        xs = lons * m_per_deg_lon
        ys = lats * m_per_deg_lat

        geom = shape(feature["geometry"])
        minx, miny, maxx, maxy = geom.bounds
        diagonal_m = math.hypot((maxx - minx) * m_per_deg_lon, (maxy - miny) * m_per_deg_lat)
        if diagonal_m < 1e-6:
            continue

        steps = np.hypot(np.diff(xs), np.diff(ys))
        max_step = float(np.max(steps))
        if max_step > SAMPLE_POINT_MAX_STEP_FRACTION_OF_DIAGONAL * diagonal_m:
            issues.append(
                f"zone {feature['properties'].get('zone_id')} sample_points jump {max_step:.0f}m, "
                f"{100 * max_step / diagonal_m:.0f}% of the zone's own {diagonal_m:.0f}m diagonal - "
                "possible disconnected-cluster route"
            )
    return issues


# A single-largest-step check (see _check_sample_point_continuity above) is blind to a path made
# of many medium-sized back-and-forth jumps instead of one dominant outlier - exactly what a
# poorly-ordered point cloud looks like, and exactly what this check is for (see its own docstring
# for the real production bug that exposed the gap: this was checked against the reported points
# and was clean, since no SINGLE step exceeded 50% of the diagonal, even though the overall path
# was a scatter with reported/optimal length ratio ~1.53).
SAMPLE_POINT_MAX_PATH_INEFFICIENCY_RATIO = 1.4


def _check_zone_geometry_validity(result: dict) -> list[str]:
    """Flags a zone whose returned geometry is invalid (self-intersecting) or has a degenerate
    (near-zero-area) interior hole - either renders in Leaflet as a stray extra boundary
    line/loop inside what should be one clean subfield. Added after a real case (field 127
    "Tworzanice 60" @0.5ha): 4 of 176 zones came back invalid, one with a hole 4.5e-18 deg^2 -
    both survived every existing cleanup pass in field_zones.py silently."""
    issues = []
    for feature in result["features"]:
        geom = shape(feature["geometry"])
        zid = feature["properties"].get("zone_id")
        polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        for poly in polys:
            if not poly.is_valid:
                issues.append(f"zone {zid} has an invalid (self-intersecting) geometry")
            if poly.interiors:
                issues.append(f"zone {zid} has {len(poly.interiors)} interior hole(s)")
    return issues


# Matches field_zones.py's own dust threshold (DUST_PART_MAX_PIXELS=2.5 * resolution_m=10 ** 2 =
# 250 m^2): a MultiPolygon secondary part below this should already be stripped by
# _split_dust_parts, which runs after every pixel-mutating pass (rebalance/donation/gap-fill/
# final resimplify). One found ABOVE this threshold is not dust by the module's own definition -
# it's a genuinely separate NDVI-similar patch that survived every dust-strip call anyway, which
# means it was disconnected from the zone's main body by a LATER pass that runs after the dust
# strip closest to it (donation/gap-fill peeling away the connecting pixels), not something
# _split_dust_parts was ever meant to catch. Confirmed directly on field 369 "Bełcz Wielki 288"
# (dev id 2) via the real subfield-scoped kret path: zone 8 @2.0ha had a 797m^2 part 108m from its
# main body; zone 1/zone 5 @4.0ha had 540m^2/1442m^2 parts, both long thin slivers (13-18m wide,
# 76-156m long) - exactly what renders in Leaflet as a disconnected, thin, "loose-ended" line
# floating near a zone instead of a clean filled shape (each part of a MultiPolygon is bound its
# own Leaflet layer independently - see the "Seventh fix" in ndvi_zone_junction_gap_bug memory for
# the same rendering mechanism on a different root cause).
CONTIGUITY_MAX_SECONDARY_PART_M2 = 250.0


def _planar_area_m2(poly) -> float:
    """Equirectangular-approx planar area in m^2 - same local-flat approximation
    _check_sample_point_continuity/_check_sample_point_path_efficiency already use elsewhere in
    this file, good enough at field scale (a few km at most)."""
    minx, miny, maxx, maxy = poly.bounds
    lat0 = math.radians((miny + maxy) / 2.0)
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(lat0)
    scaled = shp_transform(lambda x, y, z=None: (x * m_per_deg_lon, y * m_per_deg_lat), poly)
    return scaled.area


def _check_zone_contiguity(result: dict) -> list[str]:
    """Flags a zone returned as a MultiPolygon with a secondary part bigger than lopata's own
    dust threshold - see CONTIGUITY_MAX_SECONDARY_PART_M2's docstring. Each zone is supposed to be
    exactly one contiguous piece (FieldZonesController's own docstring: "kazda jednym spojnym
    kawalkiem") - a survivor above the dust floor is a real, visible defect, not noise."""
    issues = []
    for feature in result["features"]:
        geom = shape(feature["geometry"])
        if geom.geom_type != "MultiPolygon":
            continue
        zid = feature["properties"].get("zone_id")
        parts = sorted(geom.geoms, key=lambda p: p.area, reverse=True)
        main = parts[0]
        main_m2 = _planar_area_m2(main)
        for part in parts[1:]:
            part_m2 = _planar_area_m2(part)
            if part_m2 > CONTIGUITY_MAX_SECONDARY_PART_M2:
                dist_m = main.distance(part) * 111320.0
                mrr = part.minimum_rotated_rectangle
                issues.append(
                    f"zone {zid} is a MultiPolygon: secondary part {part_m2:.0f}m2, "
                    f"{dist_m:.0f}m from the zone's {main_m2 / 10000:.2f}ha main body - "
                    "not dust-sized, renders as a disconnected floating/loose-ended shape"
                )
    return issues


def _nn_greedy_path_length(xy: np.ndarray) -> float:
    """Independent reference: greedy nearest-neighbor walk (own implementation, not calling into
    field_zones.py) over the same point SET, starting from the point most extreme along the
    cloud's own PCA major axis - see field_zones.py's _farthest_point_fallback for why this choice
    of start point. Used only as a "how good could this path have been" yardstick, not as the
    actual route construction method."""
    n = len(xy)
    if n < 2:
        return 0.0
    centered = xy - xy.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    start = int(np.argmin(centered @ axis))
    remaining = set(range(n))
    remaining.remove(start)
    current = start
    total = 0.0
    while remaining:
        dists = {i: float(np.linalg.norm(xy[i] - xy[current])) for i in remaining}
        nxt = min(dists, key=dists.get)
        total += dists[nxt]
        remaining.remove(nxt)
        current = nxt
    return total


def _check_sample_point_path_efficiency(result: dict) -> list[str]:
    """Flags a zone whose sample_points, connected in the ORDER RETURNED, form a path much longer
    than a reasonable nearest-neighbor ordering of the same points would - catches a generally
    scattered/zigzagging route that _check_sample_point_continuity's single-largest-step check
    cannot (see SAMPLE_POINT_MAX_PATH_INEFFICIENCY_RATIO's own docstring)."""
    issues = []
    for feature in result["features"]:
        points = feature["properties"].get("sample_points")
        if not points or len(points) < 4:
            continue
        lons = np.array([p[0] for p in points])
        lats = np.array([p[1] for p in points])
        lat0 = math.radians(float(np.mean(lats)))
        m_per_deg_lat = 111320.0
        m_per_deg_lon = 111320.0 * math.cos(lat0)
        xy = np.column_stack([lons * m_per_deg_lon, lats * m_per_deg_lat])

        reported_length = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
        best_length = _nn_greedy_path_length(xy)
        if best_length < 1e-6:
            continue
        ratio = reported_length / best_length
        if ratio > SAMPLE_POINT_MAX_PATH_INEFFICIENCY_RATIO:
            issues.append(
                f"zone {feature['properties'].get('zone_id')} sample_points path is {ratio:.2f}x "
                f"longer than a reasonable nearest-neighbor ordering ({reported_length:.0f}m vs "
                f"{best_length:.0f}m) - possible scattered/zigzag route"
            )
    return issues


def _run_one(field: dict, polygon: list[tuple[float, float]], target_ha: float, use_subfield_path: bool) -> bool:
    """Runs one field/target combination through compute_field_zones and prints/returns whether
    it looked clean. use_subfield_path controls which of the two independent code paths inside
    compute_field_zones actually runs (see PATH_LABELS below) - both need covering, not just the
    whole-field-only one this script used to exclusively test."""
    result = fz.compute_field_zones(
        polygon_lonlat=polygon,
        zone_polygon_lonlat=polygon if use_subfield_path else None,
        target_plot_size_ha=target_ha, field_id=field["field_id"],
        max_sample_points_per_zone=MAX_SAMPLE_POINTS_PER_ZONE,
    )
    areas = sorted(f["properties"]["area_ha"] for f in result["features"])
    types = [f["geometry"]["type"] for f in result["features"]]
    field_area_ha = result["field_area_ha"]

    target_max_ha = min(4.0, target_ha * 1.25)
    ideal_n_zones = math.ceil(field_area_ha / target_ha)

    issues = []
    if any(a > target_max_ha + 1e-6 for a in areas):
        issues.append(f"zone over cap ({target_max_ha}ha)")
    if any(a < 0.01 for a in areas):
        issues.append("degenerate near-zero-area zone")
    if result["n_zones"] > ideal_n_zones:
        issues.append(f"more zones than ideal (ideal={ideal_n_zones})")
    issues.extend(_check_sample_point_continuity(result))
    issues.extend(_check_sample_point_path_efficiency(result))
    issues.extend(_check_zone_geometry_validity(result))
    issues.extend(_check_zone_contiguity(result))

    status = "OK" if not issues else "CHECK: " + "; ".join(issues)
    path_label = "subfield" if use_subfield_path else "whole-field"
    print(
        f"  @{target_ha}ha [{path_label}] -> n_zones={result['n_zones']} (ideal={ideal_n_zones}) "
        f"areas={areas} multipolygons={types.count('MultiPolygon')} -- {status}"
    )
    return not issues


def run() -> bool:
    """Returns True if every field/target combination looks clean, False if anything worth a
    second look was found - printed either way.

    Runs EVERY field/target through BOTH of compute_field_zones' independent code paths:
    whole-field-only (zone_polygon_lonlat=None) and subfield-scoped (zone_polygon_lonlat=polygon,
    i.e. the whole field passed as its own "subfield"). These are NOT redundant - they diverge
    deep inside the function (_snap_to_zone_boundary and its own separate rebalance/cap-check only
    run on the subfield-scoped path - see field_zones.py's own "if zone_polygon_lonlat is not
    None" branches). Testing only the whole-field path (as this script did until 2026-07-29) is
    not a superset check: a real, previously-shipped bug (a boundary-resimplify bulge manufacturing
    an extra over-ideal zone) was fixed once on the whole-field path, verified clean here, but
    still reached production because krecik's real frontend (routes.component.ts's
    runLopataZoning()) ALWAYS sends zone_polygon_lonlat - even for the default "whole field as one
    manually-drawn sample" case - so it only ever exercises the subfield-scoped path, which had
    the exact same bug independently in a different function neither this script nor the first fix
    ever touched. See D:\\..\\memory ndvi_zone_construction_fixes_2026-07-28 for the full story."""
    all_ok = True
    for field in FIELDS:
        polygon = _wkt_to_lonlat(field["wkt_2180"])
        print(f"=== field {field['field_id']} \"{field['name']}\" ===")
        for target_ha in TARGET_SIZES_HA:
            all_ok &= _run_one(field, polygon, target_ha, use_subfield_path=False)
            all_ok &= _run_one(field, polygon, target_ha, use_subfield_path=True)
    return all_ok


if __name__ == "__main__":
    ok = run()
    print()
    print("ALL CLEAN" if ok else "SOME COMBINATIONS NEED A LOOK - see CHECK lines above")
