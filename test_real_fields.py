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
"""

from pyproj import Transformer

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
]

TARGET_SIZES_HA = [1.0, 2.0, 3.0, 4.0]


def _wkt_to_lonlat(wkt_2180: str) -> list[tuple[float, float]]:
    coords_str = wkt_2180.replace("POLYGON ((", "").replace("))", "")
    pairs = [tuple(map(float, p.strip().split())) for p in coords_str.split(",")]
    transformer = Transformer.from_crs("EPSG:2180", "EPSG:4326", always_xy=True)
    return [transformer.transform(x, y) for x, y in pairs]


def run() -> bool:
    """Returns True if every field/target combination looks clean, False if anything worth a
    second look was found - printed either way."""
    all_ok = True
    for field in FIELDS:
        polygon = _wkt_to_lonlat(field["wkt_2180"])
        print(f"=== field {field['field_id']} \"{field['name']}\" ===")
        for target_ha in TARGET_SIZES_HA:
            result = fz.compute_field_zones(
                polygon_lonlat=polygon, target_plot_size_ha=target_ha, field_id=field["field_id"]
            )
            areas = sorted(f["properties"]["area_ha"] for f in result["features"])
            types = [f["geometry"]["type"] for f in result["features"]]
            field_area_ha = result["field_area_ha"]
            import math

            target_max_ha = min(4.0, target_ha * 1.25)
            ideal_n_zones = math.ceil(field_area_ha / target_ha)

            issues = []
            if any(a > target_max_ha + 1e-6 for a in areas):
                issues.append(f"zone over cap ({target_max_ha}ha)")
            if any(a < 0.01 for a in areas):
                issues.append("degenerate near-zero-area zone")
            if result["n_zones"] > ideal_n_zones:
                issues.append(f"more zones than ideal (ideal={ideal_n_zones})")

            status = "OK" if not issues else "CHECK: " + "; ".join(issues)
            if issues:
                all_ok = False
            print(
                f"  @{target_ha}ha -> n_zones={result['n_zones']} (ideal={ideal_n_zones}) "
                f"areas={areas} multipolygons={types.count('MultiPolygon')} -- {status}"
            )
    return all_ok


if __name__ == "__main__":
    ok = run()
    print()
    print("ALL CLEAN" if ok else "SOME COMBINATIONS NEED A LOOK - see CHECK lines above")
