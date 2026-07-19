from datetime import datetime, timedelta, timezone

from sentinelhub import (
    BBox,
    CRS,
    DataCollection,
    MimeType,
    MosaickingOrder,
    SentinelHubRequest,
    SHConfig,
)

from config import settings

SH_CONFIG = SHConfig()
SH_CONFIG.sh_client_id = settings.sh_client_id
SH_CONFIG.sh_client_secret = settings.sh_client_secret
SH_CONFIG.sh_base_url = settings.sh_base_url
SH_CONFIG.sh_token_url = settings.sh_token_url

# Sentinel-2 L2A served through the Copernicus Data Space Ecosystem endpoint
# (the built-in DataCollection.SENTINEL2_L2A points at the legacy SH deployment).
S2L2A_CDSE = DataCollection.SENTINEL2_L2A.define_from(
    "s2l2a_cdse", service_url=SH_CONFIG.sh_base_url
)

# Agricultural "traffic light" NDVI ramp (blue = water/non-vegetated, then red -> yellow ->
# green over 0.0-1.0, ColorBrewer RdYlGn) - evenly spaced 0.1-wide stops across the whole
# 0.0-1.0 range so healthy/dense crop (commonly NDVI ~0.6-0.9) is still visibly differentiated,
# unlike the old ramp which only had two, nearly-identical dark-green stops above 0.6.
NDVI_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B04", "B08", "dataMask"] }],
    output: { bands: 4 }
  };
}

var RAMP = [
  [-1.0, 0x08306b],
  [-0.2, 0x2166ac],
  [-0.05, 0x67a9cf],
  [0.0, 0xa50026],
  [0.1, 0xd73027],
  [0.2, 0xf46d43],
  [0.3, 0xfdae61],
  [0.4, 0xfee08b],
  [0.5, 0xffffbf],
  [0.6, 0xd9ef8b],
  [0.7, 0xa6d96a],
  [0.8, 0x66bd63],
  [0.9, 0x1a9850],
  [1.0, 0x006837]
];

function hexToRgb(hex) {
  return [((hex >> 16) & 255) / 255, ((hex >> 8) & 255) / 255, (hex & 255) / 255];
}

function ndviColor(ndvi) {
  for (var i = 1; i < RAMP.length; i++) {
    if (ndvi <= RAMP[i][0]) {
      var v0 = RAMP[i - 1][0], c0 = hexToRgb(RAMP[i - 1][1]);
      var v1 = RAMP[i][0], c1 = hexToRgb(RAMP[i][1]);
      var t = v1 === v0 ? 0 : (ndvi - v0) / (v1 - v0);
      return [
        c0[0] + t * (c1[0] - c0[0]),
        c0[1] + t * (c1[1] - c0[1]),
        c0[2] + t * (c1[2] - c0[2])
      ];
    }
  }
  return hexToRgb(RAMP[RAMP.length - 1][1]);
}

function evaluatePixel(sample) {
  var ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04 + 1e-6);
  ndvi = Math.max(-1, Math.min(1, ndvi));
  var rgb = ndviColor(ndvi);
  return [rgb[0], rgb[1], rgb[2], sample.dataMask];
}
"""


def fetch_ndvi_png(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    width: int = 512,
    height: int = 512,
    max_cloud_cover: float = 30.0,
    search_days: int = 30,
) -> bytes:
    """Fetch a standard-ramp NDVI PNG for the given WGS84 bbox.

    Picks the most recent Sentinel-2 L2A scene within the last `search_days`
    days whose cloud cover is at or below `max_cloud_cover` percent.
    """
    bbox = BBox(bbox=[min_lon, min_lat, max_lon, max_lat], crs=CRS.WGS84)
    time_to = datetime.now(timezone.utc)
    time_from = time_to - timedelta(days=search_days)

    request = SentinelHubRequest(
        evalscript=NDVI_EVALSCRIPT,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=S2L2A_CDSE,
                time_interval=(time_from, time_to),
                mosaicking_order=MosaickingOrder.MOST_RECENT,
                maxcc=max_cloud_cover / 100,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
        bbox=bbox,
        size=(width, height),
        config=SH_CONFIG,
    )

    data = request.get_data(decode_data=False)
    if not data:
        raise LookupError(
            "Brak dostepnych zdjec Sentinel-2 dla podanego obszaru i zakresu dat"
        )
    return data[0].content


# Raw (uncolored) NDVI + validity mask, as FLOAT32, for numeric analysis (e.g. zoning).
NDVI_RAW_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B04", "B08", "dataMask"] }],
    output: { bands: 2, sampleType: "FLOAT32" }
  };
}

function evaluatePixel(sample) {
  var ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04 + 1e-6);
  return [ndvi, sample.dataMask];
}
"""


def fetch_ndvi_array(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    width: int,
    height: int,
    max_cloud_cover: float = 30.0,
    search_days: int = 30,
):
    """Fetch raw NDVI values (not colored) for the given WGS84 bbox.

    :return: numpy array of shape (height, width, 2) with bands [ndvi, dataMask].
    """
    bbox = BBox(bbox=[min_lon, min_lat, max_lon, max_lat], crs=CRS.WGS84)
    time_to = datetime.now(timezone.utc)
    time_from = time_to - timedelta(days=search_days)

    request = SentinelHubRequest(
        evalscript=NDVI_RAW_EVALSCRIPT,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=S2L2A_CDSE,
                time_interval=(time_from, time_to),
                mosaicking_order=MosaickingOrder.MOST_RECENT,
                maxcc=max_cloud_cover / 100,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=(width, height),
        config=SH_CONFIG,
    )

    data = request.get_data(decode_data=True)
    if not data:
        raise LookupError(
            "Brak dostepnych zdjec Sentinel-2 dla podanego obszaru i zakresu dat"
        )
    return data[0]
