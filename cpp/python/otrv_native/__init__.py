"""C++ core of Open Tesla Reality View (see cpp/README.md). tsplat uses it automatically when installed."""

from ._core import (
    Camera,
    Splats,
    geo_fit,
    load_splats,
    render,
    sei_decode_proto,
    sei_extract,
    sei_extract_bytes,
    splats_from_arrays,
)

__all__ = [
    "Camera",
    "Splats",
    "geo_fit",
    "load_splats",
    "render",
    "sei_decode_proto",
    "sei_extract",
    "sei_extract_bytes",
    "splats_from_arrays",
]
