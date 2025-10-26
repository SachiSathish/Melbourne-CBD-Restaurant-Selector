# geo.py
import re
from typing import Optional, Tuple

# Edit/extend these as you like
LANDMARKS = {
    "flinders street station": (-37.8183, 144.9671),
    "southern cross station":  (-37.8183, 144.9526),
    "melbourne central":       (-37.8105, 144.9631),
    "parliament station":      (-37.8110, 144.9730),
    "state library":           (-37.8099, 144.9656),
    "federation square":       (-37.8179, 144.9691),
    "bourke street mall":      (-37.8142, 144.9632),
}

def resolve_location(text: str) -> Optional[Tuple[float,float,str]]:
    q = text.lower()
    for name, (lat, lon) in LANDMARKS.items():
        if name in q:
            return lat, lon, name  # (lat, lon, canonical_name)
    return None
