"""
osm_provider.py — Live OpenStreetMap data provider for WesternLocate.

Architecture decision: we use TWO complementary OSM APIs.

  • Nominatim (https://nominatim.openstreetmap.org)
        Purpose: geocoding only — turn "Takoradi" into (4.89, -1.75).
        Limit:   1 request/second, real User-Agent required, no bulk use.

  • Overpass API (https://overpass-api.de/api/interpreter)
        Purpose: structured queries by tag — "all hospitals within 30km of X".
        Limit:   no hard cap, but per-query timeout (~25s); be polite.

Both calls are cached in-memory with a TTL. On any failure we return [] and
the caller falls back to the curated dataset. The hybrid ranker treats this
as "no OSM signal", not as an error.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("westernlocate.osm")

# ───────────────────────────────────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────────────────────────────────

USER_AGENT = os.getenv(
    "OSM_USER_AGENT",
    "WesternLocate/1.0 (Western Region Ghana academic project; contact: austinbediako4@gmail.com)",
)

NOMINATIM_BASE = "https://nominatim.openstreetmap.org"
OVERPASS_BASE = "https://overpass-api.de/api/interpreter"

NOMINATIM_TIMEOUT = 8
OVERPASS_TIMEOUT = 25
CACHE_TTL_SECONDS = 60 * 60

# Western Region of Ghana bbox (south, west, north, east)
WR_BBOX = (4.55, -3.27, 6.10, -1.45)

OSM_CATEGORY_TAGS: Dict[str, List[Tuple[str, str]]] = {
    "hospital":   [("amenity", "hospital"), ("amenity", "clinic"), ("healthcare", "hospital")],
    "restaurant": [("amenity", "restaurant"), ("amenity", "fast_food"), ("amenity", "cafe")],
    "hotel":      [("tourism", "hotel"), ("tourism", "guest_house"), ("tourism", "hostel"), ("tourism", "motel")],
    "school":     [("amenity", "school"), ("amenity", "college")],
    "university": [("amenity", "university")],
    "bank":       [("amenity", "bank"), ("amenity", "atm")],
    "fort":       [("historic", "fort"), ("historic", "castle"), ("historic", "ruins")],
    "beach":      [("natural", "beach")],
    "market":     [("amenity", "marketplace")],
    "police":     [("amenity", "police")],
    "pharmacy":   [("amenity", "pharmacy")],
    "fuel":       [("amenity", "fuel")],
    "nature":     [("leisure", "nature_reserve"), ("boundary", "national_park"), ("tourism", "attraction")],
}

# ───────────────────────────────────────────────────────────────────────────
# TTL cache + Nominatim rate limiter
# ───────────────────────────────────────────────────────────────────────────


class _TTLCache:
    def __init__(self, ttl: int):
        self.ttl = ttl
        self._store: Dict[str, Tuple[float, object]] = {}
        self._lock = threading.Lock()

    def get(self, key: str):
        with self._lock:
            entry = self._store.get(key)
            if not entry:
                return None
            ts, value = entry
            if time.time() - ts > self.ttl:
                self._store.pop(key, None)
                return None
            return value

    def set(self, key: str, value):
        with self._lock:
            self._store[key] = (time.time(), value)


_cache = _TTLCache(CACHE_TTL_SECONDS)


class _RateLimiter:
    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._last_call = 0.0
        self._lock = threading.Lock()

    def wait(self):
        with self._lock:
            now = time.time()
            wait_for = self.min_interval - (now - self._last_call)
            if wait_for > 0:
                time.sleep(wait_for)
            self._last_call = time.time()


_nominatim_limiter = _RateLimiter(min_interval=1.05)


# ───────────────────────────────────────────────────────────────────────────
# HTTP helpers
# ───────────────────────────────────────────────────────────────────────────


def _http_get(url: str, timeout: int) -> Optional[dict]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        log.warning("HTTP GET failed for %s: %s", url[:120], e)
        return None


def _http_post(url: str, data: str, timeout: int) -> Optional[dict]:
    req = urllib.request.Request(
        url,
        data=data.encode("utf-8"),
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        log.warning("HTTP POST failed for %s: %s", url, e)
        return None


# ───────────────────────────────────────────────────────────────────────────
# Geocoding (Nominatim)
# ───────────────────────────────────────────────────────────────────────────


def geocode(place: str) -> Optional[Tuple[float, float]]:
    if not place or not place.strip():
        return None

    key = f"geocode::{place.strip().lower()}"
    cached = _cache.get(key)
    if cached is not None:
        return cached if cached != "MISS" else None

    s, w, n, e = WR_BBOX
    params = urllib.parse.urlencode({
        "q": f"{place}, Western Region, Ghana",
        "format": "json",
        "limit": "1",
        "viewbox": f"{w},{n},{e},{s}",
        "bounded": "1",
    })
    url = f"{NOMINATIM_BASE}/search?{params}"

    _nominatim_limiter.wait()
    data = _http_get(url, NOMINATIM_TIMEOUT)
    if not data:
        _cache.set(key, "MISS")
        return None
    try:
        first = data[0]
        coords = (float(first["lat"]), float(first["lon"]))
        _cache.set(key, coords)
        return coords
    except (IndexError, KeyError, ValueError):
        _cache.set(key, "MISS")
        return None


# ───────────────────────────────────────────────────────────────────────────
# Category search (Overpass)
# ───────────────────────────────────────────────────────────────────────────


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _build_overpass_query(category: str, center: Tuple[float, float], radius_km: float) -> Optional[str]:
    tags = OSM_CATEGORY_TAGS.get(category)
    if not tags:
        return None
    radius_m = int(radius_km * 1000)
    lat, lon = center
    parts = []
    for k, v in tags:
        parts.append(f'  node["{k}"="{v}"]["name"](around:{radius_m},{lat},{lon});')
        parts.append(f'  way["{k}"="{v}"]["name"](around:{radius_m},{lat},{lon});')
        parts.append(f'  relation["{k}"="{v}"]["name"](around:{radius_m},{lat},{lon});')
    body = "\n".join(parts)
    return f"[out:json][timeout:{OVERPASS_TIMEOUT}];\n(\n{body}\n);\nout center tags;"


def _normalize_overpass_element(el: dict, category: str) -> Optional[dict]:
    tags = el.get("tags") or {}
    name = tags.get("name")
    if not name:
        return None
    if "lat" in el and "lon" in el:
        lat, lon = el["lat"], el["lon"]
    elif "center" in el and "lat" in el["center"] and "lon" in el["center"]:
        lat, lon = el["center"]["lat"], el["center"]["lon"]
    else:
        return None

    address_parts = [
        tags.get("addr:housename"),
        tags.get("addr:street"),
        tags.get("addr:suburb") or tags.get("addr:village"),
        tags.get("addr:city") or tags.get("addr:town"),
    ]
    address = ", ".join(p for p in address_parts if p) or "Western Region, Ghana"

    description_bits = []
    if tags.get("description"):
        description_bits.append(tags["description"])
    if tags.get("cuisine"):
        description_bits.append(f"Cuisine: {tags['cuisine'].replace(';', ', ')}.")
    if tags.get("opening_hours"):
        description_bits.append(f"Hours: {tags['opening_hours']}.")
    if not description_bits:
        description_bits.append(f"{category.title()} in the Western Region of Ghana, listed on OpenStreetMap.")
    description = " ".join(description_bits)

    return {
        "id": f"osm-{el.get('type','node')[0]}{el.get('id','')}",
        "name": name,
        "category": category,
        "town": tags.get("addr:city") or tags.get("addr:town") or tags.get("addr:village") or "",
        "address": address,
        "latitude": float(lat),
        "longitude": float(lon),
        "rating": 3.5,
        "phone": tags.get("phone") or tags.get("contact:phone") or "",
        "hours": tags.get("opening_hours") or "",
        "description": description,
        "tags": [category]
                + ([tags["cuisine"].replace(";", ",")] if tags.get("cuisine") else [])
                + ([tags["amenity"]] if tags.get("amenity") else [])
                + ([tags["tourism"]] if tags.get("tourism") else []),
        "verified_source": "OpenStreetMap (live)",
        "source": "osm",
    }


def search_category(
    category: str,
    center: Tuple[float, float],
    radius_km: float = 30.0,
    max_results: int = 25,
) -> List[dict]:
    if category not in OSM_CATEGORY_TAGS:
        return []

    cache_key = f"overpass::{category}::{center[0]:.3f},{center[1]:.3f}::{radius_km}"
    cached = _cache.get(cache_key)
    if cached is not None:
        log.debug("Overpass cache hit for %s", cache_key)
        return cached

    query = _build_overpass_query(category, center, radius_km)
    if not query:
        return []

    body = urllib.parse.urlencode({"data": query})
    log.info("Overpass query: category=%s center=%s radius=%skm", category, center, radius_km)
    payload = _http_post(OVERPASS_BASE, body, OVERPASS_TIMEOUT)

    if not payload or "elements" not in payload:
        _cache.set(cache_key, [])
        return []

    places = []
    for el in payload["elements"]:
        norm = _normalize_overpass_element(el, category)
        if norm:
            if _haversine_km(norm["latitude"], norm["longitude"], *center) > radius_km * 1.3:
                continue
            places.append(norm)

    seen = set()
    unique = []
    for p in places:
        key = (p["name"].lower().strip(), p["town"].lower().strip())
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)

    unique = unique[:max_results]
    _cache.set(cache_key, unique)
    log.info("Overpass returned %d places for %s", len(unique), category)
    return unique


def fetch_live_places(query_text: str, center: Tuple[float, float], categories: List[str]) -> List[dict]:
    out: List[dict] = []
    for cat in categories:
        try:
            out.extend(search_category(cat, center))
        except Exception as e:
            log.exception("Overpass search failed for %s: %s", cat, e)
    return out
