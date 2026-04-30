"""
nlp_engine.py — WesternLocate NLP & Ranking Engine

Pipeline:
  1. Tokenise + clean (stopwords + Porter stemming)
  2. Synonym expansion (Ghanaian food terms, service synonyms)
  3. Category detection — drives live OSM queries in app.py
  4. Reference-location detection — drives proximity scoring
  5. TF-IDF cosine similarity over all candidate place documents
  6. Composite score = 0.5·TFIDF + 0.3·Rating + 0.2·Proximity
  7. Top-N descending

The ranker is source-agnostic: it takes any iterable of place dicts that
follow our schema, so curated entries and live OSM entries score uniformly.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import urllib.parse
from typing import List, Optional, Tuple

log = logging.getLogger("westernlocate.nlp")

# ── Soft dependency: NLTK + scikit-learn (graceful degradation) ──────────────
try:
    import nltk  # type: ignore
    from nltk.corpus import stopwords  # type: ignore
    from nltk.stem import PorterStemmer  # type: ignore
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

    _DATA_DIR = os.path.join(os.path.dirname(__file__), "nltk_data")
    nltk.data.path.insert(0, _DATA_DIR)
    for pkg, sub in [("stopwords", "corpora"), ("punkt", "tokenizers")]:
        try:
            nltk.data.find(f"{sub}/{pkg}")
        except LookupError:
            try:
                nltk.download(pkg, download_dir=_DATA_DIR, quiet=True)
            except Exception:
                pass

    # Final probe — if stopwords still aren't available, fall through to fallback.
    _STOP = set(stopwords.words("english"))
    _STEMMER = PorterStemmer()
    _NLP_AVAILABLE = True
except Exception as _e:
    log.warning("NLP libs unavailable; using keyword-overlap fallback. Reason: %s", _e)
    _STOP = {"the", "a", "an", "i", "is", "are", "to", "of", "for", "in", "on", "at",
             "and", "or", "with", "where", "can", "get", "find", "show", "me"}
    _STEMMER = None
    _NLP_AVAILABLE = False


# ───────────────────────────────────────────────────────────────────────────
# Curated dataset
# ───────────────────────────────────────────────────────────────────────────

_DATASET_PATH = os.path.join(os.path.dirname(__file__), "places_dataset.json")


def _load_dataset() -> List[dict]:
    try:
        with open(_DATASET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for p in data:
            p.setdefault("source", "curated")
            p.setdefault("verified_source", "WesternLocate curated dataset")
        log.info("Loaded %d places from curated dataset.", len(data))
        return data
    except Exception as e:
        log.error("Could not load curated dataset: %s", e)
        return []


CURATED_PLACES: List[dict] = _load_dataset()


# ───────────────────────────────────────────────────────────────────────────
# Reference geography
# ───────────────────────────────────────────────────────────────────────────

TOWN_COORDS = {
    "takoradi":      (4.8941, -1.7536),
    "sekondi":       (4.9342, -1.7082),
    "tarkwa":        (5.3000, -1.9940),
    "axim":          (4.8697, -2.2390),
    "busua":         (4.7430, -1.9990),
    "dixcove":       (4.8148, -1.9688),
    "agona nkwanta": (4.9230, -1.9600),
    "half assini":   (5.0167, -2.8500),
    "prestea":       (5.4340, -2.1460),
    "bogoso":        (5.5330, -2.0500),
    "eikwe":         (4.9600, -2.3100),
    "beyin":         (5.0100, -2.6800),
    "nsuta":         (5.3200, -2.0300),
    "ankasa":        (5.2800, -2.6800),
    "shama":         (5.0167, -1.6333),
    "elubo":         (5.1167, -2.8000),
    "esiama":        (4.9870, -2.4140),
    "butre":         (4.8333, -1.9167),
}
DEFAULT_COORDS = TOWN_COORDS["takoradi"]


# ───────────────────────────────────────────────────────────────────────────
# Synonyms & category mapping
# ───────────────────────────────────────────────────────────────────────────

CATEGORY_TRIGGERS = {
    # Food / restaurant
    "fufu":         ("restaurant", ["fufu", "cassava", "cocoyam", "soup"]),
    "light soup":   ("restaurant", ["light soup", "chicken soup", "meat soup", "clear soup"]),
    "banku":        ("restaurant", ["banku", "corn dough", "tilapia", "okro"]),
    "kenkey":       ("restaurant", ["kenkey", "ga kenkey", "fante kenkey", "fish"]),
    "waakye":       ("restaurant", ["waakye", "rice and beans", "red rice"]),
    "jollof":       ("restaurant", ["jollof", "jollof rice", "party rice", "rice"]),
    "tilapia":      ("restaurant", ["tilapia", "fresh fish", "grilled fish", "seafood"]),
    "kelewele":     ("restaurant", ["kelewele", "spicy plantain", "fried plantain"]),
    "groundnut":    ("restaurant", ["groundnut", "peanut soup", "nkate"]),
    "kontomire":    ("restaurant", ["kontomire", "cocoyam leaves", "spinach stew", "abom"]),
    "omo tuo":      ("restaurant", ["omo tuo", "rice ball"]),
    "pepper soup":  ("restaurant", ["pepper soup", "spicy soup", "prawn soup"]),
    "chop bar":     ("restaurant", ["chop bar", "local restaurant", "canteen"]),
    "restaurant":   ("restaurant", ["restaurant", "eatery", "food", "dining"]),
    "eat":          ("restaurant", ["eat", "food", "meal"]),
    "food":         ("restaurant", ["food", "eat", "meal"]),
    "lunch":        ("restaurant", ["lunch", "food", "restaurant"]),
    "dinner":       ("restaurant", ["dinner", "food", "restaurant"]),
    "breakfast":    ("restaurant", ["breakfast", "food", "restaurant"]),
    "seafood":      ("restaurant", ["seafood", "fish", "tilapia", "prawn", "lobster"]),

    # Hotel
    "hotel":         ("hotel", ["hotel", "lodge", "guest house", "accommodation", "stay"]),
    "lodge":         ("hotel", ["lodge", "hotel", "accommodation"]),
    "guesthouse":    ("hotel", ["guest house", "hotel", "lodging"]),
    "guest house":   ("hotel", ["guest house", "hotel", "lodging"]),
    "accommodation": ("hotel", ["accommodation", "hotel", "stay", "lodging"]),
    "stay":          ("hotel", ["stay", "hotel", "accommodation"]),

    # Health
    "hospital":  ("hospital", ["hospital", "clinic", "medical", "doctor", "emergency"]),
    "clinic":    ("hospital", ["clinic", "hospital", "medical"]),
    "doctor":    ("hospital", ["doctor", "hospital", "clinic"]),
    "emergency": ("hospital", ["emergency", "hospital", "ambulance"]),
    "pharmacy":  ("pharmacy", ["pharmacy", "drug store", "chemist", "medicine"]),
    "drugstore": ("pharmacy", ["pharmacy", "drug store", "chemist"]),

    # Education
    "school":     ("school", ["school", "education", "JHS", "SHS"]),
    "shs":        ("school", ["senior high school", "SHS", "secondary"]),
    "jhs":        ("school", ["junior high school", "JHS"]),
    "university": ("university", ["university", "college", "tertiary", "UMaT", "TTU"]),
    "college":    ("university", ["college", "university", "tertiary"]),

    # Money
    "bank": ("bank", ["bank", "banking", "finance"]),
    "atm":  ("bank", ["ATM", "cash", "withdraw"]),
    "cash": ("bank", ["cash", "ATM", "money"]),

    # Heritage / nature
    "fort":     ("fort", ["fort", "castle", "historical", "heritage"]),
    "castle":   ("fort", ["castle", "fort", "heritage"]),
    "beach":    ("beach", ["beach", "sea", "ocean", "coast", "swim"]),
    "swim":     ("beach", ["swim", "beach", "ocean"]),
    "nature":   ("nature", ["nature", "forest", "park", "reserve", "wildlife"]),
    "forest":   ("nature", ["forest", "rainforest", "park"]),
    "park":     ("nature", ["park", "reserve", "wildlife"]),
    "wildlife": ("nature", ["wildlife", "park", "nature"]),

    # Other services
    "market": ("market", ["market", "shopping", "buy"]),
    "police": ("police", ["police", "station", "security"]),
    "fuel":   ("fuel", ["fuel", "petrol", "gas station", "diesel"]),
    "petrol": ("fuel", ["petrol", "fuel", "gas station"]),

    # Mining / festival (curated only — no clean OSM tag)
    "mining":   ("mining", ["mining", "mine", "gold", "manganese"]),
    "mine":     ("mining", ["mine", "mining", "gold"]),
    "gold":     ("mining", ["gold", "mining", "mine"]),
    "festival": ("festival", ["festival", "kundum", "celebration", "cultural"]),
    "kundum":   ("festival", ["kundum", "ahanta", "nzema", "festival"]),
}

PLACE_INTENT_KEYWORDS = {
    "where", "find", "show", "list", "recommend", "best", "near", "closest",
    "nearby", "nearest", "any", "good", "visit", "go to", "get", "take me",
}


# ───────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────


def _haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _proximity_score(p_lat, p_lon, r_lat, r_lon, max_km=300):
    return max(0.0, 1.0 - _haversine(p_lat, p_lon, r_lat, r_lon) / max_km)


def _rating_score(rating, max_rating=5.0):
    return float(rating or 3.0) / max_rating


def _tokenise(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    if _NLP_AVAILABLE and _STEMMER:
        tokens = [_STEMMER.stem(t) for t in tokens if t not in _STOP and len(t) > 1]
    else:
        tokens = [t for t in tokens if t not in _STOP and len(t) > 1]
    return " ".join(tokens)


def _place_document(p: dict) -> str:
    return " ".join([
        p.get("name", ""),
        p.get("description", ""),
        p.get("category", ""),
        p.get("address", ""),
        p.get("town", ""),
        " ".join(p.get("tags", []) or []),
    ])


# ───────────────────────────────────────────────────────────────────────────
# Public introspection
# ───────────────────────────────────────────────────────────────────────────


def is_place_query(message: str) -> bool:
    msg = message.lower()
    if any(kw in msg for kw in PLACE_INTENT_KEYWORDS):
        return True
    return any(trigger in msg for trigger in CATEGORY_TRIGGERS)


def detect_categories(message: str) -> List[str]:
    msg = message.lower()
    cats: List[str] = []
    for trigger, (cat, _) in CATEGORY_TRIGGERS.items():
        if trigger in msg and cat not in cats:
            cats.append(cat)
    return cats


def expand_query(query: str) -> str:
    q = query.lower()
    extras = []
    for trigger, (_, syns) in CATEGORY_TRIGGERS.items():
        if trigger in q:
            extras.extend(syns)
    return query + " " + " ".join(extras)


def detect_reference_location(query: str) -> Tuple[float, float]:
    q = query.lower()
    for town, coords in TOWN_COORDS.items():
        if town in q:
            return coords
    return DEFAULT_COORDS


# ───────────────────────────────────────────────────────────────────────────
# Core ranker
# ───────────────────────────────────────────────────────────────────────────


def rank_places(
    query: str,
    candidates: Optional[List[dict]] = None,
    top_n: int = 5,
    category_filter: Optional[str] = None,
    weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),
) -> List[dict]:
    if candidates is None:
        candidates = CURATED_PLACES
    if not candidates:
        return []

    if category_filter:
        filtered = [p for p in candidates if p.get("category", "").lower() == category_filter.lower()]
        if filtered:
            candidates = filtered

    expanded = expand_query(query)
    ref = detect_reference_location(query)

    docs = [_tokenise(_place_document(p)) for p in candidates]
    q_tok = _tokenise(expanded)

    sims: List[float]
    if _NLP_AVAILABLE and any(docs) and q_tok:
        try:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95)
            corpus = docs + [q_tok]
            mat = vectorizer.fit_transform(corpus)
            sims = cosine_similarity(mat[-1], mat[:-1]).flatten().tolist()
        except Exception as e:
            log.warning("TF-IDF failure, falling back to overlap: %s", e)
            sims = _keyword_overlap_scores(q_tok, docs)
    else:
        sims = _keyword_overlap_scores(q_tok, docs)

    w_tfidf, w_rating, w_prox = weights
    scored = []
    for i, place in enumerate(candidates):
        tfidf = float(sims[i])
        rating = _rating_score(place.get("rating", 3.0))
        plat, plon = place["latitude"], place["longitude"]
        dist = _haversine(plat, plon, ref[0], ref[1])
        prox = _proximity_score(plat, plon, ref[0], ref[1])

        final = w_tfidf * tfidf + w_rating * rating + w_prox * prox

        out = dict(place)
        out["tfidf_score"] = round(tfidf, 4)
        out["rating_score"] = round(rating, 4)
        out["proximity_score"] = round(prox, 4)
        out["final_score"] = round(final, 4)
        out["distance_km"] = round(dist, 1)
        scored.append(out)

    scored.sort(key=lambda x: x["final_score"], reverse=True)

    relevant = [p for p in scored if p["tfidf_score"] > 0]
    if not relevant:
        relevant = scored
    return relevant[:top_n]


def _keyword_overlap_scores(q_tok: str, docs: List[str]) -> List[float]:
    q_words = set(q_tok.split())
    raw = []
    for doc in docs:
        d_words = set(doc.split())
        overlap = len(q_words & d_words)
        raw.append(overlap / max(len(q_words), 1))
    m = max(raw) if raw else 0
    return [s / m if m > 0 else 0.0 for s in raw]


# ───────────────────────────────────────────────────────────────────────────
# LLM context formatter
# ───────────────────────────────────────────────────────────────────────────


def format_results_for_llm(places: List[dict], query: str) -> str:
    if not places:
        return "No relevant places found in the Western Region for this query."

    lines = [
        f"RANKED RESULTS for query: '{query}'",
        "(Composite score: 50% text relevance + 30% star rating + 20% proximity)",
        "",
    ]
    for i, p in enumerate(places, 1):
        maps_url = (
            "https://www.google.com/maps/search/?api=1&query="
            + urllib.parse.quote_plus(f"{p['name']} {p.get('town','')} Western Region Ghana")
        )
        source_label = "OpenStreetMap (live)" if p.get("source") == "osm" else "Curated dataset"
        lines.append(f"{i}. {p['name']}")
        lines.append(f"   Category: {p['category'].title()}")
        lines.append(f"   Location: {p.get('address', '—')}")
        lines.append(f"   Distance: {p['distance_km']} km from reference point")
        lines.append(f"   Star Rating: {p.get('rating', 'n/a')}/5")
        lines.append(f"   Source: {source_label}")
        lines.append(
            f"   Scores → Relevance: {p['tfidf_score']:.2f} | "
            f"Rating: {p['rating_score']:.2f} | "
            f"Proximity: {p['proximity_score']:.2f} | "
            f"FINAL: {p['final_score']:.2f}"
        )
        lines.append(f"   Description: {p.get('description', '')}")
        if p.get("hours"):
            lines.append(f"   Hours: {p['hours']}")
        if p.get("phone"):
            lines.append(f"   Phone: {p['phone']}")
        lines.append(f"   Maps: {maps_url}")
        lines.append("")
    return "\n".join(lines)
