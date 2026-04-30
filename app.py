#!/usr/bin/env python3
"""
WesternLocate — AI Guide for the Western Region of Ghana
Hybrid Edition: TF-IDF ranking over (curated dataset ⊕ live OpenStreetMap).
"""

import logging
import os
import sqlite3
import sys
import uuid
from datetime import timedelta

from dotenv import load_dotenv
from flask import (
    Flask, flash, g, jsonify, redirect, render_template, request, url_for,
)
from flask_login import (
    LoginManager, UserMixin, current_user, login_required, login_user,
    logout_user,
)
from groq import Groq
from werkzeug.security import check_password_hash, generate_password_hash

from nlp_engine import (
    CURATED_PLACES, detect_categories, detect_reference_location,
    format_results_for_llm, is_place_query, rank_places,
)
from osm_provider import fetch_live_places

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("westernlocate.app")

# ─── App / config ────────────────────────────────────────────────────────────
load_dotenv()

app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.getenv("SECRET_KEY", "dev-secret-key-change-in-production"),
    DATABASE_PATH=os.getenv("DATABASE_PATH", "instance/westernlocate.db"),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    PERMANENT_SESSION_LIFETIME=timedelta(days=14),
    REMEMBER_COOKIE_DURATION=timedelta(days=14),
    REMEMBER_COOKIE_HTTPONLY=True,
    REMEMBER_COOKIE_SAMESITE="Lax",
)
os.makedirs(os.path.dirname(app.config["DATABASE_PATH"]), exist_ok=True)

GROQ_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_KEY:
    log.warning("GROQ_API_KEY is not set. LLM responses will fail until configured.")
groq_client = Groq(api_key=GROQ_KEY) if GROQ_KEY else None

ENABLE_OSM = os.getenv("ENABLE_OSM", "true").lower() == "true"

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"
login_manager.session_protection = "basic"  # 'strong' rotates IDs and broke login flow


# ─── DB ──────────────────────────────────────────────────────────────────────
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(app.config["DATABASE_PATH"])
        g.db.row_factory = sqlite3.Row
    return g.db


def close_db(_=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


app.teardown_appcontext(close_db)


def init_db():
    db = get_db()
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations (id)
        );
        """
    )
    db.commit()
    log.info("Database initialised at %s", app.config["DATABASE_PATH"])


with app.app_context():
    init_db()


# ─── User model ──────────────────────────────────────────────────────────────
class User(UserMixin):
    def __init__(self, id_, username, email):
        self.id = int(id_)
        self.username = username
        self.email = email


@login_manager.user_loader
def load_user(user_id):
    """
    Flask-Login serialises the id to a string in the session cookie. The
    'login-after-logout fails' bug surfaces when this returns None for an
    otherwise valid session — usually because the id type doesn't match.
    """
    try:
        uid = int(user_id)
    except (TypeError, ValueError):
        log.warning("load_user got non-integer id: %r", user_id)
        return None
    db = get_db()
    row = db.execute(
        "SELECT id, username, email FROM users WHERE id = ?", (uid,)
    ).fetchone()
    if not row:
        log.warning("load_user: no user found for id=%s", uid)
        return None
    return User(row["id"], row["username"], row["email"])


# ─── System prompt for the LLM ───────────────────────────────────────────────
SYSTEM_PROMPT = """You are WesternLocate, a professional AI guide for the Western Region of Ghana.

You will sometimes be given RANKED RESULTS from a hybrid retrieval pipeline that combines a curated local dataset with live OpenStreetMap data. When ranked results are provided, you MUST present them in the exact order given — do NOT re-rank.

ABSOLUTE FORMATTING RULES (the chat UI depends on these):

1. Start directly with the first result. Never use greetings like "Hello" or "Hi".

2. For each ranked place, use this exact block, with a BLANK LINE between every line:

**[N]. [Place Name]**

[Category] · [Distance] km away · ⭐ [Rating]/5 · _[Source]_

[Description from the data]

📍 [Address]

🕒 [Hours if available]

📞 [Phone if available]

📊 Scores → Relevance: [tfidf] · Rating: [rating] · Proximity: [proximity] → **Final: [final]**

[🗺️ View on Google Maps]([maps url])

3. Leave TWO blank lines between numbered places.

4. Be factual. Only use information given in the ranked context. Do NOT invent ratings, hours, phone numbers, or fake review quotes. If a field is missing, simply omit that line.

5. Only recommend places in the Western Region of Ghana.

6. End every place-listing response with: "Need more details or directions to any of these? Just ask!"

If no ranked results were attached (general conversation, directions, festival explanations, mining-industry questions), reply naturally and factually. Keep paragraphs short with a blank line between them.

User name: {username}
Today: April 2026"""


# ─── LLM ─────────────────────────────────────────────────────────────────────
def call_llm(history, username, ranked_context=None):
    if not groq_client:
        return "The AI service is not configured. Please set GROQ_API_KEY in the .env file."

    system_msg = {"role": "system", "content": SYSTEM_PROMPT.format(username=username)}

    if ranked_context:
        history = list(history)
        original = history[-1]["content"]
        history[-1] = {
            "role": "user",
            "content": (
                f"User query: {original}\n\n"
                f"--- RANKED RESULTS (present in this exact order) ---\n"
                f"{ranked_context}"
            ),
        }
    full_messages = [system_msg] + history

    try:
        completion = groq_client.chat.completions.create(
            messages=full_messages,
            model="llama-3.3-70b-versatile",
            temperature=0.4,
            max_tokens=2000,
            top_p=0.9,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        log.exception("Groq error: %s", e)
        return "I'm having a connection issue right now. Please try again in a moment. 🌍"


# ─── Hybrid retrieval ────────────────────────────────────────────────────────
def hybrid_rank(user_query: str, top_n: int = 5):
    candidates = list(CURATED_PLACES)
    osm_count = 0

    if ENABLE_OSM:
        cats = detect_categories(user_query)
        if cats:
            ref = detect_reference_location(user_query)
            try:
                live = fetch_live_places(user_query, ref, cats)
                osm_count = len(live)
                seen = {(p["name"].lower().strip(), p.get("town", "").lower().strip())
                        for p in candidates}
                for p in live:
                    key = (p["name"].lower().strip(), p.get("town", "").lower().strip())
                    if key not in seen:
                        candidates.append(p)
                        seen.add(key)
                log.info(
                    "Hybrid: query=%r cats=%s curated=%d osm=%d merged=%d",
                    user_query, cats, len(CURATED_PLACES), osm_count, len(candidates),
                )
            except Exception as e:
                log.exception("OSM fetch failed; continuing with curated only: %s", e)

    ranked = rank_places(user_query, candidates=candidates, top_n=top_n)
    return ranked, osm_count


# ─── Routes ──────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return redirect(url_for("chat") if current_user.is_authenticated else url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("chat"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")

        if not username or not email or not password:
            flash("All fields are required.", "error")
            return render_template("register.html")
        if password != confirm:
            flash("Passwords do not match.", "error")
            return render_template("register.html")
        if len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
            return render_template("register.html")

        db = get_db()
        if db.execute(
            "SELECT id FROM users WHERE username = ? OR email = ?", (username, email)
        ).fetchone():
            flash("Username or email already exists.", "error")
            return render_template("register.html")

        # FIX: pin to pbkdf2:sha256. werkzeug 3 defaults to scrypt, which can
        # silently fail to verify across Python builds on some platforms.
        password_hash = generate_password_hash(password, method="pbkdf2:sha256", salt_length=16)
        db.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (username, email, password_hash),
        )
        db.commit()
        log.info("New user registered: %s", username)
        flash("Account created! Please log in.", "success")
        return redirect(url_for("login"))
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("chat"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        remember = request.form.get("remember") == "on"

        log.info("Login attempt for username=%r remember=%s", username, remember)

        if not username or not password:
            flash("Please enter both your username and password.", "error")
            return render_template("login.html")

        db = get_db()
        user = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()

        if not user:
            log.info("Login failed: no user named %r", username)
            flash("Invalid username or password.", "error")
            return render_template("login.html")

        try:
            ok = check_password_hash(user["password_hash"], password)
        except Exception as e:
            log.exception(
                "Password verification raised for user=%r (likely a legacy hash format): %s",
                username, e,
            )
            flash("Invalid username or password.", "error")
            return render_template("login.html")

        if not ok:
            log.info("Login failed: bad password for user=%r", username)
            flash("Invalid username or password.", "error")
            return render_template("login.html")

        user_obj = User(user["id"], user["username"], user["email"])
        login_user(user_obj, remember=remember)
        log.info("Login successful: user=%s id=%s", username, user["id"])
        return redirect(url_for("chat"))

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    uname = current_user.username
    logout_user()
    log.info("User logged out: %s", uname)
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))


@app.route("/chat")
@login_required
def chat():
    return render_template("chat.html", username=current_user.username)


@app.route("/api/chat", methods=["POST"])
@login_required
def api_chat():
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()
    conversation_id = data.get("conversation_id")

    if not user_message:
        return jsonify({"error": "Message cannot be empty"}), 400

    db = get_db()
    is_new = False
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
        title = user_message[:50] + ("..." if len(user_message) > 50 else "")
        db.execute(
            "INSERT INTO conversations (id, user_id, title) VALUES (?, ?, ?)",
            (conversation_id, current_user.id, title),
        )
        db.commit()
        is_new = True

    db.execute(
        "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
        (conversation_id, "user", user_message),
    )
    db.commit()

    history_rows = db.execute(
        "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY timestamp DESC LIMIT 10",
        (conversation_id,),
    ).fetchall()
    history = [{"role": m["role"], "content": m["content"]} for m in reversed(history_rows)]

    ranked_context = None
    pipeline_meta = {"used": False}
    if is_place_query(user_message):
        ranked, osm_count = hybrid_rank(user_message, top_n=5)
        if ranked:
            ranked_context = format_results_for_llm(ranked, user_message)
            pipeline_meta = {
                "used": True,
                "results": len(ranked),
                "osm_count": osm_count,
                "categories": detect_categories(user_message),
            }

    ai_response = call_llm(history, current_user.username, ranked_context=ranked_context)

    db.execute(
        "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
        (conversation_id, "assistant", ai_response),
    )
    db.execute(
        "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (conversation_id,),
    )
    db.commit()

    return jsonify({
        "response": ai_response,
        "conversation_id": conversation_id,
        "is_new": is_new,
        "pipeline": pipeline_meta,
    })


@app.route("/api/conversations", methods=["GET"])
@login_required
def api_conversations():
    db = get_db()
    rows = db.execute(
        "SELECT id, title, created_at, updated_at FROM conversations "
        "WHERE user_id = ? ORDER BY updated_at DESC",
        (current_user.id,),
    ).fetchall()
    return jsonify([
        {"id": r["id"], "title": r["title"],
         "created_at": r["created_at"], "updated_at": r["updated_at"]}
        for r in rows
    ])


@app.route("/api/conversation/<conv_id>", methods=["GET"])
@login_required
def api_conversation(conv_id):
    db = get_db()
    conv = db.execute(
        "SELECT * FROM conversations WHERE id = ? AND user_id = ?",
        (conv_id, current_user.id),
    ).fetchone()
    if not conv:
        return jsonify({"error": "Conversation not found"}), 404
    msgs = db.execute(
        "SELECT role, content, timestamp FROM messages WHERE conversation_id = ? ORDER BY timestamp ASC",
        (conv_id,),
    ).fetchall()
    return jsonify({
        "id": conv_id, "title": conv["title"],
        "messages": [{"role": m["role"], "content": m["content"], "timestamp": m["timestamp"]} for m in msgs],
    })


@app.route("/api/conversation/<conv_id>", methods=["DELETE"])
@login_required
def delete_conversation(conv_id):
    db = get_db()
    conv = db.execute(
        "SELECT * FROM conversations WHERE id = ? AND user_id = ?",
        (conv_id, current_user.id),
    ).fetchone()
    if not conv:
        return jsonify({"error": "Not found"}), 404
    db.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
    db.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
    db.commit()
    return jsonify({"success": True})


@app.route("/api/places/search", methods=["GET"])
@login_required
def api_search_places():
    """Diagnostic endpoint — useful for facilitator demo."""
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "Provide ?q=your+query"}), 400
    ranked, osm_count = hybrid_rank(query, top_n=10)
    return jsonify({
        "query": query,
        "categories": detect_categories(query),
        "osm_results_fetched": osm_count,
        "results": [
            {
                "name": p["name"], "category": p["category"],
                "town": p.get("town"), "rating": p.get("rating"),
                "distance_km": p["distance_km"],
                "source": p.get("source", "curated"),
                "tfidf_score": p["tfidf_score"],
                "rating_score": p["rating_score"],
                "proximity_score": p["proximity_score"],
                "final_score": p["final_score"],
            }
            for p in ranked
        ],
    })


@app.route("/healthz")
def healthz():
    return jsonify({"ok": True, "osm_enabled": ENABLE_OSM, "curated_places": len(CURATED_PLACES)})


if __name__ == "__main__":
    log.info("🚀 WesternLocate (Hybrid) | http://127.0.0.1:5000  | OSM=%s", ENABLE_OSM)
    app.run(debug=os.getenv("FLASK_DEBUG", "true").lower() == "true", host="0.0.0.0", port=5000)
