"""
app.py — Emotional AI Task & Research Agent — REST API (Flask)
FlutterFlow মোবাইল অ্যাপ থেকে সরাসরি কল করার জন্য বানানো JSON API।
Render.com-এ হোস্ট হওয়ার জন্য ডিজাইন করা।

Start Command (Render): gunicorn app:app --workers 1 --threads 8 --timeout 60
(--workers 1 রাখা must — নাহলে in-memory reminder timer আলাদা প্রসেসে ভাগ হয়ে যাবে
 এবং ঠিকমতো ফায়ার নাও করতে পারে। একাধিক worker দরকার হলে scheduler-কে
 আলাদা background worker service-এ সরাতে হবে।)
"""

import os
import io
import json
import time
import base64
import logging
import threading
from datetime import datetime, timezone

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

from groq import Groq
from supabase import create_client, Client

try:
    from googlesearch import search as google_search_lib
    GOOGLE_LIB_AVAILABLE = True
except ImportError:
    GOOGLE_LIB_AVAILABLE = False

try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False

import firebase_admin
from firebase_admin import credentials, messaging

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("emotional-ai-agent-api")

# ============================================================
# ENV VARS
# ============================================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
SUPABASE_BUCKET = os.environ.get("SUPABASE_STORAGE_BUCKET", "user-images")
FIREBASE_CREDENTIALS_JSON = os.environ.get("FIREBASE_CREDENTIALS_JSON")
GROQ_VISION_MODEL_ID = os.environ.get("GROQ_VISION_MODEL_ID", "qwen/qwen3.6-27b")
GROQ_TEXT_MODEL_ID = os.environ.get("GROQ_TEXT_MODEL_ID", "openai/gpt-oss-120b")
API_SECRET_KEY = os.environ.get("API_SECRET_KEY")  # ঐচ্ছিক কিন্তু strongly recommended — নিচে auth অংশে দেখুন

_missing = [k for k, v in {"GROQ_API_KEY": GROQ_API_KEY, "SUPABASE_URL": SUPABASE_URL,
                            "SUPABASE_KEY": SUPABASE_KEY}.items() if not v]
if _missing:
    raise RuntimeError(f"প্রয়োজনীয় Environment Variable পাওয়া যায়নি: {', '.join(_missing)}")

if not API_SECRET_KEY:
    logger.warning(
        "⚠️ API_SECRET_KEY সেট করা নেই — এই মুহূর্তে যে কেউ আপনার API URL জানলেই "
        "ব্যবহার করতে পারবে। পাবলিক লঞ্চের আগে অবশ্যই সেট করুন।"
    )

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
groq_client: Groq = Groq(api_key=GROQ_API_KEY)

firebase_app = None
if FIREBASE_CREDENTIALS_JSON:
    try:
        cred = credentials.Certificate(json.loads(FIREBASE_CREDENTIALS_JSON))
        firebase_app = firebase_admin.initialize_app(cred)
    except Exception as e:
        logger.error(f"Firebase init failed: {e}")
else:
    logger.warning("FIREBASE_CREDENTIALS_JSON সেট নেই — পুশ নোটিফিকেশন কাজ করবে না।")

app = Flask(__name__)
CORS(app)  # FlutterFlow/মোবাইল অ্যাপ থেকে ক্রস-অরিজিন রিকোয়েস্ট অনুমোদন করে


# ============================================================
# সাধারণ AUTH — শুধু আপনার অ্যাপ যেন কল করতে পারে
# ============================================================

def require_api_key(f):
    from functools import wraps

    @wraps(f)
    def wrapper(*args, **kwargs):
        if API_SECRET_KEY:
            provided = request.headers.get("X-API-Key")
            if provided != API_SECRET_KEY:
                return jsonify({"error": "Unauthorized — ভুল বা অনুপস্থিত API key"}), 401
        return f(*args, **kwargs)
    return wrapper


# ============================================================
# TASK SCHEDULER (আগের মতোই, শুধু Streamlit নির্ভরতা সরানো হয়েছে)
# ============================================================

class TaskScheduler:
    def __init__(self):
        self._timers: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

    def schedule(self, task_id: str, run_at: datetime, payload: dict):
        delay = max((run_at - datetime.now(timezone.utc)).total_seconds(), 0)
        with self._lock:
            self._cancel_locked(task_id)
            timer = threading.Timer(delay, self._fire, args=(task_id, payload))
            timer.daemon = True
            self._timers[task_id] = timer
            timer.start()
        logger.info(f"Task {task_id} scheduled — fires in {delay:.1f}s")

    def cancel(self, task_id: str):
        with self._lock:
            self._cancel_locked(task_id)

    def _cancel_locked(self, task_id: str):
        t = self._timers.pop(task_id, None)
        if t:
            t.cancel()

    def _fire(self, task_id: str, payload: dict):
        logger.info(f"🔔 Firing task {task_id}")
        try:
            send_fcm_notification(
                fcm_token=payload["fcm_token"], title=payload.get("title", "⏰ রিমাইন্ডার"),
                body=payload.get("body", ""), data=payload.get("data", {}),
            )
            supabase.table("user_tasks").update({"status": "sent"}).eq("id", task_id).execute()
        except Exception as e:
            logger.error(f"Task {task_id} firing failed: {e}")
        finally:
            with self._lock:
                self._timers.pop(task_id, None)


scheduler = TaskScheduler()


def _rearm_pending_tasks():
    try:
        now_iso = datetime.now(timezone.utc).isoformat()
        res = (supabase.table("user_tasks").select("*")
               .eq("status", "scheduled").gt("run_at", now_iso).execute())
        for row in res.data:
            scheduler.schedule(row["id"], datetime.fromisoformat(row["run_at"]), {
                "fcm_token": row["fcm_token"], "title": "⏰ রিমাইন্ডার",
                "body": row["title"], "data": {"task_id": row["id"], "type": "reminder"},
            })
        logger.info(f"{len(res.data)}টি pending task re-armed হয়েছে।")
    except Exception as e:
        logger.error(f"rearm_pending_tasks ব্যর্থ: {e}")


_rearm_pending_tasks()


# ============================================================
# FCM PUSH
# ============================================================

def send_fcm_notification(fcm_token: str, title: str, body: str, data: dict | None = None) -> bool:
    if not firebase_app or not fcm_token:
        return False
    try:
        message = messaging.Message(
            notification=messaging.Notification(title=title, body=body),
            data={str(k): str(v) for k, v in (data or {}).items()},
            token=fcm_token,
            android=messaging.AndroidConfig(priority="high", notification=messaging.AndroidNotification(
                sound="default", channel_id="reminders")),
        )
        messaging.send(message)
        return True
    except Exception as e:
        logger.error(f"FCM send failed: {e}")
        return False


# ============================================================
# ইমেজ কমপ্রেশন + আপলোড
# ============================================================

def compress_image(image_bytes: bytes, quality: int = 20, max_dimension: int = 1280) -> bytes:
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_dimension:
        ratio = max_dimension / max(w, h)
        img = img.resize((max(1, int(w * ratio)), max(1, int(h * ratio))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def upload_image_to_supabase(image_bytes: bytes, user_id: str) -> str | None:
    filename = f"{user_id}/{int(time.time() * 1000)}.jpg"
    try:
        supabase.storage.from_(SUPABASE_BUCKET).upload(filename, image_bytes, {"content-type": "image/jpeg"})
        return supabase.storage.from_(SUPABASE_BUCKET).get_public_url(filename)
    except Exception as e:
        logger.error(f"Supabase image upload failed: {e}")
        return None


# ============================================================
# ওয়েব সার্চ
# ============================================================

_SEARCH_TRIGGER_KEYWORDS = [
    "আজকের", "লেটেস্ট", "latest", "current", "news", "খবর", "আপডেট", "কত টাকা",
    "স্কোর", "score", "today", "এখন", "প্রতিযোগী", "competitor", "market", "মার্কেট",
    "বিশ্লেষণ", "analysis", "trend", "ট্রেন্ড",
]


def should_trigger_search(text: str) -> bool:
    t = text.lower()
    return any(k.lower() in t for k in _SEARCH_TRIGGER_KEYWORDS)


def web_search_tool(query: str, num_results: int = 5) -> list[dict]:
    results: list[dict] = []
    if GOOGLE_LIB_AVAILABLE:
        try:
            for url in google_search_lib(query, num_results=num_results, lang="bn"):
                results.append({"url": url, "title": "", "snippet": "", "source": "google"})
        except Exception as e:
            logger.warning(f"googlesearch ব্যর্থ, DuckDuckGo fallback: {e}")
    if not results and DDGS_AVAILABLE:
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=num_results):
                    results.append({"url": r.get("href", ""), "title": r.get("title", ""),
                                     "snippet": r.get("body", ""), "source": "duckduckgo"})
        except Exception as e:
            logger.error(f"DuckDuckGo search ব্যর্থ: {e}")
    return results


# ============================================================
# AI কল (Groq)
# ============================================================

def _image_to_data_url(image_bytes: bytes) -> str:
    return f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8')}"


def call_ai(user_text: str, image_bytes: bytes | None, chat_history: list, search_context: str = "") -> str:
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    system_prompt = (
        "তুমি একজন সহানুভূতিশীল, ইমোশনালি ইন্টেলিজেন্ট AI অ্যাসিস্ট্যান্ট। "
        "ইউজারের টাস্ক ম্যানেজমেন্ট, রিসার্চ ও মানসিক সাপোর্টে সাহায্য করো। "
        "সবসময় উষ্ণ, বন্ধুত্বপূর্ণ টোনে, ইউজার যে ভাষায় লেখে সেই ভাষায় উত্তর দাও। "
        f"বর্তমান তারিখ/সময় (UTC): {now_str}।"
    )
    if search_context:
        system_prompt += f"\n\nনিচের লাইভ সার্চ রেজাল্ট প্রাসঙ্গিক হলে ব্যবহার করো:\n{search_context}"

    messages = [{"role": "system", "content": system_prompt}]
    for m in chat_history[-10:]:
        c = m["content"]
        if isinstance(c, list):
            c = next((b["text"] for b in c if b.get("type") == "text"), "")
        messages.append({"role": m["role"], "content": c})

    if image_bytes:
        content = [{"type": "text", "text": user_text},
                   {"type": "image_url", "image_url": {"url": _image_to_data_url(image_bytes)}}]
        model_id = GROQ_VISION_MODEL_ID
    else:
        content = user_text
        model_id = GROQ_TEXT_MODEL_ID

    messages.append({"role": "user", "content": content})

    try:
        response = groq_client.chat.completions.create(
            model=model_id, messages=messages, max_tokens=800, temperature=0.7)
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Groq inference ব্যর্থ (model={model_id}): {e}")
        return "দুঃখিত, এই মুহূর্তে AI মডেলের সাথে সংযোগ করা যাচ্ছে না। একটু পরে আবার চেষ্টা করুন।"


# ============================================================
# SUPABASE DATA ACCESS
# ============================================================

def save_chat_message(user_id, role, content, image_url=None):
    supabase.table("chat_history").insert({
        "user_id": user_id, "role": role, "content": content,
        "image_url": image_url, "created_at": datetime.now(timezone.utc).isoformat(),
    }).execute()


def touch_user_profile(user_id: str):
    """ইউজার প্রোফাইল না থাকলে তৈরি করে (ডিফল্ট প্ল্যান 'free'), থাকলে last_active আপডেট করে।
    এতে এডমিন প্যানেল সব ইউজার লিস্ট করতে পারবে।"""
    try:
        now_iso = datetime.now(timezone.utc).isoformat()
        existing = supabase.table("user_profiles").select("user_id").eq("user_id", user_id).execute()
        if existing.data:
            supabase.table("user_profiles").update({"last_active": now_iso}).eq("user_id", user_id).execute()
        else:
            supabase.table("user_profiles").insert({
                "user_id": user_id, "plan": "free",
                "created_at": now_iso, "last_active": now_iso,
            }).execute()
    except Exception as e:
        logger.error(f"touch_user_profile ব্যর্থ: {e}")


def load_chat_history(user_id, limit=20):
    res = (supabase.table("chat_history").select("role, content")
           .eq("user_id", user_id).order("created_at", desc=True).limit(limit).execute())
    return list(reversed(res.data))


# ============================================================
# ROUTES
# ============================================================

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "service": "emotional-ai-agent-api"})


@app.route("/chat", methods=["POST"])
@require_api_key
def chat():
    """
    multipart/form-data দিয়ে কল করতে হবে:
      - user_id (text, required)
      - message (text, required)
      - image (file, optional)
    """
    try:
        user_id = request.form.get("user_id", "").strip()
        user_text = request.form.get("message", "").strip()
        if not user_id or not user_text:
            return jsonify({"error": "user_id এবং message আবশ্যক"}), 400

        image_bytes, image_url = None, None
        if "image" in request.files and request.files["image"].filename:
            try:
                raw = request.files["image"].read()
                image_bytes = compress_image(raw)
                image_url = upload_image_to_supabase(image_bytes, user_id)
            except Exception as e:
                return jsonify({"error": f"[step:image] {type(e).__name__}: {e}"}), 500

        try:
            chat_history = load_chat_history(user_id)
        except Exception as e:
            return jsonify({"error": f"[step:load_history] {type(e).__name__}: {e}"}), 500

        search_context = ""
        if should_trigger_search(user_text):
            try:
                results = web_search_tool(user_text)
                search_context = "\n".join(
                    f"- {r.get('title', '')} — {r.get('url', '')} {r.get('snippet', '')}".strip()
                    for r in results
                )
            except Exception as e:
                return jsonify({"error": f"[step:web_search] {type(e).__name__}: {e}"}), 500

        try:
            reply = call_ai(user_text, image_bytes, chat_history, search_context)
        except Exception as e:
            return jsonify({"error": f"[step:call_ai] {type(e).__name__}: {e}"}), 500

        try:
            save_chat_message(user_id, "user", user_text, image_url)
        except Exception as e:
            return jsonify({"error": f"[step:save_user_msg] {type(e).__name__}: {e}"}), 500

        try:
            save_chat_message(user_id, "assistant", reply)
        except Exception as e:
            return jsonify({"error": f"[step:save_assistant_msg] {type(e).__name__}: {e}"}), 500

        return jsonify({"reply": reply, "image_url": image_url})
    except Exception as e:
        logger.exception("‌/chat route-এ ব্যর্থ")
        return jsonify({
            "error": f"[step:unknown] {type(e).__name__}: {str(e)}",
            "debug_supabase_url": repr(SUPABASE_URL),
        }), 500


@app.route("/chat/history", methods=["GET"])
@require_api_key
def chat_history():
    user_id = request.args.get("user_id", "").strip()
    if not user_id:
        return jsonify({"error": "user_id আবশ্যক"}), 400
    limit = int(request.args.get("limit", 20))
    return jsonify({"history": load_chat_history(user_id, limit)})


@app.route("/tasks", methods=["POST"])
@require_api_key
def create_task_route():
    data = request.get_json(force=True)
    user_id = data.get("user_id")
    title = data.get("title")
    run_at_str = data.get("run_at")  # ISO 8601 format, e.g. "2026-07-20T09:30:00+00:00"
    fcm_token = data.get("fcm_token")
    if not all([user_id, title, run_at_str, fcm_token]):
        return jsonify({"error": "user_id, title, run_at, fcm_token — সবগুলো আবশ্যক"}), 400
    try:
        run_at = datetime.fromisoformat(run_at_str)
    except ValueError:
        return jsonify({"error": "run_at অবশ্যই ISO 8601 ফরম্যাটে হতে হবে"}), 400

    res = supabase.table("user_tasks").insert({
        "user_id": user_id, "title": title, "run_at": run_at.isoformat(),
        "status": "scheduled", "fcm_token": fcm_token,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }).execute()
    task_id = res.data[0]["id"]
    scheduler.schedule(task_id, run_at, {
        "fcm_token": fcm_token, "title": "⏰ রিমাইন্ডার", "body": title,
        "data": {"task_id": task_id, "type": "reminder"},
    })
    return jsonify({"task_id": task_id, "status": "scheduled"})


@app.route("/tasks", methods=["GET"])
@require_api_key
def get_tasks_route():
    user_id = request.args.get("user_id", "").strip()
    if not user_id:
        return jsonify({"error": "user_id আবশ্যক"}), 400
    res = supabase.table("user_tasks").select("*").eq("user_id", user_id).order("run_at").execute()
    return jsonify({"tasks": res.data})


@app.route("/tasks/<task_id>", methods=["DELETE"])
@require_api_key
def cancel_task_route(task_id):
    scheduler.cancel(task_id)
    supabase.table("user_tasks").update({"status": "cancelled"}).eq("id", task_id).execute()
    return jsonify({"status": "cancelled"})


@app.route("/mood", methods=["POST"])
@require_api_key
def save_mood_route():
    data = request.get_json(force=True)
    user_id = data.get("user_id")
    mood = data.get("mood")
    note = data.get("note", "")
    if not user_id or not mood:
        return jsonify({"error": "user_id ও mood আবশ্যক"}), 400
    supabase.table("user_mood").insert({
        "user_id": user_id, "mood": mood, "note": note,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }).execute()
    return jsonify({"status": "saved"})


@app.route("/mood/latest", methods=["GET"])
@require_api_key
def latest_mood_route():
    user_id = request.args.get("user_id", "").strip()
    if not user_id:
        return jsonify({"error": "user_id আবশ্যক"}), 400
    res = (supabase.table("user_mood").select("*").eq("user_id", user_id)
           .order("created_at", desc=True).limit(1).execute())
    return jsonify({"mood": res.data[0] if res.data else None})


@app.route("/version", methods=["GET"])
def version_route():
    res = supabase.table("app_version").select("*").order("id", desc=True).limit(1).execute()
    return jsonify(res.data[0] if res.data else {})


if __name__ == "__main__":
    # লোকাল টেস্টের জন্য (Render-এ gunicorn দিয়ে চলবে, এটা চলবে না)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
