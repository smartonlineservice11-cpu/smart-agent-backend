"""
app.py — Emotional AI Task & Research Agent (Backend)
Render.com / Hugging Face Spaces (Streamlit) হোস্টিং-এর জন্য ডিজাইন করা।

ফিচারসমূহ:
 1. Groq API (Llama/Qwen vision-capable মডেল) — টেক্সট + ইমেজ প্রসেসিং, সম্পূর্ণ ফ্রি টায়ার
 2. ফ্রি ওয়েব সার্চ টুল (googlesearch-python + duckduckgo fallback)
 3. Supabase (Postgres + Storage) ইন্টিগ্রেশন
 4. আপলোডের আগে ব্যাকএন্ডে ইমেজ কমপ্রেশন
 5. app_version টেবিল থেকে ভার্সন চেক
 6. os.environ থেকে সব সিক্রেট রিড (কোনো hard-coded key নেই)
 7. In-memory threading.Timer ভিত্তিক রিমাইন্ডার + Firebase FCM push
"""

import os
import io
import json
import time
import base64
import logging
import threading
from datetime import datetime, timezone

import streamlit as st
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
logger = logging.getLogger("emotional-ai-agent")

# ============================================================
# ENVIRONMENT VARIABLES (Hugging Face Space → Settings → Variables and secrets)
# ============================================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
SUPABASE_BUCKET = os.environ.get("SUPABASE_STORAGE_BUCKET", "user-images")
FIREBASE_CREDENTIALS_JSON = os.environ.get("FIREBASE_CREDENTIALS_JSON")
# ছবি + টেক্সট দুটোই প্রসেস করতে পারে এমন Groq vision মডেল (প্রয়োজনে env var দিয়ে বদলানো যাবে)
GROQ_VISION_MODEL_ID = os.environ.get("GROQ_VISION_MODEL_ID", "qwen/qwen3.6-27b")
# শুধু টেক্সট চ্যাটের জন্য দ্রুততর/সস্তা মডেল (ছবি ছাড়া মেসেজে ব্যবহৃত হবে)
GROQ_TEXT_MODEL_ID = os.environ.get("GROQ_TEXT_MODEL_ID", "openai/gpt-oss-120b")

_REQUIRED_ENV = {"GROQ_API_KEY": GROQ_API_KEY, "SUPABASE_URL": SUPABASE_URL, "SUPABASE_KEY": SUPABASE_KEY}
_missing = [k for k, v in _REQUIRED_ENV.items() if not v]
if _missing:
    st.error(
        f"⚠️ প্রয়োজনীয় Environment Variable পাওয়া যায়নি: **{', '.join(_missing)}**\n\n"
        "Render → Environment থেকে যোগ করুন।"
    )
    st.stop()

if not FIREBASE_CREDENTIALS_JSON:
    logger.warning("FIREBASE_CREDENTIALS_JSON সেট নেই — পুশ নোটিফিকেশন কাজ করবে না (বাকি সব ফিচার চলবে)।")

# ============================================================
# SINGLETON CLIENTS (st.cache_resource → প্রতি rerun-এ নতুন করে তৈরি হয় না)
# ============================================================

@st.cache_resource
def get_supabase_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)


@st.cache_resource
def get_groq_client() -> Groq:
    return Groq(api_key=GROQ_API_KEY)


@st.cache_resource
def get_firebase_app():
    if not FIREBASE_CREDENTIALS_JSON:
        return None
    try:
        cred_dict = json.loads(FIREBASE_CREDENTIALS_JSON)
        cred = credentials.Certificate(cred_dict)
        if not firebase_admin._apps:
            return firebase_admin.initialize_app(cred)
        return firebase_admin.get_app()
    except Exception as e:
        logger.error(f"Firebase init failed: {e}")
        return None


supabase: Client = get_supabase_client()
groq_client: Groq = get_groq_client()
firebase_app = get_firebase_app()


# ============================================================
# TASK SCHEDULER — ইন-মেমোরি threading.Timer ভিত্তিক (NO DB POLLING LOOP)
# ============================================================

class TaskScheduler:
    """
    ইউজারের রিমাইন্ডার/টাস্কের জন্য ইন-মেমোরি countdown scheduler।
    প্রতি মিনিট ডেটাবেজ পোল না করে, ঠিক নির্দিষ্ট সময়ে single-shot
    threading.Timer দিয়ে ট্রিগার হয় — CPU/ব্যান্ডউইথ খরচ প্রায় শূন্য।

    সীমাবদ্ধতা: টাইমারগুলো process memory-তে থাকে বলে HF Spaces ফ্রি
    টায়ারে Space sleep/restart হলে হারিয়ে যায়। এটা সামলাতে
    `init_pending_tasks_on_startup()` চালু হওয়ার সময় Supabase থেকে
    ভবিষ্যতের pending task একবার লোড করে re-arm করে — এটা recurring
    polling নয়, শুধু startup-এ একবার চলা রিকভারি স্টেপ।
    """

    def __init__(self):
        self._timers: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

    def schedule(self, task_id: str, run_at: datetime, payload: dict):
        delay = (run_at - datetime.now(timezone.utc)).total_seconds()
        delay = max(delay, 0)
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
                fcm_token=payload["fcm_token"],
                title=payload.get("title", "⏰ রিমাইন্ডার"),
                body=payload.get("body", ""),
                data=payload.get("data", {}),
            )
            supabase.table("user_tasks").update({"status": "sent"}).eq("id", task_id).execute()
        except Exception as e:
            logger.error(f"Task {task_id} firing failed: {e}")
        finally:
            with self._lock:
                self._timers.pop(task_id, None)


@st.cache_resource
def get_scheduler() -> TaskScheduler:
    return TaskScheduler()


scheduler = get_scheduler()


# ============================================================
# FIREBASE CLOUD MESSAGING (FCM) — ইনস্ট্যান্ট পুশ (০$ বাজেট, Spark plan-এও ফ্রি)
# ============================================================

def send_fcm_notification(fcm_token: str, title: str, body: str, data: dict | None = None) -> bool:
    if not firebase_app:
        logger.warning("Firebase app initialized নয় — নোটিফিকেশন স্কিপ করা হলো।")
        return False
    if not fcm_token:
        logger.warning("FCM token খালি — নোটিফিকেশন পাঠানো যায়নি।")
        return False
    try:
        message = messaging.Message(
            notification=messaging.Notification(title=title, body=body),
            data={str(k): str(v) for k, v in (data or {}).items()},
            token=fcm_token,
            android=messaging.AndroidConfig(
                priority="high",
                notification=messaging.AndroidNotification(sound="default", channel_id="reminders"),
            ),
            apns=messaging.APNSConfig(
                payload=messaging.APNSPayload(aps=messaging.Aps(sound="default"))
            ),
        )
        response = messaging.send(message)
        logger.info(f"FCM sent successfully: {response}")
        return True
    except Exception as e:
        logger.error(f"FCM send failed: {e}")
        return False


# ============================================================
# ইমেজ কমপ্রেশন (আপলোডের আগে)
# ============================================================

def compress_image(image_bytes: bytes, quality: int = 20, max_dimension: int = 1280) -> bytes:
    """
    JPEG quality≈20 (অর্থাৎ ~৮০% quality reduction) + বড় হলে resize —
    ফাইল সাইজ উল্লেখযোগ্যভাবে কমিয়ে দেয়, দৃশ্যমান গুণমান মোটামুটি বজায় রেখে।
    """
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
        supabase.storage.from_(SUPABASE_BUCKET).upload(
            filename, image_bytes, {"content-type": "image/jpeg"}
        )
        return supabase.storage.from_(SUPABASE_BUCKET).get_public_url(filename)
    except Exception as e:
        logger.error(f"Supabase image upload failed: {e}")
        return None


# ============================================================
# ফ্রি ওয়েব সার্চ টুল (লাইভ রিসার্চ)
# ============================================================

_SEARCH_TRIGGER_KEYWORDS = [
    "আজকের", "লেটেস্ট", "latest", "current", "news", "খবর", "আপডেট",
    "কত টাকা", "স্কোর", "score", "today", "এখন", "২০২৬", "2026",
]


def should_trigger_search(text: str) -> bool:
    t = text.lower()
    return any(k.lower() in t for k in _SEARCH_TRIGGER_KEYWORDS)


def web_search_tool(query: str, num_results: int = 5) -> list[dict]:
    """
    কোনো API key ছাড়া ফ্রি লাইভ ওয়েব সার্চ।
    প্রথমে googlesearch-python, ব্যর্থ হলে duckduckgo (ddgs) fallback।
    """
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
                    results.append({
                        "url": r.get("href", ""),
                        "title": r.get("title", ""),
                        "snippet": r.get("body", ""),
                        "source": "duckduckgo",
                    })
        except Exception as e:
            logger.error(f"DuckDuckGo search ব্যর্থ: {e}")

    return results


# ============================================================
# Groq দিয়ে AI কল (টেক্সট + ভিশন)
# ============================================================

def _image_to_data_url(image_bytes: bytes) -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def call_ai(
    user_text: str,
    image_bytes: bytes | None = None,
    chat_history: list | None = None,
    search_context: str = "",
) -> str:
    """
    ছবি থাকলে GROQ_VISION_MODEL_ID (multimodal) মডেল ব্যবহার করে,
    নাহলে দ্রুততর GROQ_TEXT_MODEL_ID মডেল ব্যবহার করে।
    """
    system_prompt = (
        "তুমি একজন সহানুভূতিশীল, ইমোশনালি ইন্টেলিজেন্ট AI অ্যাসিস্ট্যান্ট। "
        "ইউজারের টাস্ক ম্যানেজমেন্ট, রিসার্চ ও মানসিক সাপোর্টে সাহায্য করো। "
        "সবসময় উষ্ণ, বন্ধুত্বপূর্ণ টোনে, ইউজার যে ভাষায় লেখে সেই ভাষায় উত্তর দাও।"
    )
    if search_context:
        system_prompt += f"\n\nনিচের লাইভ সার্চ রেজাল্ট প্রাসঙ্গিক হলে ব্যবহার করো:\n{search_context}"

    messages = [{"role": "system", "content": system_prompt}]
    if chat_history:
        # ছবিসহ পুরনো মেসেজ (list-type content) পাঠালে টেক্সট-অনলি মডেল এরর দিতে পারে,
        # তাই হিস্ট্রিতে থাকা কনটেন্ট সবসময় প্লেইন টেক্সট হিসেবে রাখা হচ্ছে।
        for m in chat_history[-10:]:
            c = m["content"]
            if isinstance(c, list):
                c = next((b["text"] for b in c if b.get("type") == "text"), "")
            messages.append({"role": m["role"], "content": c})

    if image_bytes:
        content = [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": _image_to_data_url(image_bytes)}},
        ]
        model_id = GROQ_VISION_MODEL_ID
    else:
        content = user_text
        model_id = GROQ_TEXT_MODEL_ID

    messages.append({"role": "user", "content": content})

    try:
        response = groq_client.chat.completions.create(
            model=model_id, messages=messages, max_tokens=800, temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Groq inference ব্যর্থ (model={model_id}): {e}")
        return "দুঃখিত, এই মুহূর্তে AI মডেলের সাথে সংযোগ করা যাচ্ছে না। একটু পরে আবার চেষ্টা করুন।"


# ============================================================
# SUPABASE DATA ACCESS
# ============================================================

def save_chat_message(user_id: str, role: str, content: str, image_url: str | None = None):
    try:
        supabase.table("chat_history").insert({
            "user_id": user_id, "role": role, "content": content,
            "image_url": image_url, "created_at": datetime.now(timezone.utc).isoformat(),
        }).execute()
    except Exception as e:
        logger.error(f"chat_history insert ব্যর্থ: {e}")


def load_chat_history(user_id: str, limit: int = 20) -> list:
    try:
        res = (
            supabase.table("chat_history").select("role, content")
            .eq("user_id", user_id).order("created_at", desc=True).limit(limit).execute()
        )
        return list(reversed(res.data))
    except Exception as e:
        logger.error(f"chat_history লোড ব্যর্থ: {e}")
        return []


def update_user_mood(user_id: str, mood: str, note: str = ""):
    try:
        supabase.table("user_mood").insert({
            "user_id": user_id, "mood": mood, "note": note,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }).execute()
    except Exception as e:
        logger.error(f"user_mood insert ব্যর্থ: {e}")


def get_latest_mood(user_id: str) -> dict | None:
    try:
        res = (
            supabase.table("user_mood").select("*").eq("user_id", user_id)
            .order("created_at", desc=True).limit(1).execute()
        )
        return res.data[0] if res.data else None
    except Exception as e:
        logger.error(f"get_latest_mood ব্যর্থ: {e}")
        return None


def create_task(user_id: str, title: str, run_at: datetime, fcm_token: str) -> str | None:
    try:
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
        return task_id
    except Exception as e:
        logger.error(f"create_task ব্যর্থ: {e}")
        return None


def get_user_tasks(user_id: str) -> list:
    try:
        res = (
            supabase.table("user_tasks").select("*").eq("user_id", user_id)
            .order("run_at", desc=False).execute()
        )
        return res.data
    except Exception as e:
        logger.error(f"get_user_tasks ব্যর্থ: {e}")
        return []


def cancel_task(task_id: str):
    scheduler.cancel(task_id)
    try:
        supabase.table("user_tasks").update({"status": "cancelled"}).eq("id", task_id).execute()
    except Exception as e:
        logger.error(f"cancel_task DB আপডেট ব্যর্থ: {e}")


def check_app_version() -> dict:
    try:
        res = supabase.table("app_version").select("*").order("id", desc=True).limit(1).execute()
        return res.data[0] if res.data else {}
    except Exception as e:
        logger.error(f"check_app_version ব্যর্থ: {e}")
        return {}


@st.cache_resource
def init_pending_tasks_on_startup():
    """
    Space চালু/রিস্টার্ট হওয়ার সময় একবার চলে: Supabase-এ থাকা
    ভবিষ্যতের 'scheduled' টাস্কগুলো আবার in-memory scheduler-এ re-arm করে।
    (এটা recurring polling নয় — শুধু cold-start রিকভারি।)
    """
    try:
        now_iso = datetime.now(timezone.utc).isoformat()
        res = (
            supabase.table("user_tasks").select("*")
            .eq("status", "scheduled").gt("run_at", now_iso).execute()
        )
        for row in res.data:
            run_at = datetime.fromisoformat(row["run_at"])
            scheduler.schedule(row["id"], run_at, {
                "fcm_token": row["fcm_token"], "title": "⏰ রিমাইন্ডার",
                "body": row["title"], "data": {"task_id": row["id"], "type": "reminder"},
            })
        logger.info(f"{len(res.data)}টি pending task re-armed হয়েছে।")
    except Exception as e:
        logger.error(f"init_pending_tasks_on_startup ব্যর্থ: {e}")
    return True


init_pending_tasks_on_startup()


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(page_title="Emotional AI Agent", page_icon="🧠", layout="wide")

if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "fcm_token" not in st.session_state:
    st.session_state.fcm_token = None
if "chat_display" not in st.session_state:
    st.session_state.chat_display = None

with st.sidebar:
    st.header("👤 ইউজার সেটিংস")
    user_id_input = st.text_input("User ID", value=st.session_state.user_id or "")
    fcm_token_input = st.text_input(
        "FCM Device Token", value=st.session_state.fcm_token or "",
        help="মোবাইল অ্যাপ থেকে পাওয়া Firebase device token — এটা ছাড়া পুশ নোটিফিকেশন যাবে না।",
    )
    if st.button("সেভ করুন", use_container_width=True):
        st.session_state.user_id = user_id_input.strip() or None
        st.session_state.fcm_token = fcm_token_input.strip() or None
        st.session_state.chat_display = None
        st.success("সেভ হয়েছে।")

    st.divider()
    version_info = check_app_version()
    if version_info:
        st.caption(f"বর্তমান ভার্সন: v{version_info.get('version', '?')}")
        if version_info.get("force_update"):
            st.warning(f"🔔 নতুন ভার্সন উপলব্ধ: v{version_info.get('version')} — অ্যাপ আপডেট করুন।")

if not st.session_state.user_id:
    st.info("শুরু করতে বাম পাশের সাইডবার থেকে User ID সেট করুন।")
    st.stop()

user_id = st.session_state.user_id

tab_chat, tab_tasks, tab_mood = st.tabs(["💬 চ্যাট ও রিসার্চ", "⏰ টাস্ক/রিমাইন্ডার", "😊 মুড ট্র্যাকিং"])

# ---------------- চ্যাট ----------------
with tab_chat:
    if st.session_state.chat_display is None:
        st.session_state.chat_display = load_chat_history(user_id)

    for msg in st.session_state.chat_display:
        with st.chat_message(msg["role"]):
            content = msg["content"]
            st.markdown(content if isinstance(content, str) else content[0]["text"])

    uploaded_image = st.file_uploader("ছবি যুক্ত করুন (ঐচ্ছিক)", type=["png", "jpg", "jpeg"])
    user_text = st.chat_input("আপনার মনের কথা বলুন...")

    if user_text:
        image_bytes, image_url = None, None
        if uploaded_image is not None:
            raw = uploaded_image.getvalue()
            image_bytes = compress_image(raw)
            image_url = upload_image_to_supabase(image_bytes, user_id)

        with st.chat_message("user"):
            st.markdown(user_text)
            if image_url:
                st.image(image_url, width=200)

        search_context = ""
        if should_trigger_search(user_text):
            with st.spinner("লাইভ তথ্য খোঁজা হচ্ছে..."):
                results = web_search_tool(user_text)
                search_context = "\n".join(
                    f"- {r.get('title', '')} — {r.get('url', '')} {r.get('snippet', '')}".strip()
                    for r in results
                )

        with st.chat_message("assistant"):
            with st.spinner("ভাবছি..."):
                reply = call_ai(
                    user_text, image_bytes,
                    chat_history=st.session_state.chat_display,
                    search_context=search_context,
                )
            st.markdown(reply)

        save_chat_message(user_id, "user", user_text, image_url)
        save_chat_message(user_id, "assistant", reply)
        st.session_state.chat_display.append({"role": "user", "content": user_text})
        st.session_state.chat_display.append({"role": "assistant", "content": reply})

# ---------------- টাস্ক/রিমাইন্ডার ----------------
with tab_tasks:
    st.subheader("নতুন রিমাইন্ডার তৈরি করুন")
    col1, col2 = st.columns(2)
    with col1:
        task_title = st.text_input("টাস্ক/রিমাইন্ডার টাইটেল", key="task_title")
        task_date = st.date_input("তারিখ", key="task_date")
    with col2:
        task_time = st.time_input("সময়", key="task_time")

    if st.button("রিমাইন্ডার সেট করুন", type="primary"):
        if not st.session_state.fcm_token:
            st.error("প্রথমে সাইডবারে FCM Device Token সেট করুন।")
        elif not task_title:
            st.error("টাইটেল দিন।")
        else:
            run_at = datetime.combine(task_date, task_time).replace(tzinfo=timezone.utc)
            task_id = create_task(user_id, task_title, run_at, st.session_state.fcm_token)
            if task_id:
                st.success(f"রিমাইন্ডার সেট হয়েছে — {run_at.strftime('%Y-%m-%d %H:%M UTC')}")
            else:
                st.error("রিমাইন্ডার তৈরি ব্যর্থ হয়েছে।")

    st.divider()
    st.subheader("আপনার টাস্কসমূহ")
    for t in get_user_tasks(user_id):
        c1, c2, c3 = st.columns([3, 2, 1])
        c1.write(t["title"])
        c2.write(f"{t['run_at']} — **{t['status']}**")
        if t["status"] == "scheduled" and c3.button("বাতিল", key=f"cancel_{t['id']}"):
            cancel_task(t["id"])
            st.rerun()

# ---------------- মুড ট্র্যাকিং ----------------
with tab_mood:
    st.subheader("আজ আপনার মুড কেমন?")
    mood_options = ["😄 খুশি", "😐 স্বাভাবিক", "😢 মন খারাপ", "😡 রাগান্বিত", "😰 উদ্বিগ্ন"]
    selected_mood = st.radio("মুড নির্বাচন করুন", mood_options, horizontal=True)
    mood_note = st.text_area("অতিরিক্ত নোট (ঐচ্ছিক)")
    if st.button("মুড সেভ করুন"):
        update_user_mood(user_id, selected_mood, mood_note)
        st.success("মুড সেভ হয়েছে!")

    latest = get_latest_mood(user_id)
    if latest:
        st.caption(f"সর্বশেষ মুড: {latest['mood']} ({latest['created_at'][:16]})")
