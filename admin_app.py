"""
admin_app.py — Smart Agent এডমিন প্যানেল
এটা মূল API (app.py) থেকে সম্পূর্ণ আলাদা, দ্বিতীয় একটা Render service হিসেবে
ডিপ্লয় হবে। শুধু আপনি (এডমিন) পাসওয়ার্ড দিয়ে ঢুকবেন।

Start Command (Render): streamlit run admin_app.py --server.port $PORT --server.address 0.0.0.0
"""

import os
from datetime import datetime, timezone

import streamlit as st
from supabase import create_client, Client

# ============================================================
# ENV VARS
# ============================================================
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD")

if not all([SUPABASE_URL, SUPABASE_KEY, ADMIN_PASSWORD]):
    st.error(
        "⚠️ প্রয়োজনীয় Environment Variable পাওয়া যায়নি "
        "(SUPABASE_URL, SUPABASE_KEY, ADMIN_PASSWORD) — Render → Environment থেকে যোগ করুন।"
    )
    st.stop()


@st.cache_resource
def get_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)


supabase = get_supabase()

st.set_page_config(page_title="Smart Agent — Admin Panel", page_icon="🛡️", layout="wide")

# ============================================================
# পাসওয়ার্ড লগইন
# ============================================================
if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False

if not st.session_state.admin_logged_in:
    st.title("🛡️ Admin Panel — লগইন")
    pw = st.text_input("পাসওয়ার্ড দিন", type="password")
    if st.button("লগইন করুন"):
        if pw == ADMIN_PASSWORD:
            st.session_state.admin_logged_in = True
            st.rerun()
        else:
            st.error("ভুল পাসওয়ার্ড।")
    st.stop()

# ============================================================
# ডেটা ফাংশন
# ============================================================

def get_all_users() -> list:
    """user_profiles টেবিল থেকে সব ইউজার আনে, না থাকলে chat_history থেকে ইউনিক user_id বের করে।"""
    try:
        res = supabase.table("user_profiles").select("*").order("created_at", desc=True).execute()
        if res.data:
            return res.data
    except Exception:
        pass
    # fallback: chat_history থেকে distinct user_id বের করা
    try:
        res = supabase.table("chat_history").select("user_id").execute()
        seen = {}
        for row in res.data:
            uid = row["user_id"]
            if uid not in seen:
                seen[uid] = {"user_id": uid, "plan": "free", "created_at": None}
        return list(seen.values())
    except Exception as e:
        st.error(f"ইউজার লোড করতে সমস্যা: {e}")
        return []


def ensure_user_profile(user_id: str):
    """user_profiles-এ রেকর্ড না থাকলে ডিফল্ট (free) দিয়ে তৈরি করে।"""
    try:
        existing = supabase.table("user_profiles").select("user_id").eq("user_id", user_id).execute()
        if not existing.data:
            supabase.table("user_profiles").insert({
                "user_id": user_id, "plan": "free",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }).execute()
    except Exception as e:
        st.warning(f"user_profiles টেবিল সম্ভবত এখনো তৈরি হয়নি: {e}")


def update_user_plan(user_id: str, new_plan: str):
    ensure_user_profile(user_id)
    supabase.table("user_profiles").update({
        "plan": new_plan, "updated_at": datetime.now(timezone.utc).isoformat(),
    }).eq("user_id", user_id).execute()


def get_user_chat_history(user_id: str, limit: int = 50) -> list:
    res = (supabase.table("chat_history").select("*").eq("user_id", user_id)
           .order("created_at", desc=True).limit(limit).execute())
    return list(reversed(res.data))


def get_user_tasks(user_id: str) -> list:
    res = supabase.table("user_tasks").select("*").eq("user_id", user_id).order("run_at").execute()
    return res.data


def get_stats() -> dict:
    users = get_all_users()
    try:
        chat_count = len(supabase.table("chat_history").select("id").execute().data)
    except Exception:
        chat_count = 0
    plan_counts = {"free": 0, "paid": 0, "unlimited": 0}
    for u in users:
        p = u.get("plan", "free")
        plan_counts[p] = plan_counts.get(p, 0) + 1
    return {"total_users": len(users), "total_messages": chat_count, "plan_counts": plan_counts}


# ============================================================
# UI
# ============================================================

with st.sidebar:
    st.title("🛡️ Admin Panel")
    if st.button("🚪 লগ-আউট"):
        st.session_state.admin_logged_in = False
        st.rerun()

st.title("📊 Smart Agent — ড্যাশবোর্ড")

# ---------- সারসংক্ষেপ কার্ড ----------
stats = get_stats()
c1, c2, c3, c4 = st.columns(4)
c1.metric("মোট ইউজার", stats["total_users"])
c2.metric("মোট মেসেজ", stats["total_messages"])
c3.metric("পেইড ইউজার", stats["plan_counts"].get("paid", 0))
c4.metric("ফ্রি ইউজার", stats["plan_counts"].get("free", 0))

st.divider()

tab_users, tab_monitor = st.tabs(["👥 ইউজার ও প্ল্যান ম্যানেজমেন্ট", "💬 চ্যাট মনিটরিং"])

# ---------------- ইউজার ম্যানেজমেন্ট ----------------
with tab_users:
    st.subheader("সব ইউজার")
    users = get_all_users()

    if not users:
        st.info("এখনো কোনো ইউজার নেই।")
    else:
        search = st.text_input("🔍 User ID দিয়ে খুঁজুন")
        filtered = [u for u in users if search.lower() in u["user_id"].lower()] if search else users

        for u in filtered:
            with st.container(border=True):
                col1, col2, col3 = st.columns([2, 2, 1])
                col1.markdown(f"**{u['user_id']}**")
                col1.caption(f"যোগ দিয়েছে: {u.get('created_at', 'অজানা')}")

                current_plan = u.get("plan", "free")
                new_plan = col2.selectbox(
                    "প্ল্যান", ["free", "paid", "unlimited"],
                    index=["free", "paid", "unlimited"].index(current_plan) if current_plan in
                    ["free", "paid", "unlimited"] else 0,
                    key=f"plan_{u['user_id']}", label_visibility="collapsed",
                )
                if col3.button("আপডেট", key=f"update_{u['user_id']}"):
                    update_user_plan(u["user_id"], new_plan)
                    st.success(f"{u['user_id']}-এর প্ল্যান '{new_plan}' করা হয়েছে।")
                    st.rerun()

# ---------------- চ্যাট মনিটরিং ----------------
with tab_monitor:
    st.subheader("ইউজারের চ্যাট হিস্ট্রি দেখুন")
    users = get_all_users()
    user_ids = [u["user_id"] for u in users]

    if not user_ids:
        st.info("এখনো কোনো চ্যাট ডেটা নেই।")
    else:
        selected_user = st.selectbox("ইউজার বেছে নিন", user_ids)

        if selected_user:
            st.markdown(f"### 💬 {selected_user}-এর চ্যাট")
            history = get_user_chat_history(selected_user)
            if not history:
                st.info("এই ইউজারের কোনো চ্যাট হিস্ট্রি নেই।")
            for msg in history:
                role_label = "🧑 ইউজার" if msg["role"] == "user" else "🤖 AI"
                with st.chat_message(msg["role"]):
                    st.caption(f"{role_label} — {msg.get('created_at', '')}")
                    st.write(msg["content"])
                    if msg.get("image_url"):
                        st.image(msg["image_url"], width=200)

            st.divider()
            st.markdown(f"### ⏰ {selected_user}-এর রিমাইন্ডার/টাস্ক")
            tasks = get_user_tasks(selected_user)
            if not tasks:
                st.info("কোনো টাস্ক নেই।")
            for t in tasks:
                st.write(f"**{t['title']}** — {t['run_at']} — স্ট্যাটাস: `{t['status']}`")
