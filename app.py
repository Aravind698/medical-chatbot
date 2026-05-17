
import os
import streamlit as st
import requests
import base64
import sqlite3
import hashlib
from pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
import PyPDF2
import docx
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="Medical Chatbot", page_icon="🩺", layout="wide")
with open("style.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------
# DATABASE SETUP
# ---------------------------------------------------------
conn = sqlite3.connect("users.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE,
    password TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS chats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_email TEXT,
    chat_title TEXT,
    role TEXT,
    message TEXT
)
""")

conn.commit()

# ---------------------------------------------------------
# AUTH FUNCTIONS
# ---------------------------------------------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(email, password):
    try:
        cursor.execute(
            "INSERT INTO users (email, password) VALUES (?, ?)",
            
(email, hash_password(password))
        )
        conn.commit()
        return True
    except:
        return False

def login_user(email, password):
    cursor.execute(
        "SELECT * FROM users WHERE email=? AND password=?",
        (email, hash_password(password))
    )
    return cursor.fetchone()

# ---------------------------------------------------------
# CHAT STORAGE
# ---------------------------------------------------------
def save_message(user, chat_title, role, message):
    cursor.execute(
        "INSERT INTO chats (user_email, chat_title, role, message) VALUES (?, ?, ?, ?)",
        (user, chat_title, role, message)
    )
    conn.commit()

def load_user_chats(user):
    cursor.execute(
        "SELECT chat_title, role, message FROM chats WHERE user_email=?",
       
 (user,)
    )
    rows = cursor.fetchall()

    chats = {}
    for title, role, message in rows:
        if title not in chats:
            chats[title] = []
        chats[title].append({"role": role, "content": message})
    return chats

def delete_chat(user, chat_title):
    cursor.execute(
        "DELETE FROM chats WHERE user_email=? AND chat_title=?",
        (user, chat_title)
    )
    conn.commit()

# ---------------------------------------------------------
# TF-IDF TITLE GENERATOR
# ---------------------------------------------------------
def generate_chat_title_tfidf(query):
    stop_words = [
        "what", "is", "are", "the", "and", "how", "to",
        "i", "have", "a", "an", "of", "for", "in", "on",
        "with", "my", "it", "this", "that"
    ]

    vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = vectorizer.fit_transform([query])

    feature_array = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]

    sorted_indices = tfidf_scores.argsort()[::-1]

    top_words = [feature_array[i] for i in sorted_indices[:3]]

    return " ".join(top_words).capitalize()

# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user" not in st.session_state:
    st.session_state.user = None

if "conversations" not in st.session_state:
    st.session_state.conversations = {}

if "active_chat" not in st.session_state:
    st.session_state.active_chat = None

# ---------------------------------------------------------

# LOGIN PAGE
def login_page():
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown("<h1 style='text-align: center; background: linear-gradient(135deg, #3b82f6 0%, #a855f7 50%, #ec4899 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; padding-bottom: 10px;'>🔐 Welcome Back</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #94a3b8; margin-bottom: 2rem;'>Sign in to access your medical chatbot</p>", unsafe_allow_html=True)
        
        choice = st.radio("Option", ["Login", "Create Account"], horizontal=True, label_visibility="collapsed")
        st.markdown("<br>", unsafe_allow_html=True)

        if choice == "Login":
            email = st.text_input("Email", placeholder="Enter your email")
            password = st.text_input("Password", type="password", placeholder="Enter your password")

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Login", type="primary", use_container_width=True):
                user = login_user(email, password)
                if user:
                    st.session_state.logged_in = True
                    st.session_state.user = email
                    st.session_state.conversations = load_user_chats(email)
                    st.rerun()
                else:
                    st.error("Invalid credentials")

        else:


            email = st.text_input("New Email", placeholder="Choose an email")
            password = st.text_input("New Password", type="password", placeholder="Choose a password")

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Create Account", type="primary", use_container_width=True):
                if create_user(email, password):
                    st.success("Account created! You can now log in.")
                else:
                    st.error("Email already exists")

# ---------------------------------------------------------
# STOP IF NOT LOGGED IN
# ---------------------------------------------------------
if not st.session_state.logged_in:
    login_page()
    st.stop()

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
GEMINI_API_KEY = st.secrets["gemini_api_key"]
MODEL_NAME = "models/gemini-2.5-flash"
PINECONE_API_KEY = st.secrets["pinecone_api_key"]

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("medical-chatbot")

@st.cache_resource

def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = load_embeddings()

# ---------------------------------------------------------
# GEMINI
# ---------------------------------------------------------
def call_gemini(prompt):
    url = f"https://generativelanguage.googleapis.com/v1/{MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"
    res = requests.post(url, json={"contents":[{"parts":[{"text":prompt}]}]})
    if res.status_code == 200:
        return res.json()["candidates"][0]["content"]["parts"][0]["text"]
    return "Error"

def call_gemini_image(prompt, image_file):
    image_bytes = image_file.getvalue()
    image_base64 = base64.b64encode(image_bytes).decode()

    payload = {
        "contents":[{
            "parts":[
                {"text":prompt},
                {"inline_data":{"mime_type":image_file.type,"data":image_base64}}
            ]
        }]
    }

    url = f"https://generativelanguage.googleapis.com/v1/{MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"
    res = requests.post(url, json=payload)

    if res.status_code == 200:
        return res.json()["candidates"][0]["content"]["parts"][0]["text"]
    return "Error"

def is_image(file):
    return file.type in ["image/png","image/jpeg","image/jpg"]

# ---------------------------------------------------------
# RAG
# ---------------------------------------------------------
def get_medical_answer(query):
    vec = embeddings.embed_query(query)
    res = index.query(vector=vec, top_k=5, include_metadata=True)
    context = "\n\n".join([m["metadata"].get("text","") for m in res.get("matches",[])])
    
    prompt = f"""
You are a highly intelligent and empathetic Medical AI Assistant.
User Query: "{query}"

If the user's query is a simple greeting or non-medical conversation (like 'hi', 'hello', 'thanks'), respond naturally, warmly, and concisely.

For medical questions, use the provided context to give an accurate, easy-to-understand answer.
Context:


{context}

Instructions for your response style:
- Be dynamic and natural! Do NOT force a rigid format like "Overview, Causes, Symptoms, Treatment" unless it perfectly fits the user's question.
- Tailor your structure to the specific query. Answer direct questions directly. Use numbered lists for steps, or bullet points for key facts.
- Use beautiful markdown formatting (**bold text** for emphasis, clear subheadings, etc.) to make your answer highly readable.
- Maintain a professional, comforting, and informative tone.
- Always include a brief, gentle disclaimer at the very end reminding the user to consult a doctor for official medical advice.
"""
    return call_gemini(prompt)

# ---------------------------------------------------------
# FILE HANDLING
# ---------------------------------------------------------
def extract_file_text(file):
    if file.type == "text/plain":
        return file.read().decode()
    elif file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        return "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
    elif file.type.endswith("document"):
        doc = docx.Document(file)
        return "\n".join([p.text for p in doc.paragraphs])
    return None

def explain_medical_file(text, filename, query):

    return call_gemini(f"{text}\n\n{query}")

# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.title("🩺 Medical Chatbot")
st.markdown("---")

# Sidebar
st.sidebar.title("💬 Chats")

if st.sidebar.button("➕ New Chat"):
    st.session_state.active_chat = None

# Chat list with delete
for title in list(st.session_state.conversations.keys()):
    c1, c2 = st.sidebar.columns([4,1])

    if c1.button(title, key=f"open_{title}"):
        st.session_state.active_chat = title

    if c2.button("🗑️", key=f"del_{title}"):
        delete_chat(st.session_state.user, title)
        del st.session_state.conversations[title]
        if st.session_state.active_chat == title:
            st.session_state.active_chat = None
        st.rerun()


# Account bottom
st.sidebar.markdown("---")
st.sidebar.write(st.session_state.user)

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.user = None
    st.session_state.conversations = {}
    st.rerun()

# Chat display
if st.session_state.active_chat:
    for msg in st.session_state.conversations[st.session_state.active_chat]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# Input
cols = st.columns([1,6,3])
with cols[0]:
    uploaded_file = st.file_uploader("", type=["pdf","txt","docx","png","jpg","jpeg"])
with cols[1]:
    query = st.chat_input("Ask a medical question...")
with cols[2]:
    if uploaded_file:
        st.markdown(f"📎 {uploaded_file.name}")

# ---------------------------------------------------------
# MAIN LOGIC

# ---------------------------------------------------------
if query:
    if not st.session_state.active_chat:
        title = generate_chat_title_tfidf(query)

        # avoid duplicate names
        base = title
        i = 1
        while title in st.session_state.conversations:
            i += 1
            title = f"{base} ({i})"

        st.session_state.active_chat = title
        st.session_state.conversations[title] = []

    chat = st.session_state.active_chat

    st.session_state.conversations[chat].append({"role":"user","content":query})
    save_message(st.session_state.user, chat, "user", query)

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        if uploaded_file:
            if is_image(uploaded_file):
                st.image(uploaded_file)
                answer = call_gemini_image(query, uploaded_file)

            else:
                text = extract_file_text(uploaded_file)
                answer = explain_medical_file(text, uploaded_file.name, query)
        else:
            answer = get_medical_answer(query)

        st.markdown(answer)

        st.session_state.conversations[chat].append({"role":"assistant","content":answer})
        save_message(st.session_state.user, chat, "assistant", answer)
