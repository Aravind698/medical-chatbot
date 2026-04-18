
import os
import streamlit as st
import requests
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
import PyPDF2
import docx

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
GEMINI_API_KEY = st.secrets["gemini_api_key"]
MODEL_NAME = "models/gemini-2.5-flash"
PINECONE_API_KEY = st.secrets["pinecone_api_key"]

INDEX_NAME = "medical-chatbot"

# ---------------------------------------------------------
# LOAD EMBEDDINGS
# ---------------------------------------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = load_embeddings()

# ---------------------------------------------------------
# CONNECT TO PINECONE
# ---------------------------------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# ---------------------------------------------------------
# GEMINI CALL
# ---------------------------------------------------------
def call_gemini(prompt: str):
    url = f"https://generativelanguage.googleapis.com/v1/{MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    elif response.status_code == 429:
        return "Daily Gemini quota exceeded. Please try again tomorrow."
    else:
        return f"Error {response.status_code}: {response.text}"

# ---------------------------------------------------------
# RAG FUNCTION (for text queries)
# ---------------------------------------------------------
def get_medical_answer(query):
    query_vector = embeddings.embed_query(query)
    results = index.query(vector=query_vector, top_k=5, include_metadata=True)
    matches = results.get("matches", [])
    contexts = [m["metadata"].get("text", "") for m in matches if m["metadata"].get("text")]
    scores = [m["score"] for m in matches]
    confidence = max(scores) if scores else 0.0
    confidence_percent = confidence * 100
    context_text = "\n\n".join(contexts)

    prompt = f"""
You are a medical expert chatbot.
First, interpret the user's query and clarify what problem they are asking about.
Use the retrieved context to ground your answer.
Respond in the following structured format:

Overview:
Causes:
Symptoms:
Treatment:
Avoidable foods:

Context:
{context_text}

User query:
{query}
"""
    answer = call_gemini(prompt)
    answer += f"\n\n**Confidence score (retrieval): {confidence_percent:.1f}%**"
    return answer

# ---------------------------------------------------------
# FILE HANDLING
# ---------------------------------------------------------
def extract_file_text(uploaded_file):
    if uploaded_file.type == "text/plain":
        return uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "application/pdf":
        reader = PyPDF2.PdfReader(uploaded_file)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])
    return None

# ---------------------------------------------------------
# FILE ANALYSIS FUNCTION
# ---------------------------------------------------------
def explain_medical_file(file_text: str, filename: str, query: str):
    prompt = f"""
You are a medical expert chatbot.
The user uploaded a medical-related file: {filename}.
They now ask: {query}

Read and interpret the file contents carefully and answer their query.

Provide a structured explanation:
Summary of the document:
Key findings:
Any diseases or medical conditions mentioned:
Explanation of those conditions in simple terms:
Suggested next steps (general, not personal medical advice):

File content:
{file_text}
"""
    return call_gemini(prompt)

# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
st.set_page_config(page_title="Medical Chatbot", page_icon="🩺", layout="wide")
st.title("🩺 Medical Chatbot")
st.markdown("---")

# Load external CSS
with open("style.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if "conversations" not in st.session_state:
    st.session_state.conversations = {}
    st.session_state.active_chat = None

st.sidebar.title("💬 Chats")
if st.sidebar.button("➕ New Chat"):
    new_chat_title = f"Untitled Chat {len(st.session_state.conversations) + 1}"
    st.session_state.conversations[new_chat_title] = []
    st.session_state.active_chat = new_chat_title

if st.session_state.conversations:
    chat_titles = list(st.session_state.conversations.keys())
    for title in chat_titles:
        if st.sidebar.button(title, key=f"chat_{title}"):
            st.session_state.active_chat = title

if st.session_state.active_chat:
    for msg in st.session_state.conversations[st.session_state.active_chat]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# Fixed bottom query bar with file uploader side by side
query_area = st.container()
with query_area:
    cols = st.columns([1, 6, 3])
    with cols[0]:
        uploaded_file = st.file_uploader(
            "",
            type=["pdf", "txt", "docx"],
            label_visibility="collapsed",
            key="file_upload"
        )
    with cols[1]:
        query = st.chat_input("Ask a medical question...")
    with cols[2]:
        if uploaded_file is not None:
            st.markdown(f"📎 **{uploaded_file.name}**")

# Handle query (text or file + query)
if query:
    if not st.session_state.active_chat:
        new_chat_title = query.split()[0].capitalize()
        st.session_state.conversations[new_chat_title] = []
        st.session_state.active_chat = new_chat_title

    st.session_state.conversations[st.session_state.active_chat].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if uploaded_file is not None:
                file_text = extract_file_text(uploaded_file)
                answer = explain_medical_file(file_text, uploaded_file.name, query)
            else:
                answer = get_medical_answer(query)
            st.markdown(answer)
            st.session_state.conversations[st.session_state.active_chat].append({"role": "assistant", "content": answer})
