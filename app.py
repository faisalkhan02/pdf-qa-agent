import os
import re
import streamlit as st
import pymupdf
import faiss
import numpy as np

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="PDF Assistant",
    page_icon="📄",
    layout="wide"
)


# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    text-align: center;
    font-size: 36px;
    font-weight: bold;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: gray;
    margin-bottom: 25px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD ENV
# =========================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("❌ GEMINI_API_KEY not found in .env file")
    st.stop()


# =========================================================
# GEMINI
# =========================================================

try:

    client = genai.Client(
        api_key=API_KEY
    )

except Exception as e:

    st.error(f"❌ Gemini initialization error: {e}")
    st.stop()


# =========================================================
# EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


with st.spinner("Loading AI model..."):

    model = load_model()


# =========================================================
# SESSION STATE
# =========================================================

if "user_name" not in st.session_state:
    st.session_state.user_name = ""

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed_file" not in st.session_state:
    st.session_state.processed_file = None


# =========================================================
# PDF TEXT EXTRACTION
# =========================================================

def extract_text(pdf_file):

    try:

        pdf_bytes = pdf_file.getvalue()

        doc = pymupdf.open(
            stream=pdf_bytes,
            filetype="pdf"
        )

        text = ""

        for page in doc:
            page_text = page.get_text("text")

            if page_text:
                text += str(page_text) + "\n"

        doc.close()
        

        return text

    except Exception as e:

        st.error(
            f"PDF extraction error: {e}"
        )

        return ""


# =========================================================
# CLEAN TEXT
# =========================================================

def clean_text(text):

    text = text.replace(
        "\x00",
        " "
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# CHUNKING
# =========================================================

def create_chunks(
    text,
    chunk_size=250,
    overlap=50
):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(
            words[start:end]
        )

        if chunk.strip():

            chunks.append(
                chunk
            )

        start += (
            chunk_size - overlap
        )

    return chunks


# =========================================================
# CREATE FAISS INDEX
# =========================================================

def create_index(chunks):

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    embeddings = embeddings.astype(
        "float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(
        embeddings
    )

    return index


# =========================================================
# PROCESS PDF
# =========================================================

def process_pdf(pdf_file):

    text = extract_text(
        pdf_file
    )

    if not text:

        return False, 0

    text = clean_text(
        text
    )

    chunks = create_chunks(
        text
    )

    if not chunks:

        return False, 0

    index = create_index(
        chunks
    )

    st.session_state.chunks = chunks

    st.session_state.index = index

    st.session_state.messages = []

    return True, len(chunks)


# =========================================================
# RETRIEVE RELEVANT CHUNKS
# =========================================================

def retrieve_chunks(
    question,
    k=5
):

    index = st.session_state.index

    chunks = st.session_state.chunks

    if index is None:

        return []

    if len(chunks) == 0:

        return []

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        show_progress_bar=False
    )

    question_embedding = question_embedding.astype(
        "float32"
    )

    k = min(
        k,
        len(chunks)
    )

    distances, indices = index.search(
        question_embedding,
        k
    )

    results = []

    for i in indices[0]:

        if i >= 0 and i < len(chunks):

            results.append(
                chunks[i]
            )

    return results


# =========================================================
# ASK GEMINI
# =========================================================

def ask_gemini(
    question,
    user_name
):

    # Retrieve document information
    relevant_chunks = retrieve_chunks(
        question,
        k=5
    )

    if not relevant_chunks:
        return (
            f"Sorry {user_name}, "
            "I could not find relevant information "
            "in the uploaded PDF."
        )

    context = "\n\n".join(
        relevant_chunks
    )

    prompt = f"""
You are a PDF Question Answering Assistant.

User name: {user_name}

Answer the user's question using ONLY the
information given in the DOCUMENT CONTEXT.

Rules:
- Do not use outside knowledge.
- Do not invent facts.
- If the answer exists in the context,
  provide the answer clearly.
- If the answer is not available in the context,
  say that you could not find it in the uploaded PDF.
- Address the user naturally by name.
- Keep the answer concise.
- Use bullet points when useful.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    try:

        response = client.interactions.create(
            model="gemini-3.6-flash",
            input=prompt
        )

        if response is None:
            return "❌ Gemini returned no response."

        if not response.output_text:
            return "❌ Gemini returned an empty response."

        return response.output_text.strip()

    except Exception as e:

        return (
            "❌ Gemini API Error:\n\n"
            f"{str(e)}"
        )

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("📄 PDF Assistant")

    st.subheader("👤 Your Name")

    name = st.text_input(
        "Enter your name",
        value=st.session_state.user_name,
        placeholder="e.g. Faisal Khan"
    )

    if name:

        st.session_state.user_name = name.strip()

    st.divider()

    st.subheader("📤 Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose your PDF",
        type=["pdf"]
    )

    if uploaded_file:

        # New PDF
        if (
            st.session_state.processed_file
            != uploaded_file.name
        ):

            with st.spinner(
                "Processing PDF..."
            ):

                success, count = process_pdf(
                    uploaded_file
                )

            if success:

                st.session_state.processed_file = (
                    uploaded_file.name
                )

                st.success(
                    "✅ PDF processed successfully"
                )

                st.info(
                    f"📚 {count} chunks created"
                )

            else:

                st.error(
                    "❌ Could not read this PDF."
                )

    if st.session_state.processed_file:

        st.divider()

        st.write(
            "📄 **Current PDF:**"
        )

        st.write(
            st.session_state.processed_file
        )

        st.write(
            f"📚 Chunks: "
            f"{len(st.session_state.chunks)}"
        )

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# =========================================================
# MAIN PAGE
# =========================================================

st.markdown(
    '<div class="main-title">📄 PDF Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions from your uploaded PDF'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SHOW INSTRUCTIONS
# =========================================================

if not st.session_state.user_name:

    st.info(
        "👈 First enter your name from the sidebar."
    )


elif st.session_state.index is None:

    st.info(
        "👈 Upload a PDF from the sidebar first."
    )


else:

    # =====================================================
    # CHAT HISTORY
    # =====================================================

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

    # =====================================================
    # CHAT INPUT
    # =====================================================

    question = st.chat_input(
        "Ask something about your PDF..."
    )

    if question:

        # Show user question
        with st.chat_message("user"):

            st.markdown(
                question
            )

        # Save question
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        # Generate response
        with st.chat_message("assistant"):

            with st.spinner(
                "🤔 Thinking..."
            ):

                answer = ask_gemini(
                    question,
                    st.session_state.user_name
                )

            st.markdown(
                answer
            )

        # Save answer
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )