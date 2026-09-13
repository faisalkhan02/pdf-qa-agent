# 📄 PDF QA Agent

A simple **PDF Question Answering (QA) Agent** built using **RAG (Retrieval-Augmented Generation)**.

The application allows users to upload a PDF, enter their name, and ask questions about the uploaded document. The system retrieves the most relevant information from the PDF and uses Gemini to generate a concise answer.

## 🚀 Features

- 📤 Upload any PDF document
- 👤 Enter user's name
- 🔍 Extract text from PDF
- ✂️ Split document into smaller chunks
- 🧠 Generate embeddings using Sentence Transformers
- 📚 Store and search embeddings using FAISS
- 🤖 Generate answers using Google Gemini
- 💬 ChatGPT-like chat interface using Streamlit
- 📖 Answers are based only on the uploaded PDF
- 🛡️ Avoids using outside information for document-based questions

## 🛠️ Technologies Used

- Python
- Streamlit
- PyMuPDF
- Sentence Transformers
- FAISS
- NumPy
- Google Gemini API
- python-dotenv

## 🔄 How It Works

```text
          Upload PDF
              ↓
       Extract PDF Text
              ↓
        Clean the Text
              ↓
       Create Text Chunks
              ↓
   Generate Sentence Embeddings
              ↓
        Store in FAISS
              ↓
       User asks a question
              ↓
    Retrieve relevant PDF chunks
              ↓
       Send context to Gemini
              ↓
          Generate Answer