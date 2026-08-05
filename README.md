# 📚 Universal Hybrid RAG Chatbot

## Overview

This project is a **Universal Hybrid Retrieval-Augmented Generation (RAG) Chatbot** that can answer questions from uploaded documents by retrieving the most relevant information before generating a response.

Unlike a traditional chatbot, the model does **not rely only on its pre-trained knowledge**. Instead, it first searches through the uploaded documents, retrieves the most relevant chunks, reranks them, and then generates a grounded answer based only on the retrieved context.

The system is designed to work with different types of documents such as:

* PDF
* DOCX
* TXT
* Markdown
* Technical Documentation
* Research Papers
* Company Policies
* Books
* Stories
* Legal Documents
* Reports

---

# 🔧 Technologies Used

| Component            | Technology                  |
| -------------------- | --------------------------- |
| Programming Language | Python                      |
| Frontend             | Streamlit                   |
| LLM                  | OpenRouter API              |
| Embedding Model      | BAAI/bge-base-en-v1.5       |
| Vector Database      | FAISS                       |
| Sparse Retrieval     | BM25                        |
| Reranker             | BGE Reranker                |
| Framework            | LangChain                   |
| Document Processing  | Universal Semantic Chunking |

---

# 🧠 How This RAG Works

The workflow of the project is:

```
Upload Documents
        │
        ▼
Document Chunking
        │
        ▼
Generate Embeddings
        │
        ▼
Store in FAISS Vector Database
        │
        ▼
Create BM25 Keyword Index
        │
        ▼
User asks a Question
        │
        ▼
Hybrid Retrieval
(BM25 + FAISS)
        │
        ▼
BGE Reranker
        │
        ▼
Most Relevant Chunks
        │
        ▼
OpenRouter LLM
        │
        ▼
Grounded Response
```

The chatbot answers questions using only the relevant information retrieved from the uploaded documents, helping reduce hallucinations and improve answer accuracy.

---

# 🚀 Running the Project

## 1. Create a Virtual Environment

```bash
python -m venv venv
```

---

## 2. Activate the Virtual Environment

### Windows (Command Prompt)

```cmd
venv\Scripts\activate
```

### Windows (PowerShell)

```powershell
.\venv\Scripts\Activate.ps1
```

---

## 3. Install Required Packages

```bash
pip install -r requirements.txt
```

---

## 4. Start the Application

```bash
streamlit run app.py
```

---

# 📄 Using the Application

1. Open the Streamlit application in your browser.
2. Select the embedding model.
3. Select the reranker model.
4. Select the LLM (OpenRouter model).
5. Upload one or more documents.
6. Click **Process Documents** to build the vector database.
7. Ask questions related to the uploaded documents.
8. The chatbot retrieves the most relevant chunks and generates a grounded answer.

---

# 📌 Features

* Hybrid Retrieval (FAISS + BM25)
* Semantic Document Chunking
* Dense Vector Search
* Keyword-Based Search
* Cross-Encoder Reranking
* Grounded Answer Generation
* Multiple Document Support
* Streamlit User Interface
* OpenRouter LLM Integration
* Modular and Extensible Architecture

---

# 📁 Project Structure

```
project/
│── app.py
│── ingest.py
│── retrieve.py
│── chunking.py
│── reranker.py
│── embeddings.py
│── config.py
│── prompt.py
│── requirements.txt
│── data/
│── vector_db/
│── utils/
└── README.md
```

---

# 🎯 Future Improvements

* Marker-based intelligent document parsing
* Multimodal RAG (Text + Images)
* OCR support for scanned documents
* Parent-Child Retrieval
* Metadata Filtering
* Query Expansion
* Citation Generation
* Conversation Memory
* Agentic AI Workflows

---

## Author

**Sujan**

Computer Science Engineering (Artificial Intelligence)
