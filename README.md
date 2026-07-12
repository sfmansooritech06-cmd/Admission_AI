# 🎓 AdmitAI – AI Powered College Admission Agent

> IBM Granite + watsonx.ai + RAG + LangChain + FAISS

An AI-powered multi-college admission assistant that answers student queries using official college admission documents through Retrieval-Augmented Generation (RAG).

## IBM Technologies
- IBM Granite
- IBM watsonx.ai
- IBM Cloud Lite

## AI Technologies
- LangChain
- FAISS
- HuggingFace Embeddings
- Flask
---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Tech Stack](#tech-stack)
4. [Project Structure](#project-structure)
5. [Installation](#installation)
6. [Configuration](#configuration)
7. [Building the Vector Store](#building-the-vector-store)
8. [Running the Application](#running-the-application)
9. [API Reference](#api-reference)
10. [Supported Colleges](#supported-colleges)
11. [Deployment on IBM Cloud](#deployment-on-ibm-cloud)
12. [Workflow Diagram](#workflow-diagram)

---

## Overview

AdmitAI is a production-ready **Retrieval-Augmented Generation (RAG)** AI assistant that:

- Answers student questions about college admissions **exclusively from official uploaded PDFs**
- **Never hallucinates** — every answer is grounded in document chunks
- **Cites sources**: College name, document filename, and page number
- Supports **21+ Indian colleges** including IITs, BITS, MANIT, LNCT, VIT, and more
- Covers **14 topic areas**: fees, eligibility, scholarships, hostel, seat matrix, placements, and more

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       AdmitAI System                        │
│                                                             │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │  Flask   │───▶│  RAG Engine  │───▶│  IBM Granite LLM │  │
│  │  Server  │    │  (LangChain) │    │  (watsonx.ai)    │  │
│  └──────────┘    └──────┬───────┘    └──────────────────┘  │
│        ▲                │                                   │
│        │          ┌─────▼──────┐                           │
│  ┌─────┴──────┐   │   FAISS    │                           │
│  │  HTML/JS   │   │ VectorStore│                           │
│  │  Frontend  │   └─────┬──────┘                           │
│  └────────────┘         │                                   │
│                   ┌─────▼──────┐                           │
│                   │ PDF Chunks │                            │
│                   │ (MiniLM    │                            │
│                   │ Embeddings)│                            │
│                   └────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer       | Technology                                      |
|-------------|--------------------------------------------------|
| **Backend** | Python 3.10+, Flask 3.x, Flask-CORS             |
| **AI/LLM**  | IBM Granite 13B Instruct v2 via watsonx.ai      |
| **RAG**     | LangChain 0.2                                   |
| **Vectors** | FAISS (CPU), sentence-transformers/all-MiniLM-L6-v2 |
| **PDF**     | PyPDFLoader (langchain-community)               |
| **Splitter**| RecursiveCharacterTextSplitter                  |
| **Frontend**| HTML5, CSS3 (glassmorphism), Vanilla JS         |
| **Cloud**   | IBM Cloud Lite                                  |

---

## Project Structure

```
College_Admission_AI/
├── app.py                  # Flask application & API routes
├── build_db.py             # Vector store builder script
├── requirements.txt
├── .env                    # Secrets (never commit!)
├── .env.example            # Template for .env
├── README.md
│
├── utils/
│   ├── __init__.py
│   ├── pdf_loader.py       # PDF loading, splitting, metadata tagging
│   ├── ibm_granite.py      # IBM watsonx.ai Granite LLM wrapper
│   └── rag.py              # FAISS vector store + RAG pipeline
│
├── templates/
│   ├── base.html           # Base template with shared styles/fonts
│   ├── index.html          # Landing page
│   └── chat.html           # Chat interface
│
├── static/
│   ├── css/
│   │   ├── style.css       # Landing page styles
│   │   └── chat.css        # Chat interface styles
│   └── js/
│       ├── script.js       # Landing page JS
│       └── chat.js         # Chat JS (send, render, markdown, sources)
│
├── data/                   # Place college PDFs here
│   ├── LNCT/
│   ├── MANIT/
│   ├── RGPV/
│   ├── DAVV/
│   ├── IIT_Indore/
│   ├── IIT_Bombay/
│   └── ...
│
└── vectorstore/            # Auto-generated FAISS index
    └── faiss_index/
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip
- IBM Cloud account (Lite plan works)
- IBM watsonx.ai project with Granite model access

### Step 1 – Clone / Download

```bash
cd College_Admission_AI
```

### Step 2 – Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Step 3 – Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configuration

Copy `.env.example` to `.env` and fill in your IBM credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```dotenv
# IBM watsonx.ai Credentials
IBM_API_KEY=your_actual_ibm_api_key
IBM_PROJECT_ID=your_actual_project_id
IBM_URL=https://us-south.ml.cloud.ibm.com

# Flask
FLASK_SECRET_KEY=change_this_to_a_random_string
FLASK_DEBUG=False

# RAG Settings
CHUNK_SIZE=800
CHUNK_OVERLAP=100
TOP_K_RESULTS=5
```

### Getting IBM Credentials

1. Sign up at [IBM Cloud](https://cloud.ibm.com) (Lite plan – free)
2. Create a **watsonx.ai** project
3. Go to **Manage → Access (IAM)** → Create API Key
4. Copy your **Project ID** from the watsonx.ai project settings
5. The URL is `https://us-south.ml.cloud.ibm.com` for US-South region

---

## Building the Vector Store

### Step 1 – Add PDFs

Place PDF files inside the appropriate college folder under `data/`:

```
data/
  MANIT/
    Admission_Brochure_2024.pdf
    Fee_Structure.pdf
    Hostel_Manual.pdf
  LNCT/
    Prospectus_2024.pdf
    Scholarship_Policy.pdf
  IIT_Indore/
    Admission_Guide.pdf
```

**PDF Naming Tips** (auto-detection):
- `fee*` → fee_structure
- `hostel*` → hostel
- `scholarship*` → scholarship
- `admission*` / `brochure*` → admission_process
- `placement*` → placement
- `seat*` → seat_matrix

### Step 2 – Build the Index

```bash
python build_db.py
```

Output:
```
============================================================
  AdmitAI – Vector Store Builder
============================================================
  Data path     : data
  Store path    : vectorstore/faiss_index
============================================================

[INFO] Found 15 PDF file(s) across all colleges.
[STEP 1] Loading and splitting documents …
[pdf_loader] Loading 3 PDF(s) for MANIT …
  → Admission_Brochure_2024.pdf: 42 page(s)
  → Fee_Structure.pdf: 8 page(s)
...
[pdf_loader] Total chunks after splitting: 1247

============================================================
  ✓ Vector store built successfully!
  Total vectors : 1247
  Time taken    : 34.2 seconds
  Saved to      : vectorstore/faiss_index
============================================================

Next step – start the application:
  python app.py
```

**Force rebuild** (after adding new PDFs):
```bash
python build_db.py --rebuild
```

---

## Running the Application

```bash
python app.py
```

Open your browser at: **http://localhost:5000**

---

## API Reference

### `POST /api/ask`

Ask a question through the RAG pipeline.

**Request:**
```json
{
  "question": "What is the fee structure for CSE at MANIT?",
  "college_filter": "MANIT"
}
```

**Response:**
```json
{
  "answer": "The B.Tech CSE fee at MANIT Bhopal for general category is...",
  "sources": [
    {
      "college_name": "MANIT",
      "pdf_name": "Fee_Structure.pdf",
      "page_number": 7,
      "document_type": "fee_structure"
    }
  ],
  "question": "What is the fee structure for CSE at MANIT?",
  "timestamp": "2024-01-15T10:30:00",
  "session_id": "abc-123"
}
```

### `GET /api/status`

Health check and vector store info.

### `GET /api/colleges`

List of all supported colleges.

### `GET /api/suggested-questions`

Returns suggested questions for the UI.

### `POST /api/clear-chat`

Clears the session chat history.

---

## Supported Colleges

| College | City |
|---------|------|
| LNCT Bhopal | Bhopal |
| MANIT Bhopal | Bhopal |
| RGPV | Bhopal |
| IET DAVV | Indore |
| SGSITS | Indore |
| IPS Academy | Indore |
| IIT Indore | Indore |
| IIT Bombay | Mumbai |
| IIT Delhi | Delhi |
| IIT Kanpur | Kanpur |
| IIT Kharagpur | Kharagpur |
| IIT Madras | Chennai |
| IIT Roorkee | Roorkee |
| IIT Guwahati | Guwahati |
| BITS Pilani | Pilani |
| VIT | Vellore |
| SRM | Chennai |
| LPU | Phagwara |
| Manipal | Manipal |
| KIIT | Bhubaneswar |
| Amity | Noida |

---

## Deployment on IBM Cloud

### IBM Cloud Foundry (Lite)

1. **Install IBM Cloud CLI**
   ```bash
   curl -fsSL https://clis.cloud.ibm.com/install/linux | sh
   ibmcloud login
   ```

2. **Create `manifest.yml`**
   ```yaml
   applications:
     - name: admitai
       memory: 512M
       instances: 1
       buildpack: python_buildpack
       command: gunicorn app:app --workers 2 --bind 0.0.0.0:$PORT
   ```

3. **Create `Procfile`**
   ```
   web: gunicorn app:app --workers 2 --bind 0.0.0.0:$PORT
   ```

4. **Set environment variables on IBM Cloud:**
   ```bash
   ibmcloud cf set-env admitai IBM_API_KEY "your-key"
   ibmcloud cf set-env admitai IBM_PROJECT_ID "your-id"
   ibmcloud cf set-env admitai IBM_URL "https://us-south.ml.cloud.ibm.com"
   ibmcloud cf set-env admitai FLASK_SECRET_KEY "your-secret"
   ```

5. **Push the app:**
   ```bash
   ibmcloud cf push
   ```

> **Note:** Include your pre-built `vectorstore/` in the deployment package, or run `build_db.py` as a pre-startup script.

---

## Workflow Diagram

```
Student Question
      │
      ▼
┌─────────────┐
│  Flask API  │  POST /api/ask
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│  Embed Question     │  MiniLM-L6-v2
│  (768-dim vector)   │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  FAISS Vector Search│  Top-K=5 chunks
│  Similarity Search  │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Build RAG Prompt   │  Question + Context
│  + Metadata         │  + Strict instructions
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  IBM Granite LLM    │  watsonx.ai API
│  granite-13b-       │  Greedy decoding
│  instruct-v2        │  Max 1024 tokens
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Format Response    │  Answer + Sources
│  + Source Citations │  (College/PDF/Page)
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Chat UI Display    │  Markdown render
│  Streaming effect   │  Source cards
└─────────────────────┘
```

---

## Error Handling

| Scenario | Response |
|----------|----------|
| Empty question | 400 with clear message |
| Vector store missing | Informative error + instructions |
| No relevant documents | Honest "not found" message |
| IBM API auth failure | Clear credential error |
| IBM rate limit | Retry suggestion message |
| Network error | User-friendly error bubble in UI |

---

## Security

- All secrets stored in `.env` (never committed)
- `.env` is in `.gitignore`
- User inputs sanitized before processing
- No hardcoded credentials anywhere
- API keys never exposed to frontend

---

*Built with ❤️ for IBM SkillsBuild AI & Cloud Hackathon*
