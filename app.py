"""
app.py – Flask application entry point for AdmitAI.

Routes:
    GET  /           → Landing page
    GET  /chat       → Chat interface
    POST /api/ask    → RAG question-answering API
    GET  /api/status → Vector store health check
    GET  /api/colleges → List of available colleges
"""

import os
import uuid
from datetime import datetime
from flask import (
    Flask, render_template, request, jsonify, session
)
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

# ── App setup ─────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "admitai-secret-key-change-in-prod")
CORS(app)

# ── Supported colleges ────────────────────────────────────────────────────────

SUPPORTED_COLLEGES = [
    {"id": "LNCT",        "name": "LNCT Bhopal",         "city": "Bhopal"},
    {"id": "MANIT",       "name": "MANIT Bhopal",         "city": "Bhopal"},
    {"id": "RGPV",        "name": "RGPV",                 "city": "Bhopal"},
    {"id": "DAVV",        "name": "IET DAVV",             "city": "Indore"},
    {"id": "SGSITS",      "name": "SGSITS",               "city": "Indore"},
    {"id": "IPS",         "name": "IPS Academy",          "city": "Indore"},
    {"id": "IIT_Indore",  "name": "IIT Indore",           "city": "Indore"},
    {"id": "IIT_Bombay",  "name": "IIT Bombay",           "city": "Mumbai"},
    {"id": "IIT_Delhi",   "name": "IIT Delhi",            "city": "Delhi"},
    {"id": "IIT_Kanpur",  "name": "IIT Kanpur",           "city": "Kanpur"},
    {"id": "IIT_Kharagpur","name":"IIT Kharagpur",        "city": "Kharagpur"},
    {"id": "IIT_Madras",  "name": "IIT Madras",           "city": "Chennai"},
    {"id": "IIT_Roorkee", "name": "IIT Roorkee",          "city": "Roorkee"},
    {"id": "IIT_Guwahati","name": "IIT Guwahati",         "city": "Guwahati"},
    {"id": "BITS",        "name": "BITS Pilani",          "city": "Pilani"},
    {"id": "VIT",         "name": "VIT",                  "city": "Vellore"},
    {"id": "SRM",         "name": "SRM Institute",        "city": "Chennai"},
    {"id": "LPU",         "name": "LPU",                  "city": "Phagwara"},
    {"id": "Manipal",     "name": "Manipal University",   "city": "Manipal"},
    {"id": "KIIT",        "name": "KIIT",                 "city": "Bhubaneswar"},
    {"id": "Amity",       "name": "Amity University",     "city": "Noida"},
]

SUGGESTED_QUESTIONS = [
    "What is the fee structure for CSE?",
    "What is the MCA eligibility criteria?",
    "How much is the hostel fee?",
    "Which scholarships are available?",
    "What documents are required for admission?",
    "What are the important admission dates?",
    "How can I apply for admission?",
    "What is the reservation policy?",
    "What is the seat matrix?",
    "Compare IIT Indore and MANIT Bhopal.",
    "Which college has lower fees for B.Tech?",
    "What are the placement statistics?",
]

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Landing page."""
    return render_template("index.html", colleges=SUPPORTED_COLLEGES)


@app.route("/chat")
def chat():
    """Chat interface."""
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
        session["chat_history"] = []
    return render_template(
        "chat.html",
        colleges=SUPPORTED_COLLEGES,
        suggested_questions=SUGGESTED_QUESTIONS,
    )


@app.route("/api/ask", methods=["POST"])
def ask():
    """
    RAG question-answering endpoint.

    Request JSON:
        {
            "question":       "What is the fee for CSE?",
            "college_filter": "MANIT"   (optional)
        }

    Response JSON:
        {
            "answer":  "...",
            "sources": [...],
            "question": "...",
            "timestamp": "...",
            "session_id": "..."
        }
    """
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Invalid JSON payload."}), 400

    question       = data.get("question", "").strip()
    college_filter = data.get("college_filter", None)

    if not question:
        return jsonify({"error": "Question cannot be empty."}), 400

    if len(question) > 1000:
        return jsonify({"error": "Question is too long (max 1000 characters)."}), 400

    # Sanitise college filter
    if college_filter:
        valid_ids = {c["id"] for c in SUPPORTED_COLLEGES}
        if college_filter not in valid_ids:
            college_filter = None

    # Run RAG pipeline
    try:
        from utils.rag import answer_question
        answer, sources = answer_question(question, college_filter=college_filter)
    except Exception as exc:
        return jsonify(
            {
                "error": f"Internal server error: {str(exc)}",
                "answer": "An unexpected error occurred. Please try again.",
                "sources": [],
            }
        ), 500

    # Update session chat history (keep last 20 messages)
    if "chat_history" not in session:
        session["chat_history"] = []

    session["chat_history"].append(
        {
            "question":  question,
            "answer":    answer,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )
    session["chat_history"] = session["chat_history"][-20:]
    session.modified = True

    return jsonify(
        {
            "answer":     answer,
            "sources":    sources,
            "question":   question,
            "timestamp":  datetime.utcnow().isoformat(),
            "session_id": session.get("session_id", ""),
        }
    )


@app.route("/api/status", methods=["GET"])
def status():
    """Health check + vector store status."""
    from utils.rag import get_vectorstore_stats
    stats = get_vectorstore_stats()
    return jsonify(
        {
            "app":          "AdmitAI",
            "version":      "1.0.0",
            "vectorstore":  stats,
            "colleges":     len(SUPPORTED_COLLEGES),
            "timestamp":    datetime.utcnow().isoformat(),
        }
    )


@app.route("/api/colleges", methods=["GET"])
def colleges():
    """Return list of supported colleges."""
    return jsonify({"colleges": SUPPORTED_COLLEGES})


@app.route("/api/suggested-questions", methods=["GET"])
def suggested():
    """Return suggested questions."""
    return jsonify({"questions": SUGGESTED_QUESTIONS})


@app.route("/api/clear-chat", methods=["POST"])
def clear_chat():
    """Clear session chat history."""
    session["chat_history"] = []
    session.modified = True
    return jsonify({"message": "Chat history cleared."})


# ── Error handlers ────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found."}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed."}), 405


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error."}), 500


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "False").lower() in ("true", "1")
    port  = int(os.getenv("PORT", 5000))
    print("\n" + "=" * 55)
    print("  AdmitAI – College Admission AI Agent")
    print("  IBM SkillsBuild Hackathon Project")
    print("=" * 55)
    print(f"  Running on http://localhost:{port}")
    print("=" * 55)

    # Startup validation: load (or auto-rebuild) vector store and print banner.
    # This ensures new PDFs are indexed before the first request arrives.
    try:
        from utils.rag import print_startup_banner
        print_startup_banner()
    except Exception as _banner_exc:
        print(f"  [WARN] Startup validation failed: {_banner_exc}")
        print("  Run:   python build_db.py --rebuild\n")

    app.run(host="0.0.0.0", port=port, debug=debug)
