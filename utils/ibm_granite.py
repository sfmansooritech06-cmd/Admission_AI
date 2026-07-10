"""
ibm_granite.py – Thin wrapper around the IBM watsonx.ai Text Generation API.

Reads IBM_API_KEY, IBM_PROJECT_ID, and IBM_URL from the environment (.env).
Exposes a single public function: generate_answer(prompt: str) -> str
"""

import os
from dotenv import load_dotenv
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────

IBM_API_KEY    = os.getenv("IBM_API_KEY", "")
IBM_PROJECT_ID = os.getenv("IBM_PROJECT_ID", "")
IBM_URL        = os.getenv("IBM_URL", "https://us-south.ml.cloud.ibm.com")

# Model ID – IBM Granite 13B Instruct v2 (available on Lite plan)
MODEL_ID = "ibm/granite-13b-instruct-v2"

# Generation parameters
GEN_PARAMS = {
    GenParams.DECODING_METHOD:   "greedy",
    GenParams.MAX_NEW_TOKENS:    1024,
    GenParams.MIN_NEW_TOKENS:    10,
    GenParams.TEMPERATURE:       0.1,
    GenParams.REPETITION_PENALTY: 1.1,
    GenParams.STOP_SEQUENCES:    ["<|endoftext|>", "Human:", "Question:"],
}

# ── Internal helpers ──────────────────────────────────────────────────────────

_model: ModelInference | None = None


def _get_model() -> ModelInference:
    """Lazily initialise and cache the ModelInference instance."""
    global _model
    if _model is not None:
        return _model

    if not IBM_API_KEY:
        raise EnvironmentError(
            "IBM_API_KEY is not set. Please configure your .env file."
        )
    if not IBM_PROJECT_ID:
        raise EnvironmentError(
            "IBM_PROJECT_ID is not set. Please configure your .env file."
        )

    credentials = Credentials(
        url=IBM_URL,
        api_key=IBM_API_KEY,
    )

    _model = ModelInference(
        model_id=MODEL_ID,
        credentials=credentials,
        project_id=IBM_PROJECT_ID,
        params=GEN_PARAMS,
    )

    print(f"[ibm_granite] Model '{MODEL_ID}' initialised successfully.")
    return _model


def build_rag_prompt(question: str, context: str) -> str:
    """
    Construct a strict RAG prompt that forces the model to answer *only*
    from the provided context and never hallucinate.
    """
    prompt = f"""You are AdmitAI, an expert College Admission Assistant. \
Your task is to answer student questions about college admissions strictly \
based on the provided context from official college documents.

Rules:
1. Answer ONLY from the context below.
2. If the answer is not in the context, say: "I don't have enough information \
in the provided documents to answer this question. Please refer to the \
official college website or contact the admissions office."
3. Be precise, clear, and structured.
4. Use bullet points or numbered lists when listing multiple items.
5. Always mention the college name when relevant.
6. Do NOT invent fees, dates, or eligibility criteria.

Context from official college documents:
\"\"\"
{context}
\"\"\"

Student Question: {question}

Answer:"""
    return prompt


def generate_answer(question: str, context: str) -> str:
    """
    Generate an answer from IBM Granite LLM given a question and RAG context.

    Parameters
    ----------
    question : str  – The student's question.
    context  : str  – Retrieved document chunks joined as a single string.

    Returns
    -------
    str – The model-generated answer.
    """
    if not question or not question.strip():
        return "Please enter a valid question."

    if not context or not context.strip():
        return (
            "No relevant documents were found to answer your question. "
            "Please ensure college PDF documents have been loaded into the system."
        )

    prompt = build_rag_prompt(question.strip(), context.strip())

    try:
        model  = _get_model()
        result = model.generate_text(prompt=prompt)
        answer = result.strip() if isinstance(result, str) else str(result).strip()
        return answer if answer else "The model returned an empty response. Please try again."

    except EnvironmentError as env_err:
        return f"Configuration Error: {env_err}"

    except Exception as exc:
        error_msg = str(exc)
        if "401" in error_msg or "authentication" in error_msg.lower():
            return (
                "Authentication failed. Please verify your IBM_API_KEY "
                "and IBM_PROJECT_ID in the .env file."
            )
        if "429" in error_msg or "rate limit" in error_msg.lower():
            return (
                "IBM watsonx.ai rate limit reached. "
                "Please wait a moment and try again."
            )
        if "404" in error_msg or "model" in error_msg.lower():
            return (
                f"Model '{MODEL_ID}' not found or not accessible on your IBM Cloud plan. "
                "Please check your watsonx.ai project settings."
            )
        return (
            f"An error occurred while generating the response: {error_msg}. "
            "Please check your IBM credentials and try again."
        )
