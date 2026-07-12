"""
ibm_granite.py  –  IBM watsonx.ai LLM wrapper  (ibm-watsonx-ai >= 1.3)

Boot sequence (lazy, first call to generate_answer):
  1.  Authenticates via APIClient using IBM_API_KEY + IBM_URL from .env.
  2.  Calls client.foundation_models.get_text_generation_model_specs() to get
      the list of text-generation models accessible on the account.  This call
      returns ONLY generation-capable models – embedding / retriever / base
      models are never included.
  3.  Also calls client.foundation_models.get_chat_model_specs() and merges
      the results so that pure-chat instruct models (e.g. Llama-3-3-70b-instruct)
      are included even if they are not in the text-generation list.
  4.  Ranks the merged list by a preference order:
        Granite 3.x instruct  →  Granite 13B instruct  →  Llama 3/4 instruct
        →  Mistral / Mixtral instruct  →  any remaining model
  5.  Iterates through the ranked list, probing each model with a tiny
      generate_text() call until one succeeds.  Skips unavailable models
      automatically and logs the skip reason.
  6.  Caches the working ModelInference + its model-id for the process lifetime.

Public API (unchanged from all previous versions):
    generate_answer(question: str, context: str) -> str
    build_rag_prompt(question: str, context: str) -> str
"""

from __future__ import annotations

import os
import logging
from typing import Any, Optional

from dotenv import load_dotenv
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.schema import (
    TextGenParameters,
    TextGenDecodingMethod,
)

load_dotenv()

log = logging.getLogger(__name__)

# ── Credentials from environment ──────────────────────────────────────────────

IBM_API_KEY    = os.getenv("IBM_API_KEY", "")
IBM_PROJECT_ID = os.getenv("IBM_PROJECT_ID", "")
IBM_URL        = os.getenv("IBM_URL", "https://us-south.ml.cloud.ibm.com").rstrip("/")

# ── Generation parameters (typed schema – SDK >= 1.0 preferred form) ──────────

GEN_PARAMS = TextGenParameters(
    decoding_method=TextGenDecodingMethod.GREEDY,
    max_new_tokens=1024,
    min_new_tokens=10,
    temperature=0.1,
    repetition_penalty=1.1,
    stop_sequences=["<|endoftext|>", "Human:", "Question:"],
)

# ── Model preference ranking ───────────────────────────────────────────────────
# Each entry is a substring matched case-insensitively against the model IDs
# returned by the API.  Only models that survive the generation-model filter
# reach this ranking step.  First match = highest priority.

_PREFERENCE: list[str] = [
    # ── Granite 3.x instruct (newest IBM-native, best for RAG) ────────────────
    "granite-3-3",
    "granite-3-2",
    "granite-3-1",
    "granite-3-0",
    "granite-3",
    # ── Earlier Granite instruct variants ────────────────────────────────────
    "granite-13b-instruct",
    "granite-20b-multilingual",
    "granite-8b-instruct",
    "granite-7b-instruct",
    # ── Any remaining Granite that survived generation filter ─────────────────
    "granite",
    # ── Llama 3 / 4 instruct ─────────────────────────────────────────────────
    "llama-3-3",
    "llama-3-2",
    "llama-3-1",
    "llama-4",
    "llama-3",
    "llama3",
    # ── Mistral / Mixtral instruct ────────────────────────────────────────────
    "mistral",
    "mixtral",
    # ── FLAN (encoder-decoder, generation-capable) ────────────────────────────
    "flan-ul2",
    "flan-t5",
    # ── Falcon / MPT instruct ─────────────────────────────────────────────────
    "falcon",
    "mpt",
]

# ── Region detection ──────────────────────────────────────────────────────────

def _detect_region() -> str:
    """Return the IBM Cloud region derived from IBM_URL."""
    url = IBM_URL.lower()
    for region in (
        "eu-de", "eu-gb", "us-east", "us-south",
        "jp-tok", "au-syd", "ca-tor",
    ):
        if region in url:
            return region
    return "us-south"


# ── APIClient factory ─────────────────────────────────────────────────────────

def _make_client() -> APIClient:
    """
    Create an authenticated APIClient.

    Raises EnvironmentError when credentials are missing.
    """
    if not IBM_API_KEY:
        raise EnvironmentError(
            "IBM_API_KEY is not set. Please add it to your .env file."
        )
    if not IBM_PROJECT_ID:
        raise EnvironmentError(
            "IBM_PROJECT_ID is not set. Please add it to your .env file."
        )
    credentials = Credentials(url=IBM_URL, api_key=IBM_API_KEY)
    return APIClient(credentials=credentials, project_id=IBM_PROJECT_ID)


# ── Model discovery via SDK 1.3+ APIClient ────────────────────────────────────

def _resource_id(resource: dict[str, Any]) -> str:
    """Extract model_id from a spec resource dict."""
    return resource.get("model_id") or resource.get("id") or ""


def _fetch_generation_model_ids(client: APIClient) -> list[str]:
    """
    Return a deduplicated, ranked list of generation-capable model IDs for
    the authenticated account, using the SDK 1.3+ FoundationModelsManager.

    Discovery strategy (stops at the first successful approach):

    Approach 1 – client.foundation_models.get_text_generation_model_specs()
        Returns only text-generation models.  This is the authoritative source.

    Approach 2 – client.foundation_models.get_chat_model_specs()
        Returns instruct / chat models.  Merged with Approach 1 results so that
        pure-chat models (e.g. meta-llama/*-instruct) are included.

    Approach 3 – client.foundation_models.get_model_specs()
        Full catalogue.  Filtered by model-ID keywords to remove embeddings and
        retrievers before returning.  Used only when both targeted calls fail.

    Approach 4 – deprecated module-level get_model_specs(url)
        Kept for compatibility with very old SDK patch versions.

    Approach 5 – ModelTypes enum
        Static list bundled with the SDK.  Needs the same keyword filtering.
    """
    seen:    set[str]  = set()
    ordered: list[str] = []

    def _add(model_id: str) -> None:
        if model_id and model_id not in seen:
            seen.add(model_id)
            ordered.append(model_id)

    def _ids_from_response(response: Any) -> list[str]:
        if not response:
            return []
        if isinstance(response, dict):
            resources = response.get("resources", [])
        elif hasattr(response, "__iter__"):
            resources = list(response)
        else:
            return []
        return [_resource_id(r) for r in resources if isinstance(r, dict)]

    # ── Approach 1: text-generation specs (SDK >= 1.3) ────────────────────────
    try:
        resp = client.foundation_models.get_text_generation_model_specs(get_all=True)
        ids  = _ids_from_response(resp)
        if ids:
            for m in ids:
                _add(m)
            log.info(
                "[ibm_granite] get_text_generation_model_specs → %d models", len(ids)
            )
    except Exception as exc:
        log.debug("[ibm_granite] get_text_generation_model_specs failed: %s", exc)

    # ── Approach 2: chat / instruct specs (SDK >= 1.3) ────────────────────────
    try:
        resp = client.foundation_models.get_chat_model_specs(get_all=True)
        ids  = _ids_from_response(resp)
        if ids:
            for m in ids:
                _add(m)
            log.info(
                "[ibm_granite] get_chat_model_specs → %d models (merged)", len(ids)
            )
    except Exception as exc:
        log.debug("[ibm_granite] get_chat_model_specs failed: %s", exc)

    if ordered:
        return ordered

    # ── Approach 3: full catalogue with client (SDK >= 1.0) ───────────────────
    try:
        resp = client.foundation_models.get_model_specs(get_all=True)
        ids  = _ids_from_response(resp)
        ids  = _filter_generation_ids(ids)
        if ids:
            for m in ids:
                _add(m)
            log.info(
                "[ibm_granite] get_model_specs (filtered) → %d generation models",
                len(ids),
            )
            return ordered
    except Exception as exc:
        log.debug("[ibm_granite] client.foundation_models.get_model_specs failed: %s", exc)

    # ── Approach 4: deprecated module-level function ──────────────────────────
    try:
        from ibm_watsonx_ai.foundation_models import get_model_specs as _gms
        resp = _gms(url=IBM_URL)
        ids  = _ids_from_response(resp)
        ids  = _filter_generation_ids(ids)
        if ids:
            for m in ids:
                _add(m)
            log.info(
                "[ibm_granite] module-level get_model_specs (filtered) → %d models",
                len(ids),
            )
            return ordered
    except Exception as exc:
        log.debug("[ibm_granite] module-level get_model_specs failed: %s", exc)

    # ── Approach 5: static ModelTypes enum ───────────────────────────────────
    try:
        from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
        ids = [e.value for e in ModelTypes]
        ids = _filter_generation_ids(ids)
        if ids:
            for m in ids:
                _add(m)
            log.info(
                "[ibm_granite] ModelTypes enum (filtered) → %d models", len(ids)
            )
            return ordered
    except Exception as exc:
        log.debug("[ibm_granite] ModelTypes enum failed: %s", exc)

    return ordered  # may be empty; caller handles that case


# ── ID-based keyword filter (used only for Approaches 3-5) ───────────────────

_EXCLUDE_SUBSTRINGS: tuple[str, ...] = (
    "embedding", "embed", "rtrvr", "retriev",
    "rerank",    "encode",
    "-base",     "-h-small",
    "slate",     "sentence", "colbert", "bi-encoder",
)


def _filter_generation_ids(ids: list[str]) -> list[str]:
    """
    Remove model IDs that are clearly not text-generation models.
    Used as a fallback when the API call doesn't already scope to generation.
    """
    result: list[str] = []
    for mid in ids:
        low = mid.lower()
        if not any(kw in low for kw in _EXCLUDE_SUBSTRINGS):
            result.append(mid)
    return result


# ── Preference ranking ────────────────────────────────────────────────────────

def _rank_model(model_id: str) -> int:
    """Return the preference rank for *model_id* (lower = higher priority)."""
    low = model_id.lower()
    for i, kw in enumerate(_PREFERENCE):
        if kw in low:
            return i
    return len(_PREFERENCE)


def _rank_candidates(ids: list[str]) -> list[str]:
    """Return *ids* sorted by preference, best first."""
    return sorted(ids, key=_rank_model)


# ── Cached model instance ─────────────────────────────────────────────────────

_model:    Optional[ModelInference] = None
_model_id: Optional[str]            = None


def _get_model() -> ModelInference:
    """
    Lazily build and cache a working ModelInference instance.

    On each process start this function:
      1. Creates an APIClient (authenticates with IBM Cloud).
      2. Calls _fetch_generation_model_ids() to get only generation models.
      3. Sorts candidates by preference (_PREFERENCE list).
      4. Probes each candidate with generate_text("Hello") until one succeeds.
      5. Caches the working instance in module-level globals.

    Raises
    ------
    EnvironmentError – IBM_API_KEY or IBM_PROJECT_ID missing from .env
    RuntimeError     – All candidate models failed (logged individually)
    """
    global _model, _model_id

    if _model is not None:
        return _model

    region = _detect_region()
    print(f"[ibm_granite] IBM Cloud region  : {region}")
    print(f"[ibm_granite] Endpoint URL      : {IBM_URL}")

    # ── Step 1: Authenticate ──────────────────────────────────────────────────
    # Raises EnvironmentError if credentials are absent.
    client = _make_client()

    # ── Step 2: Discover generation models ───────────────────────────────────
    raw_ids    = _fetch_generation_model_ids(client)
    candidates = _rank_candidates(raw_ids)

    print(f"[ibm_granite] Generation models available : {len(candidates)}")

    if not candidates:
        raise RuntimeError(
            "No text-generation foundation models were found for this IBM Cloud "
            "account. Please verify your watsonx.ai project settings and plan."
        )

    print(f"[ibm_granite] Ranked #1 candidate         : {candidates[0]}")
    if len(candidates) > 1:
        top3 = candidates[1:4]
        print(f"[ibm_granite] Fallback candidates         : {top3}")

    # ── Step 3: Probe candidates ──────────────────────────────────────────────
    credentials = Credentials(url=IBM_URL, api_key=IBM_API_KEY)
    last_exc: Optional[Exception] = None

    for candidate_id in candidates:
        try:
            instance = ModelInference(
                model_id=candidate_id,
                credentials=credentials,
                project_id=IBM_PROJECT_ID,
                params=GEN_PARAMS,
                # validate=True  →  SDK validates the model_id against the
                # catalogue on construction (default in 1.4.x).  Keep it on so
                # genuinely invalid IDs fail fast here rather than at generate.
                validate=True,
            )
            # Probe: a real generate_text call confirms the model works end-to-end.
            # Some models pass construction validation but are restricted on the
            # specific IBM Cloud plan or region.
            probe_result = instance.generate_text(prompt="Hello")

            # generate_text returns str on success; empty string or None = failure
            if probe_result is not None:
                _model    = instance
                _model_id = candidate_id
                print(f"[ibm_granite] ✓ Active model : {_model_id}")
                log.info("[ibm_granite] Model selected and verified: %s", _model_id)
                return _model

            log.warning(
                "[ibm_granite] Candidate '%s' returned empty probe response – skipping.",
                candidate_id,
            )

        except Exception as exc:
            log.warning(
                "[ibm_granite] Candidate '%s' failed: %s – trying next.",
                candidate_id,
                exc,
            )
            last_exc = exc

    raise RuntimeError(
        f"Tried {len(candidates)} generation model(s) — none responded successfully. "
        "Please check your IBM Cloud plan and watsonx.ai project permissions. "
        f"Last SDK error: {last_exc}"
    )


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_rag_prompt(question: str, context: str) -> str:
    """
    Construct a strict RAG prompt that instructs the model to answer FULLY
    from the supplied *context*, minimising "I don't know" responses when
    the context actually contains relevant information.
    """
    return (
        "You are AdmitAI, an expert College Admission Assistant for Indian colleges.\n"
        "Answer student questions about college admissions using ONLY the context "
        "extracted from official college documents provided below.\n\n"
        "RULES (follow strictly):\n"
        "1. Base your answer ENTIRELY on the context below. Do not add outside knowledge.\n"
        "2. If the context contains PARTIAL information, provide what is available and "
        "note that some details may be incomplete.\n"
        "3. Say 'I don't have enough information in the provided documents' ONLY when "
        "the context contains ZERO relevant information for the question.\n"
        "4. Be precise, clear, and well-structured in your answer.\n"
        "5. Use bullet points or numbered lists when listing multiple items "
        "(fees, eligibility, courses, dates, etc.).\n"
        "6. Always mention the college name when citing specific data.\n"
        "7. Do NOT invent or assume fees, dates, ranks, or eligibility criteria.\n"
        "8. For comparison questions, structure the answer as a table or side-by-side list.\n"
        "9. For topic categories like Fees / Hostel / Placement / Scholarship / Eligibility "
        "/ Cutoff / Courses — extract ALL relevant numbers, dates, and conditions.\n\n"
        "Context from official college documents:\n"
        "'''\n"
        f"{context}\n"
        "'''\n\n"
        f"Student Question: {question}\n\n"
        "Detailed Answer based on the context above:"
    )


# ── Public API ────────────────────────────────────────────────────────────────

def generate_answer(question: str, context: str) -> str:
    """
    Generate an answer from the IBM watsonx.ai LLM using the RAG context.

    Parameters
    ----------
    question : str  – The student's natural-language question.
    context  : str  – Retrieved document chunks joined as a single string.

    Returns
    -------
    str – The model answer, or a user-friendly error message.
          Raw Python exceptions are never propagated to the caller.
    """
    if not question or not question.strip():
        return "Please enter a valid question."

    if not context or not context.strip():
        return (
            "No relevant documents were found for your question. "
            "Please ensure college PDF documents have been loaded into the system."
        )

    prompt = build_rag_prompt(question.strip(), context.strip())

    try:
        model  = _get_model()
        result = model.generate_text(prompt=prompt)
        answer = result.strip() if isinstance(result, str) else str(result).strip()
        return answer or "The model returned an empty response. Please try again."

    except EnvironmentError as err:
        log.error("[ibm_granite] Credential error: %s", err)
        return (
            "Configuration Error: IBM credentials are missing or incomplete. "
            "Please set IBM_API_KEY and IBM_PROJECT_ID in your .env file."
        )

    except RuntimeError as err:
        log.error("[ibm_granite] Model init error: %s", err)
        return (
            "Service Unavailable: AdmitAI could not connect to IBM watsonx.ai. "
            "Please verify your IBM Cloud account and project settings, then try again."
        )

    except Exception as exc:
        msg       = str(exc)
        msg_lower = msg.lower()
        log.error("[ibm_granite] Unexpected error during generation: %s", msg)

        if any(k in msg_lower for k in ("401", "authentication", "unauthorized", "api key")):
            return (
                "Authentication Failed: Your IBM API key was rejected. "
                "Please verify IBM_API_KEY in your .env file."
            )

        if any(k in msg_lower for k in ("403", "forbidden", "not authorized", "access denied")):
            return (
                "Access Denied: Your IBM Cloud account does not have permission "
                "to use this model. Please check your watsonx.ai plan and project settings."
            )

        if any(k in msg_lower for k in ("429", "rate limit", "too many requests", "quota")):
            return (
                "Rate Limit Reached: IBM watsonx.ai is temporarily throttling requests. "
                "Please wait a moment and try again."
            )

        if any(k in msg_lower for k in ("timeout", "timed out", "connection", "network")):
            return (
                "Connection Error: Could not reach IBM watsonx.ai. "
                "Please check your internet connection and try again."
            )

        if any(k in msg_lower for k in ("404", "not found")):
            return (
                "Model Not Found: The selected model is not accessible on your account. "
                "Please check your IBM Cloud plan."
            )

        return (
            "An unexpected error occurred while generating the response. "
            "Please try again. If the problem persists, check your IBM Cloud settings."
        )
