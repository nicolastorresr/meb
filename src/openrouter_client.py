"""
openrouter_client.py
OpenRouter API client with retry logic, exponential backoff, and JSON validation.

Response schema (from models):
  {
    "answer":     "<string>",
    "confidence": <int 0–100>,
    "reasoning":  "<string>"   ← captured for future analysis; not used yet
  }
"""

import os
import time
import json
import logging
from typing import Optional

import openai

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a factual question-answering assistant. "
    "You must respond ONLY with a valid JSON object in exactly this format:\n"
    '{"answer": "<your answer>", "confidence": <integer 0-100>, '
    '"reasoning": "<one sentence explaining why you are confident>"}\n'
    "Rules:\n"
    "- Do NOT include any text outside the JSON object.\n"
    "- Do NOT use markdown code fences.\n"
    "- 'answer' must be a concise factual answer (a word, number, or short phrase).\n"
    "- 'confidence' must be an integer between 0 and 100.\n"
    "- 'reasoning' must be a single sentence (≤30 words) justifying your confidence level."
)


def get_client() -> openai.OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY environment variable is not set.")
    return openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def parse_json_response(raw: str) -> dict:
    """
    Parse model response into a dict with answer, confidence, reasoning.
    Strips markdown fences. reasoning is optional for backward compat.
    """
    text = raw.strip()
    if text.startswith("```"):
        lines = [l for l in text.splitlines() if not l.startswith("```")]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON decode error: {exc}. Raw: {raw!r}") from exc

    if "answer" not in data:
        raise ValueError(f"Missing 'answer' key in response: {data}")
    if "confidence" not in data:
        raise ValueError(f"Missing 'confidence' key in response: {data}")

    try:
        data["confidence"] = int(data["confidence"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'confidence' is not an integer: {data['confidence']}") from exc

    if not (0 <= data["confidence"] <= 100):
        logger.warning("Confidence %d out of range, clamping.", data["confidence"])
        data["confidence"] = max(0, min(100, data["confidence"]))

    # reasoning is optional — default to empty string if model omits it
    if "reasoning" not in data:
        data["reasoning"] = ""

    return data


def query_model(
    model_name: str,
    messages: list[dict],
    max_tokens: int = 512,
    temperature: float = 0.0,
    max_retries: int = 3,
    backoff_base: float = 2.0,
    timeout: float = 30.0,
) -> tuple[dict, str]:
    """
    Query a model via OpenRouter with retries and exponential backoff.

    Returns:
        (parsed_dict, raw_text)
        parsed_dict has keys: answer, confidence, reasoning
    """
    client = get_client()
    full_messages = messages
    if not full_messages or full_messages[0].get("role") != "system":
        full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.debug("Querying model=%s attempt=%d/%d", model_name, attempt, max_retries)
            response = client.chat.completions.create(
                model=model_name,
                messages=full_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
            )
            raw_text = response.choices[0].message.content or ""
            parsed = parse_json_response(raw_text)
            return parsed, raw_text
        except (ValueError, openai.APIError, openai.APITimeoutError) as exc:
            last_exc = exc
            wait = backoff_base ** attempt
            logger.warning(
                "Attempt %d/%d failed for %s: %s. Retrying in %.1fs.",
                attempt, max_retries, model_name, exc, wait,
            )
            time.sleep(wait)

    raise RuntimeError(
        f"All {max_retries} attempts failed for model {model_name}. Last: {last_exc}"
    )
