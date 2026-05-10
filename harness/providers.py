"""harness/providers.py — LLM provider adapters with constrained/structured output.

Each adapter has the signature:
    (model_id: str, api_key: str, system_prompt: str, user_message: str) -> dict

The returned dict always has the shape:
    {
        "thoughts": str,
        "file_changes": [{"path": str, "content": str}, ...],
        "commands": [str, ...]
    }

Raises ValueError(raw_response) if the provider returns a malformed or empty response.
"""

from __future__ import annotations

import json
from typing import Any

# ---------------------------------------------------------------------------
# JSON Schema shared across all providers
# ---------------------------------------------------------------------------

_FILE_CHANGE_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "content": {"type": "string"},
    },
    "required": ["path", "content"],
    "additionalProperties": False,
}

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "thoughts": {"type": "string"},
        "file_changes": {
            "type": "array",
            "items": _FILE_CHANGE_SCHEMA,
        },
        "commands": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["thoughts", "file_changes", "commands"],
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def _validate_response(obj: Any, raw: str) -> dict:
    """Verify parsed object matches expected schema shape; raise ValueError on failure."""
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict response, got {type(obj).__name__}. Raw: {raw}")
    for key in ("thoughts", "file_changes", "commands"):
        if key not in obj:
            raise ValueError(f"Response missing key '{key}'. Raw: {raw}")
    if not isinstance(obj["thoughts"], str) or not obj["thoughts"].strip():
        raise ValueError(f"'thoughts' must be a non-empty string. Raw: {raw}")
    if not isinstance(obj["file_changes"], list):
        raise ValueError(f"'file_changes' must be a list. Raw: {raw}")
    if not isinstance(obj["commands"], list):
        raise ValueError(f"'commands' must be a list. Raw: {raw}")
    return obj


# ---------------------------------------------------------------------------
# Adapter: OpenAI-compatible (OpenAI, Mistral, any OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------

def openai_compatible(
    model_id: str,
    api_key: str,
    system_prompt: str,
    user_message: str,
) -> dict:
    """Use JSON Schema response_format for strictly-structured output."""
    import openai  # noqa: PLC0415

    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model_id,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "strategy_response",
                "strict": True,
                "schema": _RESPONSE_SCHEMA,
            },
        },
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )

    raw = response.choices[0].message.content if response.choices else ""
    if not raw:
        raise ValueError(f"OpenAI returned empty content. Full response: {response}")

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"OpenAI response is not valid JSON: {exc}. Raw: {raw}") from exc

    return _validate_response(obj, raw)


# ---------------------------------------------------------------------------
# Adapter: Anthropic (tool-use forced call)
# ---------------------------------------------------------------------------

_ANTHROPIC_TOOL_DEF = {
    "name": "respond",
    "description": "Emit your structured response for this agentic loop iteration.",
    "input_schema": _RESPONSE_SCHEMA,
}


def anthropic_adapter(
    model_id: str,
    api_key: str,
    system_prompt: str,
    user_message: str,
) -> dict:
    """Force the model to call the 'respond' tool; extract tool_use input block."""
    import anthropic  # noqa: PLC0415

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model_id,
        max_tokens=8192,
        system=system_prompt,
        tools=[_ANTHROPIC_TOOL_DEF],
        tool_choice={"type": "tool", "name": "respond"},
        messages=[{"role": "user", "content": user_message}],
    )

    # Find the tool_use content block
    tool_block = next(
        (block for block in response.content if block.type == "tool_use"),
        None,
    )
    if tool_block is None:
        raw = str(response.content)
        raise ValueError(f"Anthropic response contained no tool_use block. Raw: {raw}")

    obj = tool_block.input  # already a dict
    raw = json.dumps(obj)
    return _validate_response(obj, raw)


# ---------------------------------------------------------------------------
# Adapter: Google Gemini
# ---------------------------------------------------------------------------

def google_adapter(
    model_id: str,
    api_key: str,
    system_prompt: str,
    user_message: str,
) -> dict:
    """Use response_mime_type + response_schema for structured JSON output."""
    import google.generativeai as genai  # noqa: PLC0415

    genai.configure(api_key=api_key)

    generation_config = genai.GenerationConfig(
        response_mime_type="application/json",
        response_schema=_RESPONSE_SCHEMA,
    )

    model = genai.GenerativeModel(
        model_name=model_id,
        system_instruction=system_prompt,
        generation_config=generation_config,
    )

    response = model.generate_content(user_message)

    raw = response.text if hasattr(response, "text") else ""
    if not raw:
        raise ValueError(f"Google returned empty response. Full response: {response}")

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Google response is not valid JSON: {exc}. Raw: {raw}") from exc

    return _validate_response(obj, raw)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ADAPTERS: dict[str, Any] = {
    "openai": openai_compatible,
    "anthropic": anthropic_adapter,
    "google": google_adapter,
}
