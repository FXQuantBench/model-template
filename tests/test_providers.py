"""tests/test_providers.py — unit tests for harness/providers.py.

All three provider libraries (openai, anthropic, google-generativeai) are
mocked via sys.modules so the tests run without any installed packages or
API credentials.
"""

import json
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers — build minimal mock modules for each provider
# ---------------------------------------------------------------------------

def _make_openai_mock(response_content: str | None = None, empty: bool = False):
    """Return a mock openai module whose client returns a structured response."""
    mock_openai = ModuleType("openai")
    client = MagicMock()
    if empty or response_content is None:
        client.chat.completions.create.return_value.choices = []
    else:
        choice = MagicMock()
        choice.message.content = response_content
        client.chat.completions.create.return_value.choices = [choice]
    mock_openai.OpenAI = MagicMock(return_value=client)
    return mock_openai


def _make_anthropic_mock(tool_input: dict | None = None):
    """Return a mock anthropic module."""
    mock_anthropic = ModuleType("anthropic")
    client = MagicMock()

    if tool_input is None:
        # No tool_use block
        msg = MagicMock()
        msg.content = []
        client.messages.create.return_value = msg
    else:
        block = MagicMock()
        block.type = "tool_use"
        block.input = tool_input
        msg = MagicMock()
        msg.content = [block]
        client.messages.create.return_value = msg

    mock_anthropic.Anthropic = MagicMock(return_value=client)
    return mock_anthropic


def _make_google_mock(response_text: str | None = None):
    """Return a mock google.generativeai module."""
    mock_genai = ModuleType("google.generativeai")
    model = MagicMock()
    if response_text is None:
        model.generate_content.return_value.text = ""
    else:
        model.generate_content.return_value.text = response_text
    mock_genai.configure = MagicMock()
    mock_genai.GenerativeModel = MagicMock(return_value=model)
    mock_genai.GenerationConfig = MagicMock(return_value=MagicMock())
    return mock_genai


_VALID_RESPONSE = {
    "thoughts": "Analysing spread patterns in the morning session.",
    "file_changes": [{"path": "strategy.py", "content": "# updated"}],
    "commands": ["/run-backtest"],
}


# ---------------------------------------------------------------------------
# _validate_response
# ---------------------------------------------------------------------------

class TestValidateResponse:
    def setup_method(self):
        from harness.providers import _validate_response
        self._fn = _validate_response

    def test_valid_dict_passes(self):
        result = self._fn(_VALID_RESPONSE, "raw")
        assert result["thoughts"] == _VALID_RESPONSE["thoughts"]

    def test_non_dict_raises(self):
        with pytest.raises(ValueError, match="Expected dict"):
            self._fn(["not", "a", "dict"], "raw")

    def test_missing_thoughts_raises(self):
        bad = {k: v for k, v in _VALID_RESPONSE.items() if k != "thoughts"}
        with pytest.raises(ValueError, match="thoughts"):
            self._fn(bad, "raw")

    def test_missing_file_changes_raises(self):
        bad = {k: v for k, v in _VALID_RESPONSE.items() if k != "file_changes"}
        with pytest.raises(ValueError, match="file_changes"):
            self._fn(bad, "raw")

    def test_missing_commands_raises(self):
        bad = {k: v for k, v in _VALID_RESPONSE.items() if k != "commands"}
        with pytest.raises(ValueError, match="commands"):
            self._fn(bad, "raw")

    def test_empty_thoughts_raises(self):
        bad = dict(_VALID_RESPONSE, thoughts="   ")
        with pytest.raises(ValueError, match="non-empty string"):
            self._fn(bad, "raw")

    def test_file_changes_not_list_raises(self):
        bad = dict(_VALID_RESPONSE, file_changes="not a list")
        with pytest.raises(ValueError, match="file_changes.*list"):
            self._fn(bad, "raw")


# ---------------------------------------------------------------------------
# openai_compatible adapter
# ---------------------------------------------------------------------------

class TestOpenAIAdapter:
    def test_valid_json_response_parsed(self):
        mock_openai = _make_openai_mock(json.dumps(_VALID_RESPONSE))
        with patch.dict(sys.modules, {"openai": mock_openai}):
            from harness.providers import openai_compatible
            result = openai_compatible("gpt-4o", "sk-fake", "sys", "user")
        assert result["thoughts"] == _VALID_RESPONSE["thoughts"]
        assert result["commands"] == ["/run-backtest"]

    def test_empty_choices_raises_value_error(self):
        mock_openai = _make_openai_mock(empty=True)
        with patch.dict(sys.modules, {"openai": mock_openai}):
            from harness.providers import openai_compatible
            with pytest.raises(ValueError, match="empty content"):
                openai_compatible("gpt-4o", "sk-fake", "sys", "user")

    def test_invalid_json_raises_value_error(self):
        mock_openai = _make_openai_mock("not valid json {{{")
        with patch.dict(sys.modules, {"openai": mock_openai}):
            from harness.providers import openai_compatible
            with pytest.raises(ValueError, match="not valid JSON"):
                openai_compatible("gpt-4o", "sk-fake", "sys", "user")

    def test_json_schema_response_format_passed(self):
        mock_openai = _make_openai_mock(json.dumps(_VALID_RESPONSE))
        with patch.dict(sys.modules, {"openai": mock_openai}):
            from harness.providers import openai_compatible
            openai_compatible("gpt-4o", "sk-fake", "sys", "user")
        client_instance = mock_openai.OpenAI.return_value
        call_kwargs = client_instance.chat.completions.create.call_args.kwargs
        assert call_kwargs["response_format"]["type"] == "json_schema"

    def test_model_id_forwarded(self):
        mock_openai = _make_openai_mock(json.dumps(_VALID_RESPONSE))
        with patch.dict(sys.modules, {"openai": mock_openai}):
            from harness.providers import openai_compatible
            openai_compatible("gpt-4-turbo", "sk-fake", "sys", "user")
        client_instance = mock_openai.OpenAI.return_value
        call_kwargs = client_instance.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4-turbo"


# ---------------------------------------------------------------------------
# anthropic_adapter
# ---------------------------------------------------------------------------

class TestAnthropicAdapter:
    def test_tool_use_block_extracted(self):
        mock_anthropic = _make_anthropic_mock(tool_input=_VALID_RESPONSE)
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            from harness.providers import anthropic_adapter
            result = anthropic_adapter("claude-3-5-sonnet-20241022", "sk-fake", "sys", "user")
        assert result["thoughts"] == _VALID_RESPONSE["thoughts"]

    def test_no_tool_use_block_raises(self):
        mock_anthropic = _make_anthropic_mock(tool_input=None)
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            from harness.providers import anthropic_adapter
            with pytest.raises(ValueError, match="tool_use"):
                anthropic_adapter("claude-3-5-sonnet-20241022", "sk-fake", "sys", "user")

    def test_tool_choice_forced(self):
        mock_anthropic = _make_anthropic_mock(tool_input=_VALID_RESPONSE)
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            from harness.providers import anthropic_adapter
            anthropic_adapter("claude-3-5-sonnet-20241022", "sk-fake", "sys", "user")
        client_instance = mock_anthropic.Anthropic.return_value
        call_kwargs = client_instance.messages.create.call_args.kwargs
        assert call_kwargs["tool_choice"] == {"type": "tool", "name": "respond"}


# ---------------------------------------------------------------------------
# google_adapter
# ---------------------------------------------------------------------------

class TestGoogleAdapter:
    def test_valid_json_response_parsed(self):
        mock_genai = _make_google_mock(json.dumps(_VALID_RESPONSE))
        with patch.dict(sys.modules, {
            "google": MagicMock(),
            "google.generativeai": mock_genai,
        }):
            from harness.providers import google_adapter
            result = google_adapter("gemini-1.5-pro", "AIza-fake", "sys", "user")
        assert result["commands"] == ["/run-backtest"]

    def test_empty_response_raises(self):
        mock_genai = _make_google_mock(response_text=None)
        with patch.dict(sys.modules, {
            "google": MagicMock(),
            "google.generativeai": mock_genai,
        }):
            from harness.providers import google_adapter
            with pytest.raises(ValueError, match="empty"):
                google_adapter("gemini-1.5-pro", "AIza-fake", "sys", "user")

    def test_invalid_json_raises(self):
        mock_genai = _make_google_mock("{bad json")
        with patch.dict(sys.modules, {
            "google": MagicMock(),
            "google.generativeai": mock_genai,
        }):
            from harness.providers import google_adapter
            with pytest.raises(ValueError, match="not valid JSON"):
                google_adapter("gemini-1.5-pro", "AIza-fake", "sys", "user")


# ---------------------------------------------------------------------------
# ADAPTERS registry
# ---------------------------------------------------------------------------

class TestAdaptersRegistry:
    def test_all_three_providers_registered(self):
        from harness.providers import ADAPTERS
        assert "openai" in ADAPTERS
        assert "anthropic" in ADAPTERS
        assert "google" in ADAPTERS

    def test_adapters_are_callable(self):
        from harness.providers import ADAPTERS
        for name, fn in ADAPTERS.items():
            assert callable(fn), f"ADAPTERS['{name}'] is not callable"
