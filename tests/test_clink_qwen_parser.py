"""Tests for the Qwen stream-json parser."""

import pytest

from clink.parsers.base import ParserError
from clink.parsers.qwen import QwenStreamJsonParser


class TestQwenStreamJsonParser:
    """Tests for QwenStreamJsonParser."""

    def test_parses_success_result(self):
        """Parser extracts content from result message."""
        stdout = "\n".join(
            [
                '{"type":"system","subtype":"init","uuid":"abc","session_id":"sess-1","model":"qwen-coder","qwen_code_version":"0.9.0"}',
                '{"type":"assistant","uuid":"msg-1","session_id":"sess-1","parent_tool_use_id":null,"message":{"id":"msg-1","type":"message","role":"assistant","model":"qwen-coder","content":[{"type":"text","text":"Analysis complete."}],"stop_reason":null,"usage":{"input_tokens":100,"output_tokens":50}}}',
                '{"type":"result","subtype":"success","uuid":"res-1","session_id":"sess-1","is_error":false,"duration_ms":1000,"duration_api_ms":800,"num_turns":1,"result":"Analysis complete.","usage":{"input_tokens":100,"output_tokens":50},"permission_denials":[]}',
            ]
        )

        parser = QwenStreamJsonParser()
        result = parser.parse(stdout, "")

        assert result.content == "Analysis complete."
        assert result.metadata["session_id"] == "sess-1"
        assert result.metadata["model"] == "qwen-coder"
        assert result.metadata["cli_version"] == "0.9.0"
        assert result.metadata["is_error"] is False
        assert result.metadata["num_turns"] == 1
        assert result.metadata["duration_ms"] == 1000
        assert len(result.metadata["raw_events"]) == 3

    def test_parses_error_result(self):
        """Parser handles error results correctly."""
        stdout = "\n".join(
            [
                '{"type":"system","subtype":"init","uuid":"abc","session_id":"sess-1","model":"qwen-coder"}',
                '{"type":"result","subtype":"error_during_execution","uuid":"res-1","session_id":"sess-1","is_error":true,"duration_ms":500,"duration_api_ms":0,"num_turns":1,"usage":{},"permission_denials":[],"error":{"message":"Rate limit exceeded"}}',
            ]
        )

        parser = QwenStreamJsonParser()
        result = parser.parse(stdout, "")

        assert result.content == "Rate limit exceeded"
        assert result.metadata["is_error"] is True
        assert result.metadata["error"]["message"] == "Rate limit exceeded"

    def test_falls_back_to_assistant_message(self):
        """Parser extracts text from assistant message when no result."""
        stdout = "\n".join(
            [
                '{"type":"system","subtype":"init","uuid":"abc","session_id":"sess-1","model":"qwen-coder"}',
                '{"type":"assistant","uuid":"msg-1","session_id":"sess-1","parent_tool_use_id":null,"message":{"id":"msg-1","type":"message","role":"assistant","model":"qwen-coder","content":[{"type":"text","text":"Here is the answer."}],"stop_reason":"end_turn","usage":{"input_tokens":50,"output_tokens":20}}}',
            ]
        )

        parser = QwenStreamJsonParser()
        result = parser.parse(stdout, "")

        assert result.content == "Here is the answer."

    def test_handles_thinking_blocks(self):
        """Parser includes thinking content from assistant messages."""
        stdout = "\n".join(
            [
                '{"type":"system","subtype":"init","uuid":"abc","session_id":"sess-1","model":"qwen-coder"}',
                '{"type":"assistant","uuid":"msg-1","session_id":"sess-1","parent_tool_use_id":null,"message":{"id":"msg-1","type":"message","role":"assistant","model":"qwen-coder","content":[{"type":"thinking","thinking":"Let me analyze this..."},{"type":"text","text":"The answer is 42."}],"stop_reason":"end_turn","usage":{}}}',
            ]
        )

        parser = QwenStreamJsonParser()
        result = parser.parse(stdout, "")

        # When falling back to assistant message, thinking is included
        assert "Let me analyze this" in result.content
        assert "The answer is 42" in result.content

    def test_raises_on_empty_output(self):
        """Parser raises error on empty output."""
        parser = QwenStreamJsonParser()

        with pytest.raises(ParserError, match="empty stdout"):
            parser.parse("", "")

        with pytest.raises(ParserError, match="empty stdout"):
            parser.parse("   \n   ", "")

    def test_raises_on_no_result_or_assistant(self):
        """Parser raises error when no result or assistant message found."""
        stdout = '{"type":"system","subtype":"init","uuid":"abc","session_id":"sess-1","model":"qwen-coder"}'

        parser = QwenStreamJsonParser()

        with pytest.raises(ParserError, match="did not contain a result or assistant"):
            parser.parse(stdout, "")

    def test_handles_stderr_fallback(self):
        """Parser returns stderr message when no other content."""
        stdout = '{"type":"system","subtype":"init","uuid":"abc","session_id":"sess-1","model":"qwen-coder"}'

        parser = QwenStreamJsonParser()
        result = parser.parse(stdout, "Some error occurred in stderr")

        assert "stderr was preserved" in result.content
        assert result.metadata["stderr"] == "Some error occurred in stderr"

    def test_preserves_usage_metadata(self):
        """Parser extracts usage information from result."""
        stdout = "\n".join(
            [
                '{"type":"system","subtype":"init","uuid":"abc","session_id":"sess-1","model":"qwen-coder"}',
                '{"type":"result","subtype":"success","uuid":"res-1","session_id":"sess-1","is_error":false,"duration_ms":1000,"duration_api_ms":800,"num_turns":2,"result":"Done.","usage":{"input_tokens":500,"output_tokens":200,"cache_read_input_tokens":100},"permission_denials":[]}',
            ]
        )

        parser = QwenStreamJsonParser()
        result = parser.parse(stdout, "")

        assert result.metadata["usage"]["input_tokens"] == 500
        assert result.metadata["usage"]["output_tokens"] == 200
        assert result.metadata["usage"]["cache_read_input_tokens"] == 100

    def test_handles_model_usage(self):
        """Parser extracts model-specific usage information."""
        stdout = "\n".join(
            [
                '{"type":"system","subtype":"init","uuid":"abc","session_id":"sess-1","model":"qwen-coder"}',
                '{"type":"result","subtype":"success","uuid":"res-1","session_id":"sess-1","is_error":false,"duration_ms":1000,"duration_api_ms":800,"num_turns":1,"result":"Done.","usage":{},"modelUsage":{"qwen-coder":{"inputTokens":100,"outputTokens":50}},"permission_denials":[]}',
            ]
        )

        parser = QwenStreamJsonParser()
        result = parser.parse(stdout, "")

        assert result.metadata["model_used"] == "qwen-coder"
        assert "qwen-coder" in result.metadata["model_usage"]

    def test_skips_malformed_json_lines(self):
        """Parser skips lines that aren't valid JSON."""
        stdout = "\n".join(
            [
                "Some random text",
                '{"type":"system","subtype":"init","uuid":"abc","session_id":"sess-1","model":"qwen-coder"}',
                "More random text",
                '{"type":"result","subtype":"success","uuid":"res-1","session_id":"sess-1","is_error":false,"duration_ms":1000,"num_turns":1,"result":"Success!","usage":{},"permission_denials":[]}',
                "trailing garbage",
            ]
        )

        parser = QwenStreamJsonParser()
        result = parser.parse(stdout, "")

        assert result.content == "Success!"
        assert len(result.metadata["raw_events"]) == 2  # Only valid JSON objects

    def test_handles_permission_denials(self):
        """Parser extracts permission denials from result."""
        stdout = "\n".join(
            [
                '{"type":"system","subtype":"init","uuid":"abc","session_id":"sess-1","model":"qwen-coder"}',
                '{"type":"result","subtype":"success","uuid":"res-1","session_id":"sess-1","is_error":false,"duration_ms":1000,"num_turns":1,"result":"Done with denials.","usage":{},"permission_denials":[{"tool_name":"bash","tool_use_id":"tu-1","tool_input":{"command":"rm -rf /"}}]}',
            ]
        )

        parser = QwenStreamJsonParser()
        result = parser.parse(stdout, "")

        assert len(result.metadata["permission_denials"]) == 1
        assert result.metadata["permission_denials"][0]["tool_name"] == "bash"
