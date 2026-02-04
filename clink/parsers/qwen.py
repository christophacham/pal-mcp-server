"""Parser for Qwen CLI stream-json output (JSONL format)."""

from __future__ import annotations

import json
from typing import Any

from .base import BaseParser, ParsedCLIResponse, ParserError


class QwenStreamJsonParser(BaseParser):
    """Parse stdout produced by `qwen -o stream-json`.

    In stream-json mode, Qwen CLI outputs one JSON object per line (JSONL).
    Each line is a complete message: system, assistant, user, or result.

    This avoids the issue where `-o json` mode buffers the entire conversation
    history (potentially hundreds of KB) and writes it all at once, which can
    cause EPIPE errors and truncated output.
    """

    name = "qwen_stream_json"

    def parse(self, stdout: str, stderr: str) -> ParsedCLIResponse:
        if not stdout.strip():
            raise ParserError("Qwen CLI returned empty stdout while stream-json output was expected")

        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        events: list[dict[str, Any]] = []
        result_message: dict[str, Any] | None = None
        last_assistant_message: dict[str, Any] | None = None
        system_message: dict[str, Any] | None = None
        errors: list[str] = []

        for line in lines:
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            events.append(event)
            event_type = event.get("type")

            if event_type == "result":
                result_message = event
            elif event_type == "assistant":
                last_assistant_message = event
            elif event_type == "system":
                system_message = event

        # Build metadata
        metadata: dict[str, Any] = {"raw_events": events}

        if system_message:
            metadata["session_id"] = system_message.get("session_id")
            metadata["model"] = system_message.get("model")
            if "qwen_code_version" in system_message:
                metadata["cli_version"] = system_message.get("qwen_code_version")

        # Extract content from result message (preferred)
        if result_message:
            metadata["is_error"] = result_message.get("is_error", False)
            metadata["num_turns"] = result_message.get("num_turns")
            metadata["duration_ms"] = result_message.get("duration_ms")
            metadata["duration_api_ms"] = result_message.get("duration_api_ms")

            usage = result_message.get("usage")
            if isinstance(usage, dict):
                metadata["usage"] = usage

            model_usage = result_message.get("modelUsage")
            if isinstance(model_usage, dict) and model_usage:
                metadata["model_usage"] = model_usage
                first_model = next(iter(model_usage.keys()))
                metadata["model_used"] = first_model

            permission_denials = result_message.get("permission_denials")
            if isinstance(permission_denials, list) and permission_denials:
                metadata["permission_denials"] = permission_denials

            # Check for error result
            if result_message.get("is_error"):
                error_info = result_message.get("error")
                if isinstance(error_info, dict):
                    error_msg = error_info.get("message", "Unknown error")
                    errors.append(error_msg)
                    metadata["error"] = error_info

            # Get result text
            result_text = result_message.get("result")
            if isinstance(result_text, str) and result_text.strip():
                if stderr and stderr.strip():
                    metadata["stderr"] = stderr.strip()
                return ParsedCLIResponse(content=result_text.strip(), metadata=metadata)

        # Fall back to extracting text from last assistant message
        if last_assistant_message:
            content = self._extract_assistant_text(last_assistant_message)
            if content:
                if stderr and stderr.strip():
                    metadata["stderr"] = stderr.strip()
                return ParsedCLIResponse(content=content, metadata=metadata)

        # If we have errors, return them
        if errors:
            if stderr and stderr.strip():
                metadata["stderr"] = stderr.strip()
            return ParsedCLIResponse(content="\n".join(errors), metadata=metadata)

        # Last resort: check stderr
        if stderr and stderr.strip():
            metadata["stderr"] = stderr.strip()
            return ParsedCLIResponse(
                content="Qwen CLI returned no textual result. Raw stderr was preserved for troubleshooting.",
                metadata=metadata,
            )

        raise ParserError("Qwen CLI stream-json output did not contain a result or assistant message")

    def _extract_assistant_text(self, assistant_msg: dict[str, Any]) -> str | None:
        """Extract text content from an assistant message."""
        message = assistant_msg.get("message")
        if not isinstance(message, dict):
            return None

        content_blocks = message.get("content")
        if not isinstance(content_blocks, list):
            return None

        text_parts: list[str] = []
        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    text_parts.append(text.strip())
            elif block_type == "thinking":
                # Optionally include thinking content
                thinking = block.get("thinking")
                if isinstance(thinking, str) and thinking.strip():
                    text_parts.append(f"[Thinking: {thinking.strip()}]")

        return "\n\n".join(text_parts) if text_parts else None
