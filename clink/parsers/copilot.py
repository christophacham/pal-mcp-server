"""Parser for GitHub Copilot CLI plaintext output."""

from __future__ import annotations

from .base import BaseParser, ParsedCLIResponse, ParserError


class CopilotPlaintextParser(BaseParser):
    """Parse stdout produced by `copilot --silent`."""

    name = "copilot_plaintext"

    def parse(self, stdout: str, stderr: str) -> ParsedCLIResponse:
        content = stdout.strip()
        if not content:
            stderr_text = stderr.strip()
            if stderr_text:
                return ParsedCLIResponse(
                    content="Copilot CLI returned no textual result. Raw stderr was preserved for troubleshooting.",
                    metadata={"stderr": stderr_text},
                )
            raise ParserError("Copilot CLI returned empty output")

        metadata: dict = {}
        stderr_text = stderr.strip()
        if stderr_text:
            metadata["stderr"] = stderr_text

        return ParsedCLIResponse(content=content, metadata=metadata)
