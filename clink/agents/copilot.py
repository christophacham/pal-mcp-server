"""GitHub Copilot CLI agent hooks."""

from __future__ import annotations

from clink.parsers.base import ParsedCLIResponse

from .base import AgentOutput, BaseCLIAgent


class CopilotAgent(BaseCLIAgent):
    """Copilot-specific behaviour.

    Copilot CLI outputs plain text (with ``--silent``).  On non-zero exit the
    agent tries to salvage whatever text appeared on stdout/stderr.
    """

    def _recover_from_error(
        self,
        *,
        returncode: int,
        stdout: str,
        stderr: str,
        sanitized_command: list[str],
        duration_seconds: float,
        output_file_content: str | None,
    ) -> AgentOutput | None:
        combined = "\n".join(part for part in (stdout, stderr) if part).strip()
        if not combined:
            return None

        metadata = {
            "cli_error_recovered": True,
            "cli_returncode": returncode,
        }

        parsed = ParsedCLIResponse(content=combined, metadata=metadata)
        return AgentOutput(
            parsed=parsed,
            sanitized_command=sanitized_command,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration_seconds,
            parser_name=self._parser.name,
            output_file_content=output_file_content,
        )
