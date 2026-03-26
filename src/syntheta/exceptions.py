"""Syntheta exception hierarchy."""


class SynthetaError(Exception):
    """Base exception for all Syntheta errors."""


# ── Config errors ──


class ConfigError(SynthetaError):
    """Base for configuration-related errors."""


class ConfigValidationError(ConfigError):
    """Config file has invalid values or structure."""


class ConfigVersionError(ConfigError):
    """Config version mismatch."""


# ── LLM errors ──


class LLMError(SynthetaError):
    """Base for LLM backend errors."""


class AuthenticationError(LLMError):
    """Invalid API key or unauthorized access (401/403)."""


class ModelNotFoundError(LLMError):
    """Requested model does not exist (404)."""


class PersistentRateLimitError(LLMError):
    """Rate limits have persisted beyond max_backoff."""


class MalformedResponseError(LLMError):
    """LLM returned unparseable or unexpected response."""


# ── Pipeline errors ──


class PipelineError(SynthetaError):
    """Base for pipeline execution errors."""


class CheckpointError(PipelineError):
    """Error saving or loading checkpoint."""


class ConfigHashMismatchError(PipelineError):
    """Config changed between runs when resuming."""


class DiskFullError(PipelineError):
    """Disk full during pipeline write."""


# ── Export errors ──


class ExportValidationError(SynthetaError):
    """Export failed because required fields are missing."""


# ── Prompt errors ──


class PromptError(SynthetaError):
    """Base for prompt template errors."""


class TemplateMissingVariableError(PromptError):
    """A prompt template placeholder has no value provided."""
