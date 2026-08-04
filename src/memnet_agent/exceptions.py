class MemnetAgentError(Exception):
    """Base exception for memnet-agent."""


class ModelAdapterError(MemnetAgentError):
    """Raised when a model cannot be invoked or its response cannot be converted to text."""


class AgentSleepingError(MemnetAgentError):
    """Raised when inference is disabled by a scheduled sleep window."""


class GraphFormatError(MemnetAgentError):
    """Raised when a graph format is unsupported or invalid."""
