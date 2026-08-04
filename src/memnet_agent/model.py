from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

from .exceptions import ModelAdapterError


@runtime_checkable
class TextModel(Protocol):
    """Minimal model contract used by :class:`MemoryAgent`."""

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Return a text response for a prompt."""


class BaseModelAdapter:
    """Adapter interface for synchronous and asynchronous model calls."""

    def generate(self, prompt: str, **kwargs: Any) -> str:
        raise NotImplementedError

    async def agenerate(self, prompt: str, **kwargs: Any) -> str:
        return await asyncio.to_thread(self.generate, prompt, **kwargs)


@dataclass(slots=True)
class CallableModelAdapter(BaseModelAdapter):
    """Wrap a function or callable object that accepts a prompt string."""

    function: Callable[..., Any]

    def generate(self, prompt: str, **kwargs: Any) -> str:
        try:
            result = self.function(prompt, **kwargs)
        except Exception as exc:  # pragma: no cover - exact model errors are external
            raise ModelAdapterError(f"Model call failed: {exc}") from exc
        if inspect.isawaitable(result):
            raise ModelAdapterError(
                "The model returned an awaitable in synchronous mode. Use `await agent.aask(...)`."
            )
        return response_to_text(result)

    async def agenerate(self, prompt: str, **kwargs: Any) -> str:
        try:
            result = self.function(prompt, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            return response_to_text(result)
        except ModelAdapterError:
            raise
        except Exception as exc:  # pragma: no cover - exact model errors are external
            raise ModelAdapterError(f"Async model call failed: {exc}") from exc


@dataclass(slots=True)
class ObjectMethodAdapter(BaseModelAdapter):
    """Wrap an object's model method such as ``invoke`` or ``generate``."""

    model: Any
    method_name: str

    def _method(self) -> Callable[..., Any]:
        method = getattr(self.model, self.method_name, None)
        if not callable(method):
            raise ModelAdapterError(
                f"Model method `{self.method_name}` is unavailable on {type(self.model).__name__}."
            )
        return method

    def generate(self, prompt: str, **kwargs: Any) -> str:
        try:
            result = self._method()(prompt, **kwargs)
        except Exception as exc:  # pragma: no cover
            raise ModelAdapterError(f"Model call failed: {exc}") from exc
        if inspect.isawaitable(result):
            raise ModelAdapterError(
                "The model method returned an awaitable in synchronous mode. "
                "Use `await agent.aask(...)`."
            )
        return response_to_text(result)

    async def agenerate(self, prompt: str, **kwargs: Any) -> str:
        method = self._method()
        try:
            if inspect.iscoroutinefunction(method):
                result = await method(prompt, **kwargs)
            else:
                result = await asyncio.to_thread(method, prompt, **kwargs)
                if inspect.isawaitable(result):
                    result = await result
            return response_to_text(result)
        except ModelAdapterError:
            raise
        except Exception as exc:  # pragma: no cover
            raise ModelAdapterError(f"Async model call failed: {exc}") from exc


def adapt_model(model: Any, *, method: str | None = None) -> BaseModelAdapter:
    """Normalize a user-supplied model into a text model adapter.

    Accepted inputs:

    * an existing :class:`BaseModelAdapter`;
    * a callable ``model(prompt, **kwargs)``;
    * an object exposing one of ``generate``, ``invoke``, ``predict``,
      ``complete`` or ``chat``.

    Use ``method=...`` when the desired method is not first in the automatic
    lookup order.
    """

    if isinstance(model, BaseModelAdapter):
        return model
    if model is None:
        raise ModelAdapterError("A text model or callable must be provided.")
    if method is not None:
        return ObjectMethodAdapter(model=model, method_name=method)

    for candidate in ("generate", "invoke", "predict", "complete", "chat"):
        if callable(getattr(model, candidate, None)):
            return ObjectMethodAdapter(model=model, method_name=candidate)
    if callable(model):
        return CallableModelAdapter(function=model)
    raise ModelAdapterError(
        "Unsupported model. Pass a callable or an object with generate/invoke/"
        "predict/complete/chat, or provide `model_method=` explicitly."
    )


def response_to_text(value: Any) -> str:
    """Convert common model response shapes into plain text."""

    text = _response_to_text(value)
    text = text.strip()
    if not text:
        raise ModelAdapterError("The model returned an empty response.")
    return text


def _response_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if value is None:
        raise ModelAdapterError("The model returned None.")

    if isinstance(value, dict):
        for key in ("output_text", "text", "content", "answer", "output", "response"):
            if key in value:
                return _response_to_text(value[key])
        if "message" in value:
            return _response_to_text(value["message"])
        if "choices" in value:
            return _response_to_text(value["choices"])

    if isinstance(value, (list, tuple)):
        if not value:
            raise ModelAdapterError("The model returned an empty list.")
        # Hugging Face pipelines usually return [{'generated_text': '...'}].
        first = value[0]
        if isinstance(first, dict):
            for key in ("generated_text", "summary_text", "translation_text", "text"):
                if key in first:
                    return _response_to_text(first[key])
        return _response_to_text(first)

    for attr in ("output_text", "content", "text", "answer", "output"):
        if hasattr(value, attr):
            return _response_to_text(getattr(value, attr))

    if hasattr(value, "message"):
        return _response_to_text(getattr(value, "message"))
    if hasattr(value, "choices"):
        return _response_to_text(getattr(value, "choices"))

    raise ModelAdapterError(
        f"Cannot convert model response of type {type(value).__name__} to text. "
        "Wrap the model in a callable that returns a string."
    )
