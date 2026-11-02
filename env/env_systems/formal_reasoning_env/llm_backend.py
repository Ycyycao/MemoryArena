"""
LLM Backend Support for Multiple Providers
Adapted from /home/zexueh/orcd/pool/memact/evaluation/core/llm_backend.py

Every provider goes through litellm. There used to be one class per vendor, each
with its own SDK and its own message-format conversion; litellm speaks all of
them and takes OpenAI-shaped messages, so a single backend covers the lot and
`backend_name` only decides which model prefix and which credentials to use.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import os

import litellm


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""
    
    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 2000,
        **kwargs
    ) -> str:
        """Generate a chat completion."""
        raise NotImplementedError("Subclasses should implement this method")


class LiteLLMBackend(LLMBackend):
    """One backend for every provider; litellm routes by the model name."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 2000,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.api_base = base_url

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        response = litellm.completion(
            model=self.model_name,
            api_key=self.api_key,
            api_base=self.api_base,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            **kwargs
        )
        # Providers return None rather than "" when a response is cut short
        # (finish_reason="length") or filtered, and .strip() on None throws.
        return (response.choices[0].message.content or "").strip()


# What each backend_name means: the model prefix litellm needs (when the caller
# has not written one already) and where the credentials come from.
_PROVIDERS = {
    "openai":     {"prefix": "",            "key": ("OPENAI_API_KEY",),
                   "base": ("OPENAI_BASE_URL", "OPENAI_API_BASE")},
    "anthropic":  {"prefix": "anthropic/",  "key": ("ANTHROPIC_API_KEY",), "base": ()},
    "openrouter": {"prefix": "openrouter/", "key": ("OPENROUTER_API_KEY",),
                   "base": ("OPENROUTER_API_BASE_URL",)},
    "gemini":     {"prefix": "gemini/",     "key": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
                   "base": ("GEMINI_BASE_URL",)},
    "google":     {"prefix": "gemini/",     "key": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
                   "base": ("GEMINI_BASE_URL",)},
}


def _first_env(names) -> Optional[str]:
    """First of `names` that is set, with any trailing slash removed.

    litellm joins the base with its own path, so a base ending in "/" produces
    a double slash ("/v1//models/...") that some gateways reject outright.
    """
    for name in names:
        value = os.getenv(name)
        if value:
            return value.rstrip("/")
    return None


def create_backend(
    backend_name: str,
    model_name: str,
    temperature: float = 0.0,
    max_tokens: int = 2000,
    **kwargs
) -> LLMBackend:
    """
    Factory function to create LLM backend instances.

    Args:
        backend_name: 'openai', 'anthropic', 'openrouter', 'gemini' (or 'google')
        model_name: Model name to use. A name that already carries a litellm
            provider prefix ("gemini/gemini-2.5-flash") is passed through as-is,
            so backend_name only matters for names without one.
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        **kwargs: Passed to the backend (api_key, base_url)

    Returns:
        LLMBackend instance
    """
    provider = _PROVIDERS.get(backend_name.lower())
    if provider is None:
        raise ValueError(
            f"Unknown backend: {backend_name}. "
            f"Supported: openai, anthropic, openrouter, gemini"
        )

    if "/" not in model_name:
        model_name = provider["prefix"] + model_name

    return LiteLLMBackend(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=kwargs.pop("api_key", None) or _first_env(provider["key"]),
        base_url=kwargs.pop("base_url", None) or _first_env(provider["base"]),
    )
