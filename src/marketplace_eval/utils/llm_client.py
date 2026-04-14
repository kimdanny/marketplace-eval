from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

try:
    from openai import AsyncOpenAI
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    AsyncOpenAI = None  # type: ignore


from dotenv import load_dotenv

load_dotenv()


class LLMClientError(RuntimeError):
    """Raised when an LLM client cannot complete a request."""


class BaseLLMClient:
    """Abstract base class for LLM clients."""

    async def generate(
        self, prompt: str, *, system_prompt: str | None = None, **kwargs
    ) -> str:  # noqa: D401 - interface only
        raise NotImplementedError


@dataclass
class VLLMConfig:
    model_id: str
    sampling_parameters: Mapping[str, Any] | None = None
    llm_kwargs: Mapping[str, Any] | None = None


class HuggingFaceVLLMClient(BaseLLMClient):
    """Client that runs a local vLLM engine backed by Hugging Face weights."""

    def __init__(self, config: VLLMConfig):
        # lazy import of vllm since we may not use it.
        try:
            from vllm import LLM, SamplingParams
        except ModuleNotFoundError:
            raise LLMClientError(
                "vLLM is not installed. Install the 'vllm' package to use HuggingFace nodes."
            )
        sampling_kwargs = dict(config.sampling_parameters or {})
        self.sampling_params = SamplingParams(**sampling_kwargs)
        llm_kwargs = dict(config.llm_kwargs or {})
        self._engine = LLM(model=config.model_id, **llm_kwargs)

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        **_: Any,
    ) -> str:
        # TODO: check if the system prompt is passed in this way in VLLM
        final_prompt = (
            prompt if system_prompt is None else f"{system_prompt}\n\n{prompt}"
        )

        def _run_generation() -> str:
            outputs = self._engine.generate(
                [final_prompt],
                sampling_params=self.sampling_params,
            )
            if not outputs:
                raise LLMClientError("vLLM returned no generations")
            return outputs[0].outputs[0].text  # type: ignore[index]

        return await asyncio.to_thread(_run_generation)


class OpenAICompatibleClient(BaseLLMClient):
    """Client for OpenAI-compatible HTTP chat completion endpoints."""

    def __init__(
        self,
        *,
        model_id: str,
        api_key: str,
        base_url: str,
        timeout: float | None = 60.0,
        default_headers: Optional[Mapping[str, str]] = None,
        default_params: Optional[Mapping[str, Any]] = None,
    ):
        if AsyncOpenAI is None:
            raise LLMClientError(
                "openai is not installed. Install openai to use OpenRouter/OpenAI nodes."
            )
        self.model_id = model_id
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.default_headers = dict(default_headers or {})
        self.default_params = dict(default_params or {})

        # Initialize AsyncOpenAI client
        self._client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            default_headers=self.default_headers,
        )

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        messages: Optional[list[dict[str, str]]] = None,
        max_retries: int = 3,
        fallback_on_failure: bool = True,
        **kwargs: Any,
    ) -> str:
        # Prepare parameters
        params = dict(self.default_params)
        params.update({k: v for k, v in kwargs.items() if v is not None})

        # Prepare messages
        if messages is None:
            chat_messages: list[dict[str, str]] = []
            if system_prompt:
                chat_messages.append({"role": "system", "content": system_prompt})
            chat_messages.append({"role": "user", "content": prompt})
        else:
            chat_messages = messages

        last_exception = None
        base_delay = 1.0  # Start with 1 second delay

        for attempt in range(max_retries):
            try:
                # Use AsyncOpenAI to make the request
                response = await self._client.chat.completions.create(
                    model=self.model_id,
                    messages=chat_messages,
                    **{
                        k: v
                        for k, v in params.items()
                        if k not in {"messages", "system_prompt"}
                    },
                )

                # Extract content from response
                if not response.choices:
                    raise LLMClientError("LLM response did not contain any choices")

                message = response.choices[0].message
                if not message or not message.content:
                    # Empty content - retry if we have attempts left
                    if attempt < max_retries - 1:
                        delay = base_delay * (2**attempt)  # Exponential backoff
                        print(
                            f"[WARNING] Empty response from {self.model_id}, retrying in {delay}s (attempt {attempt + 1}/{max_retries})"
                        )
                        await asyncio.sleep(delay)
                        continue
                    # Last attempt with empty response
                    if fallback_on_failure:
                        print(
                            f"[ERROR] All retries exhausted for {self.model_id}, returning fallback response"
                        )
                        return "I cannot answer this question"
                    raise LLMClientError(
                        "LLM response message content missing or invalid"
                    )

                return message.content

            except LLMClientError as exc:
                last_exception = exc
                # Retry on empty content errors
                if (
                    "content missing or invalid" in str(exc)
                    and attempt < max_retries - 1
                ):
                    delay = base_delay * (2**attempt)
                    print(
                        f"[WARNING] LLMClientError for {self.model_id}: {exc}, retrying in {delay}s (attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(delay)
                    continue
                # Last attempt failed
                if fallback_on_failure:
                    print(
                        f"[ERROR] All retries exhausted for {self.model_id} with error: {exc}, returning fallback response"
                    )
                    return "I cannot answer this question"
                raise

            except Exception as exc:
                last_exception = exc
                # Retry on network/timeout errors
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    print(
                        f"[WARNING] Exception for {self.model_id}: {exc}, retrying in {delay}s (attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(delay)
                    continue
                # Last attempt failed
                if fallback_on_failure:
                    print(
                        f"[ERROR] All retries exhausted for {self.model_id} with exception: {exc}, returning fallback response"
                    )
                    return "I cannot answer this question"
                raise LLMClientError(
                    f"LLM request failed after {max_retries} attempts: {str(exc)}"
                ) from exc

        # Should not reach here, but just in case
        if fallback_on_failure:
            error_msg = str(last_exception) if last_exception else "Unknown error"
            print(
                f"[ERROR] Unexpected fallback for {self.model_id}: {error_msg}, returning fallback response"
            )
            return "I cannot answer this question"

        if last_exception:
            raise LLMClientError(
                f"LLM request failed after {max_retries} attempts: {str(last_exception)}"
            ) from last_exception
        raise LLMClientError(f"LLM request failed after {max_retries} attempts")


def _resolve_api_key(env_key_name: str) -> str:
    api_key = os.getenv(env_key_name)
    if not api_key:
        raise LLMClientError(f"{env_key_name} not provided in environment variable ")
    return api_key


def create_llm_client(config: Mapping[str, Any] | None) -> BaseLLMClient | None:
    """Create an LLM client from configuration data."""

    if not config:
        return None

    provider = str(config.get("provider") or config.get("type") or "").lower()
    model_id = config.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise LLMClientError("'model_id' is required for generator models")

    if provider in {"", "huggingface", "vllm"}:
        sampling_params = config.get("sampling_parameters") or config.get(
            "sampling_params"
        )
        llm_kwargs = config.get("llm_kwargs") or {}
        vllm_config = VLLMConfig(
            model_id=model_id,
            sampling_parameters=sampling_params,
            llm_kwargs=llm_kwargs,
        )
        return HuggingFaceVLLMClient(vllm_config)

    if provider in {"openrouter", "openai"}:
        env_key_name = (
            "OPENROUTER_API_KEY" if provider == "openrouter" else "OPENAI_API_KEY"
        )
        api_key = _resolve_api_key(env_key_name)
        base_url = config.get("base_url")
        if not base_url:
            base_url = (
                "https://openrouter.ai/api/v1"
                if provider == "openrouter"
                else "https://api.openai.com/v1"
            )
        default_headers = config.get("headers") or {}
        default_params = (
            config.get("generation_parameters") or config.get("params") or {}
        )
        timeout = config.get("timeout", 60.0)
        return OpenAICompatibleClient(
            model_id=model_id,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            default_headers=default_headers,
            default_params=default_params,
        )

    raise LLMClientError(f"Unsupported LLM provider '{provider}'")
