from __future__ import annotations
import httpx
from .config import settings
from .prompts import SYSTEM_PROMPT

class LLM:
    async def complete(self, user_prompt: str) -> str:
        raise NotImplementedError

class OpenAILLM(LLM):
    async def complete(self, user_prompt: str) -> str:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is missing.")
        headers = {"Authorization": f"Bearer {settings.openai_api_key}"}
        payload = {
            "model": settings.openai_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        }
        async with httpx.AsyncClient(timeout=60) as client:
            try:
                r = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                r.raise_for_status()
                data = r.json()
                return data["choices"][0]["message"]["content"]
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    raise RuntimeError(
                        "OpenAI rate limit exceeded. Please try again later or switch to Ollama by setting LLM_PROVIDER=ollama in your .env file."
                    ) from e
                elif e.response.status_code == 401:
                    raise RuntimeError(
                        "Invalid OpenAI API key. Please check your OPENAI_API_KEY in the .env file."
                    ) from e
                else:
                    raise RuntimeError(f"OpenAI API error: {e.response.status_code} - {e.response.text}") from e

class OllamaLLM(LLM):
    async def complete(self, user_prompt: str) -> str:
        payload = {
            "model": settings.ollama_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.2},
        }
        async with httpx.AsyncClient(timeout=120) as client:
            try:
                r = await client.post(f"{settings.ollama_base_url}/api/chat", json=payload)
                r.raise_for_status()
                data = r.json()
                return data["message"]["content"]
            except httpx.ConnectError:
                raise RuntimeError(
                    f"Cannot connect to Ollama at {settings.ollama_base_url}. Make sure Ollama is running with 'ollama serve' and the model '{settings.ollama_model}' is pulled."
                )
            except httpx.HTTPStatusError as e:
                raise RuntimeError(f"Ollama API error: {e.response.status_code} - {e.response.text}") from e

def get_llm() -> LLM:
    if settings.llm_provider.lower() == "ollama":
        return OllamaLLM()
    return OpenAILLM()
