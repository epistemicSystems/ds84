"""LLM service for interacting with OpenAI and Anthropic APIs"""
import httpx
import json
from typing import Dict, Any, Optional, List
from app.config import settings


class LLMService:
    """Service for LLM API interactions"""

    def __init__(self):
        self.openai_api_key = settings.llm.openai_api_key
        self.anthropic_api_key = settings.llm.anthropic_api_key
        self.timeout = settings.llm.timeout

    async def complete(
        self,
        prompt: str,
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        provider: str = "openai"
    ) -> str:
        """Complete a prompt using specified LLM provider"""
        if provider == "openai":
            return await self._complete_openai(prompt, model, temperature, max_tokens)
        elif provider == "anthropic":
            return await self._complete_anthropic(prompt, model, temperature, max_tokens)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    async def _complete_openai(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Complete using OpenAI API"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
            )

            if response.status_code != 200:
                raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")

            result = response.json()
            return result["choices"][0]["message"]["content"]

    async def _complete_anthropic(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Complete using Anthropic API"""
        if not self.anthropic_api_key:
            raise ValueError("Anthropic API key not configured")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.anthropic_api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
            )

            if response.status_code != 200:
                raise Exception(f"Anthropic API error: {response.status_code} - {response.text}")

            result = response.json()
            return result["content"][0]["text"]

    async def generate_embedding(
        self,
        text: str,
        model: str = "text-embedding-ada-002"
    ) -> List[float]:
        """Generate embeddings using OpenAI"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "input": text,
                    "model": model
                }
            )

            if response.status_code != 200:
                raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")

            result = response.json()
            return result["data"][0]["embedding"]

    async def generate_embeddings_batch(
        self,
        texts: List[str],
        model: str = "text-embedding-ada-002"
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts in batch"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "input": texts,
                    "model": model
                }
            )

            if response.status_code != 200:
                raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")

            result = response.json()
            return [item["embedding"] for item in result["data"]]


# Global LLM service instance
llm_service = LLMService()
