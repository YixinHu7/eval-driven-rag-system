from openai import OpenAI

from src.core.config import settings


class LLMClient:
    def __init__(self) -> None:
        if settings.llm.provider != "openai":
            raise ValueError(f"Unsupported LLM provider: {settings.llm.provider}")

        if not settings.llm.api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Add it to your .env file before using LLM generation."
            )

        self.client = OpenAI(api_key=settings.llm.api_key)
        self.model_name = settings.llm.model_name

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful technical documentation assistant. "
                        "Answer only using the provided context. "
                        "If the context does not support the answer, say so."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
        )

        content = response.choices[0].message.content
        return content.strip() if content else ""