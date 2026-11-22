import asyncio
from pathlib import Path
from typing import Any


class EmbedderUnavailableError(RuntimeError):
    pass


class FastEmbedEmbedder:
    def __init__(self, model_name: str, cache_dir: str | Path) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise EmbedderUnavailableError(
                "FastEmbed is required for semantic cache embeddings."
            ) from exc

        self.name = model_name
        self._model: Any = TextEmbedding(
            model_name=model_name,
            cache_dir=str(cache_dir),
        )
        self.dim = len(self.embed_sync("warm up"))
        self._sem = asyncio.Semaphore(2)

    def embed_sync(self, text: str) -> list[float]:
        vector = next(iter(self._model.embed([text])))
        if hasattr(vector, "tolist"):
            return [float(value) for value in vector.tolist()]
        return [float(value) for value in vector]

    async def embed(self, text: str) -> list[float]:
        async with self._sem:
            return await asyncio.to_thread(self.embed_sync, text)
