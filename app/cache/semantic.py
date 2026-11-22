import asyncio
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    Range,
    VectorParams,
)


INDEXES = [
    ("team_id", PayloadSchemaType.INTEGER),
    ("model_requested", PayloadSchemaType.KEYWORD),
    ("context_hash", PayloadSchemaType.KEYWORD),
    ("params_hash", PayloadSchemaType.KEYWORD),
    ("num_sig", PayloadSchemaType.KEYWORD),
    ("created_at", PayloadSchemaType.INTEGER),
]


def collection_name(embed_model: str) -> str:
    model_slug = re.sub(r"[^a-z0-9]+", "_", embed_model.lower()).strip("_")
    return "semantic_cache_" + (model_slug or "model")


@dataclass
class Hit:
    score: float
    payload: dict[str, Any]


class SemanticCache:
    def __init__(
        self,
        client: AsyncQdrantClient,
        collection: str,
        dim: int,
        threshold: float,
        ttl_s: int,
        embed_model: str,
    ) -> None:
        self.client = client
        self.collection = collection
        self.dim = dim
        self.threshold = threshold
        self.ttl_s = ttl_s
        self.embed_model = embed_model
        self._ready = False
        self._ensure_lock = asyncio.Lock()

    async def ensure_collection(self) -> None:
        if self._ready:
            return
        async with self._ensure_lock:
            if self._ready:
                return

            if not await self.client.collection_exists(self.collection):
                try:
                    await self.client.create_collection(
                        collection_name=self.collection,
                        vectors_config=VectorParams(
                            size=self.dim,
                            distance=Distance.COSINE,
                        ),
                    )
                except UnexpectedResponse:
                    if not await self.client.collection_exists(self.collection):
                        raise

            info = await self.client.get_collection(self.collection)
            vector_config = info.config.params.vectors
            actual_dim = getattr(vector_config, "size", None)
            if actual_dim != self.dim:
                raise RuntimeError(
                    f"collection {self.collection} has dim {actual_dim}, "
                    f"embedder has {self.dim}"
                )

            for field, schema in INDEXES:
                await self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema=schema,
                )
            self._ready = True

    @staticmethod
    def _scope(
        *,
        team_id: int,
        model_requested: str,
        context_hash: str,
        params_hash: str,
        num_sig: str,
    ) -> list[FieldCondition]:
        return [
            FieldCondition(key="team_id", match=MatchValue(value=team_id)),
            FieldCondition(
                key="model_requested",
                match=MatchValue(value=model_requested),
            ),
            FieldCondition(
                key="context_hash",
                match=MatchValue(value=context_hash),
            ),
            FieldCondition(key="params_hash", match=MatchValue(value=params_hash)),
            FieldCondition(key="num_sig", match=MatchValue(value=num_sig)),
        ]

    async def lookup(self, vector: list[float], **scope: Any) -> Hit | None:
        await self.ensure_collection()
        # Whole-second timestamps need an exclusive TTL boundary.
        cutoff = int(time.time()) - self.ttl_s + 1
        result = await self.client.query_points(
            collection_name=self.collection,
            query=vector,
            query_filter=Filter(
                must=self._scope(**scope)
                + [
                    FieldCondition(
                        key="created_at",
                        range=Range(gte=cutoff),
                    )
                ]
            ),
            limit=1,
            score_threshold=self.threshold,
            with_payload=True,
        )
        if not result.points:
            return None

        point = result.points[0]
        if point.score is None or point.score < self.threshold:
            return None
        return Hit(score=float(point.score), payload=point.payload or {})

    async def store(
        self,
        vector: list[float],
        *,
        text_hash: str,
        entry: dict[str, Any],
        **scope: Any,
    ) -> None:
        await self.ensure_collection()
        point_id = uuid.uuid5(
            uuid.NAMESPACE_URL,
            "|".join(
                [
                    str(scope["team_id"]),
                    scope["model_requested"],
                    scope["context_hash"],
                    scope["params_hash"],
                    scope["num_sig"],
                    text_hash,
                ]
            ),
        )
        await self.client.upsert(
            collection_name=self.collection,
            points=[
                PointStruct(
                    id=str(point_id),
                    vector=vector,
                    payload={
                        **scope,
                        "embed_model": self.embed_model,
                        **entry,
                    },
                )
            ],
        )

    async def purge_expired(self) -> None:
        await self.ensure_collection()
        cutoff = int(time.time()) - self.ttl_s + 1
        await self.client.delete(
            collection_name=self.collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="created_at",
                            range=Range(lt=cutoff),
                        )
                    ]
                )
            ),
        )

    async def count_team(self, team_id: int) -> int:
        await self.ensure_collection()
        result = await self.client.count(
            collection_name=self.collection,
            exact=True,
            count_filter=Filter(
                must=[
                    FieldCondition(
                        key="team_id",
                        match=MatchValue(value=team_id),
                    )
                ]
            ),
        )
        return result.count

    async def purge_team(self, team_id: int) -> None:
        await self.ensure_collection()
        await self.client.delete(
            collection_name=self.collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="team_id",
                            match=MatchValue(value=team_id),
                        )
                    ]
                )
            ),
        )

    async def purge_all(self) -> None:
        await self.client.delete_collection(self.collection)
        self._ready = False
        await self.ensure_collection()
