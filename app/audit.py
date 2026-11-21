import json
from typing import Any, Mapping


async def audit(
    db: Any,
    actor: str,
    event: str,
    details: Mapping[str, Any],
) -> None:
    await db.execute(
        """
        INSERT INTO audit_log (actor, event, details)
        VALUES ($1, $2, $3::jsonb)
        """,
        actor,
        event,
        json.dumps(details),
    )
