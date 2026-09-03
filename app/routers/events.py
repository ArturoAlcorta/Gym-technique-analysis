import json
import uuid

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.db import get_session
from app.models import Analysis
from app.pubsub import subscribe

router = APIRouter()


@router.get("/analyses/{analysis_id}/events")
async def analysis_events(analysis_id: uuid.UUID, session: Session = Depends(get_session)):
    analysis = session.get(Analysis, analysis_id)
    initial_status = analysis.status if analysis else "error"

    async def event_stream():
        # Already finished before the browser subscribed: emit one terminal
        # event and close, instead of hanging on a channel nobody will publish to.
        if initial_status in ("done", "error"):
            yield f"data: {json.dumps({'stage': initial_status, 'status': initial_status})}\n\n"
            return
        async for payload in subscribe(analysis_id):
            yield f"data: {json.dumps(payload)}\n\n"
            if payload.get("status") in ("done", "error"):
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")
