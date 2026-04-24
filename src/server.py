from __future__ import annotations
from dataclasses import asdict
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .environment import TicketmeltEnv
from .models import Action, Observation

app = FastAPI(title="TicketMelt", version="0.1.0")
_env = TicketmeltEnv()


class ResetRequest(BaseModel):
    seed: Optional[int] = None


class StepRequest(BaseModel):
    commitment: str
    channel_msg: str = ""


def _obs_to_dict(obs: Observation) -> dict:
    return {
        "current_round": obs.current_round,
        "total_rounds": obs.total_rounds,
        "my_service": asdict(obs.my_service),
        "my_engineer_name": obs.my_engineer_name,
        "peer_progress": obs.peer_progress,
        "history": [asdict(r) for r in obs.history],
        "done": obs.done,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset(req: ResetRequest = None):
    seed = req.seed if req is not None else None
    obs = _env.reset(seed=seed)
    return _obs_to_dict(obs)


@app.post("/step")
def step(req: StepRequest):
    try:
        action = Action(commitment=req.commitment, channel_msg=req.channel_msg)
        obs, reward, done, info = _env.step(action)
        return {"observation": _obs_to_dict(obs), "reward": reward, "done": done, "info": info}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state():
    return _env.state()
