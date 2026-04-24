import importlib
import pytest
from fastapi.testclient import TestClient


def fresh_client():
    """Reload server module to get a clean env state."""
    import src.server as server_module
    importlib.reload(server_module)
    return TestClient(server_module.app)


def test_health():
    client = fresh_client()
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_reset_returns_observation():
    client = fresh_client()
    r = client.post("/reset", json={"seed": 42})
    assert r.status_code == 200
    data = r.json()
    assert data["current_round"] == 0
    assert data["total_rounds"] == 8
    assert data["my_engineer_name"] == "trained"
    assert "my_service" in data
    assert "peer_progress" in data
    assert data["done"] is False


def test_step_returns_correct_shape():
    client = fresh_client()
    client.post("/reset", json={"seed": 0})
    r = client.post("/step", json={"commitment": "DEPLOY_PROD_B", "channel_msg": "going B"})
    assert r.status_code == 200
    data = r.json()
    assert "observation" in data
    assert "reward" in data
    assert "done" in data
    assert "info" in data
    assert data["reward"] == 0.0   # intermediate round — always 0


def test_step_before_reset_returns_400():
    client = fresh_client()
    r = client.post("/step", json={"commitment": "MONITOR", "channel_msg": ""})
    assert r.status_code == 400


def test_state_returns_full_state():
    client = fresh_client()
    client.post("/reset", json={"seed": 1})
    r = client.get("/state")
    assert r.status_code == 200
    data = r.json()
    assert "services" in data
    assert "trained" in data["services"]
    assert "history" in data


def test_full_episode_ends_with_done_true():
    client = fresh_client()
    client.post("/reset", json={"seed": 10})
    done = False
    for _ in range(10):
        r = client.post("/step", json={"commitment": "DEPLOY_PROD_B", "channel_msg": ""})
        data = r.json()
        if data["done"]:
            done = True
            assert data["reward"] in (0.0, 1.0)
            break
    assert done
