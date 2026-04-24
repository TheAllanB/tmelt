from unittest.mock import MagicMock, patch
from src.environment import TicketmeltEnv
from src.prompt import observation_to_prompt
from src.rollout import run_episode, parse_action_from_model


def _make_mock_model(response='{"commitment": "DEPLOY_PROD_B", "channel_msg": "B"}'):
    """Return (model, tokenizer) mocks that always produce `response`."""
    tokenizer = MagicMock()
    tokenizer.return_value = {"input_ids": MagicMock(shape=[1, 10])}
    tokenizer.decode.return_value = response
    tokenizer.apply_chat_template.return_value = "formatted_prompt"
    tokenizer.eos_token_id = 2

    fake_inputs = {"input_ids": MagicMock()}
    fake_inputs["input_ids"].shape = [1, 10]
    tokenizer.side_effect = None
    tokenizer.__call__ = MagicMock(return_value=fake_inputs)

    model = MagicMock()
    fake_output = MagicMock()
    fake_output.__getitem__ = lambda self, i: MagicMock()
    model.generate.return_value = [fake_output]

    return model, tokenizer


def test_run_episode_returns_required_keys():
    model, tokenizer = _make_mock_model()
    env = TicketmeltEnv(seed=42)
    result = run_episode(model, tokenizer, env, observation_to_prompt, seed=42, device="cpu")
    assert "history" in result
    assert "final_reward" in result
    assert "info" in result


def test_run_episode_history_length_equals_total_rounds():
    model, tokenizer = _make_mock_model()
    env = TicketmeltEnv(seed=42, total_rounds=3)
    result = run_episode(model, tokenizer, env, observation_to_prompt, seed=42, device="cpu")
    assert len(result["history"]) == 3


def test_run_episode_final_reward_is_binary():
    model, tokenizer = _make_mock_model()
    env = TicketmeltEnv(seed=42)
    result = run_episode(model, tokenizer, env, observation_to_prompt, seed=42, device="cpu")
    assert result["final_reward"] in (0.0, 1.0)


def test_run_episode_history_entries_have_correct_keys():
    model, tokenizer = _make_mock_model()
    env = TicketmeltEnv(seed=1, total_rounds=2)
    result = run_episode(model, tokenizer, env, observation_to_prompt, seed=1, device="cpu")
    entry = result["history"][0]
    assert "prompt" in entry
    assert "response" in entry
    assert "action" in entry
    assert "reward" in entry


def test_parse_action_from_model_valid_json():
    action = parse_action_from_model('{"commitment": "DEPLOY_PROD_A", "channel_msg": "go"}')
    assert action.commitment == "DEPLOY_PROD_A"
    assert action.channel_msg == "go"


def test_parse_action_from_model_invalid_defaults_to_monitor():
    action = parse_action_from_model("completely invalid text with nothing useful")
    assert action.commitment == "MONITOR"
