from src.environment import TicketmeltEnv
from src.prompt import observation_to_prompt, parse_action


def _obs(seed=42):
    env = TicketmeltEnv(seed=seed)
    return env.reset()


def test_prompt_contains_round_info():
    obs = _obs()
    prompt = observation_to_prompt(obs)
    assert "1" in prompt        # current round (displayed 1-indexed)
    assert "8" in prompt        # total_rounds default


def test_prompt_contains_service_name():
    obs = _obs()
    prompt = observation_to_prompt(obs)
    assert obs.my_service.name in prompt


def test_prompt_contains_all_three_commitment_options():
    obs = _obs()
    prompt = observation_to_prompt(obs)
    assert "DEPLOY_PROD_A" in prompt
    assert "DEPLOY_PROD_B" in prompt
    assert "MONITOR" in prompt


def test_prompt_contains_deadline():
    obs = _obs()
    prompt = observation_to_prompt(obs)
    assert str(obs.my_service.deadline_round) in prompt


def test_prompt_contains_fix_rounds_remaining():
    obs = _obs()
    prompt = observation_to_prompt(obs)
    assert str(obs.my_service.fix_rounds_remaining) in prompt


def test_parse_action_valid_json():
    text = '{"commitment": "DEPLOY_PROD_B", "channel_msg": "going to B"}'
    action = parse_action(text)
    assert action.commitment == "DEPLOY_PROD_B"
    assert action.channel_msg == "going to B"


def test_parse_action_json_embedded_in_text():
    text = 'Sure! {"commitment": "DEPLOY_PROD_A", "channel_msg": "hi"} done.'
    action = parse_action(text)
    assert action.commitment == "DEPLOY_PROD_A"


def test_parse_action_keyword_fallback():
    text = "I think I should DEPLOY_PROD_B right now because peers crowd A"
    action = parse_action(text)
    assert action.commitment == "DEPLOY_PROD_B"


def test_parse_action_invalid_defaults_to_monitor():
    action = parse_action("total gibberish with no keywords")
    assert action.commitment == "MONITOR"
