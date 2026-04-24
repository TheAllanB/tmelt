from __future__ import annotations
from contextlib import nullcontext
from typing import Callable, Optional

from .environment import TicketmeltEnv
from .models import Action, Observation
from .prompt import observation_to_prompt, parse_action


def parse_action_from_model(text: str) -> Action:
    """Thin alias so tests can import this name explicitly."""
    return parse_action(text)


def run_episode(
    model,
    tokenizer,
    env: TicketmeltEnv,
    prompt_fn: Callable[[Observation], str] = observation_to_prompt,
    seed: Optional[int] = None,
    device: str = "cuda",
    max_new_tokens: int = 64,
    temperature: float = 0.7,
) -> dict:
    """
    Run one full episode with the given model.

    Returns:
        {
            "history": [{"prompt": str, "response": str, "action": dict, "reward": float}, ...],
            "final_reward": float,
            "info": dict,
        }
    """
    try:
        import torch
        no_grad = torch.no_grad
    except ImportError:
        no_grad = nullcontext

    obs = env.reset(seed=seed)
    episode_history = []
    reward = 0.0
    info = {}

    while not obs.done:
        prompt = prompt_fn(obs)
        messages = [{"role": "user", "content": prompt}]

        try:
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(input_text, return_tensors="pt")
        except Exception:
            inputs = tokenizer(prompt, return_tensors="pt")

        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        with no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
            )

        response_ids = output_ids[0][input_len:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
        action = parse_action(response_text)

        obs, reward, done, info = env.step(action)
        episode_history.append({
            "prompt": prompt,
            "response": response_text,
            "action": action.to_dict(),
            "reward": reward,
            "done": done,
        })

    return {"history": episode_history, "final_reward": reward, "info": info}
