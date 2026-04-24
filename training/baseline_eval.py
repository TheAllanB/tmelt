#!/usr/bin/env python3
"""
Baseline evaluation for TICKETMELT.

Runs N episodes with a (pre-trained) model and writes component rewards to JSON.
Run BEFORE training so you have a comparison baseline for the plots.

Usage:
    python training/baseline_eval.py --model Qwen/Qwen2.5-3B-Instruct --n_episodes 50 --output baseline_results.json
    python training/baseline_eval.py --model ./ticketmelt_final --output trained_results.json
"""
import argparse
import json
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--n_episodes", type=int, default=50)
    p.add_argument("--output", default="baseline_results.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed_offset", type=int, default=0,
                   help="First seed used. Avoids overlap with training seeds.")
    return p.parse_args()


def load_model(model_name, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=device
    )
    model.eval()
    return model, tokenizer


def main():
    args = parse_args()
    model, tokenizer = load_model(args.model, args.device)

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.environment import TicketmeltEnv
    from src.prompt import observation_to_prompt
    from src.rollout import run_episode

    results = []
    wins = 0

    for i in range(args.n_episodes):
        seed = args.seed_offset + i
        env = TicketmeltEnv(seed=seed)
        result = run_episode(
            model, tokenizer, env, observation_to_prompt, seed=seed, device=args.device
        )
        final_reward = result["final_reward"]
        bd = result["info"].get("reward_breakdown", {})
        summary = result["info"].get("episode_summary", {})

        results.append({
            "episode": i,
            "seed": seed,
            "final_reward": final_reward,
            "r1": bd.get("r1_service_restored", 0.0),
            "r2": bd.get("r2_site_uptime", 0.0),
            "r3": bd.get("r3_clean_deploys", 0.0),
            "r4": bd.get("r4_yield_to_critical", 0.0),
            "weighted_sum": bd.get("weighted_sum", 0.0),
            "services_restored": summary.get("services_restored", 0),
            "total_collisions": summary.get("total_collisions", 0),
        })
        wins += int(final_reward == 1.0)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{args.n_episodes} — win rate: {wins/(i+1):.1%}")

    n = args.n_episodes
    avg = lambda key: sum(r[key] for r in results) / n
    output = {
        "model": args.model,
        "n_episodes": n,
        "win_rate": wins / n,
        "avg_r1": avg("r1"),
        "avg_r2": avg("r2"),
        "avg_r3": avg("r3"),
        "avg_r4": avg("r4"),
        "episodes": results,
    }

    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"\nSaved → {args.output}")
    print(
        f"win_rate={output['win_rate']:.1%}  "
        f"R1={output['avg_r1']:.3f}  R2={output['avg_r2']:.3f}  "
        f"R3={output['avg_r3']:.3f}  R4={output['avg_r4']:.3f}"
    )


if __name__ == "__main__":
    main()
