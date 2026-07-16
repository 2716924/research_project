"""Save and resume RCT result files."""

import json
import os

from rct.constants import EVAL_SEEDS, VARIANT_LABELS


def result_path(output_dir, variant, game_id):
    safe_game = game_id.replace("/", "_")
    return os.path.join(output_dir, f"{variant}_results_{safe_game}.json")


def load_existing_results(out_path):
    if os.path.exists(out_path):
        with open(out_path) as f:
            data = json.load(f)
        results = data.get("results", [])
        completed_ids = {r["individual_id"] for r in results}
        return results, completed_ids
    return [], set()


def save_results(out_path, metadata, game_results):
    tmp_path = out_path + ".tmp"
    payload = dict(metadata)
    payload["results"] = game_results
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, out_path)


def build_metadata(variant, game_id, total_frames, eval_frequency, num_eval_episodes):
    return {
        "variant": variant,
        "variant_label": VARIANT_LABELS[variant],
        "game_id": game_id,
        "total_frames": total_frames,
        "eval_frequency": eval_frequency,
        "num_eval_episodes": num_eval_episodes,
        "eval_seeds": EVAL_SEEDS[:num_eval_episodes],
    }
