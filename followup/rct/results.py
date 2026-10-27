"""Save and resume RCT result files."""

import json
import os
import hashlib
import math
import platform
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

from rct.constants import EVAL_SEEDS, VARIANT_LABELS, TREATMENT_DEFAULTS


def result_path(output_dir, variant, game_id):
    safe_game = game_id.replace("/", "_")
    return os.path.join(output_dir, f"{variant}_results_{safe_game}.json")


def load_existing_results(out_path, expected_metadata=None, expected_individuals=None):
    """Read historical results, or strictly validate a requested training resume.

    Read-only inspection without expected_metadata remains compatible with old
    data. Training must supply metadata and its cohort; legacy/mismatched runs
    cannot be silently relabelled with the current implementation.
    """
    if os.path.exists(out_path):
        with open(out_path) as f:
            data = json.load(f)
        results = data.get("results", [])
        if expected_metadata is not None:
            mismatches = [key for key, value in expected_metadata.items() if data.get(key) != value]
            if mismatches:
                raise ValueError(
                    f"Refusing incompatible resume of {out_path}: {', '.join(mismatches)}. "
                    "Use a new --output-dir; existing results have not been changed."
                )
            _validate_completed_results(results, expected_metadata, expected_individuals)
        completed_ids = {r["individual_id"] for r in results}
        return results, completed_ids
    return [], set()


def save_results(out_path, metadata, game_results):
    tmp_path = out_path + ".tmp"
    payload = dict(metadata)
    payload["results"] = game_results
    _validate_completed_results(game_results, metadata)
    # Serialize before touching any file; JSON NaN/Infinity are invalid results.
    serialized = json.dumps(payload, indent=2, allow_nan=False)
    with open(tmp_path, "w") as f:
        f.write(serialized)
    os.replace(tmp_path, out_path)


def build_metadata(variant, game_id, total_frames, eval_frequency, num_eval_episodes,
                   recipe="shared", eval_limit="raise", config_overrides=None):
    """Result-file header. Follow-up options are recorded only when they differ
    from the RCT defaults, so default metadata keeps its original form."""
    import torch

    settings = training_settings(recipe, eval_limit)
    metadata = {
        "variant": variant,
        "variant_label": VARIANT_LABELS[variant],
        "game_id": game_id,
        "total_frames": total_frames,
        "eval_frequency": eval_frequency,
        "num_eval_episodes": num_eval_episodes,
        "eval_seeds": EVAL_SEEDS[:num_eval_episodes],
        "training_protocol": {
            "schema_version": 2,
            "shared_settings": deepcopy(settings),
            "treatment_settings": deepcopy(TREATMENT_DEFAULTS.get(variant, {})),
        },
        "provenance": {
            "source_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(Path(__file__).parent.glob("*.py"))
            },
            "python_version": platform.python_version(),
            "torch_num_threads": torch.get_num_threads(),
            "package_versions": {name: version(name) for name in ("numpy", "torch", "gymnasium", "minatar")},
        },
    }
    if recipe != "shared":
        metadata["training_protocol"]["recipe"] = recipe
    if config_overrides:
        metadata["training_protocol"]["config_overrides"] = deepcopy(config_overrides)
    return metadata


def training_settings(recipe="shared", eval_limit="raise"):
    """The recipe's shared settings with the chosen evaluation-limit behaviour."""
    from rct.constants import EVAL_LIMIT_BEHAVIORS, RECIPES

    if recipe not in RECIPES:
        raise ValueError(f"Unknown recipe: {recipe}")
    if eval_limit not in EVAL_LIMIT_BEHAVIORS:
        raise ValueError(f"Unknown evaluation limit behaviour: {eval_limit}")
    settings = deepcopy(RECIPES[recipe])
    settings["evaluation_limit_behavior"] = EVAL_LIMIT_BEHAVIORS[eval_limit]
    return settings


def _validate_completed_results(results, metadata, individuals=None):
    from rct.evaluation import compute_auc

    expected_length = metadata["total_frames"] // metadata["eval_frequency"]
    cohort = {r["individual_id"]: r for r in individuals} if individuals is not None else None
    ids = set()
    for run in results:
        individual_id = run["individual_id"]
        if individual_id in ids:
            raise ValueError(f"Duplicate completed individual {individual_id}")
        ids.add(individual_id)
        if cohort is not None:
            expected = cohort.get(individual_id)
            if expected is None or (run["seed"], run["config"]) != (expected["seed"], expected["config"]):
                raise ValueError(f"Resume cohort mismatch for individual {individual_id}")
        trajectory = run["eval_trajectory"]
        if len(trajectory) != expected_length or not trajectory or not all(math.isfinite(x) for x in trajectory):
            raise ValueError(f"Incomplete or nonfinite evaluation trajectory for individual {individual_id}")
        auc = compute_auc(trajectory, metadata["eval_frequency"])
        if not math.isfinite(run["auc"]) or not math.isclose(auc, run["auc"], rel_tol=1e-12, abs_tol=1e-6):
            raise ValueError(f"AUC mismatch for individual {individual_id}")
