"""Run the full RCT trial loop for one algorithmic variant."""

import gymnasium as gym

from rct.constants import GAMES, POPULATION_SPACE, VARIANT_LABELS
from rct.evaluation import compute_auc
from rct.results import build_metadata, load_existing_results, result_path, save_results, training_settings
from rct.sampling import sample_individuals
from rct.slurm_utils import should_stop_for_time_limit
from rct.torch_setup import DEVICE
from rct.trainer import train_individual
from rct.tracking import AttemptTracker, TrackingSettings


def execute_rct_trial(
    variant,
    num_individuals=30,
    total_frames=5_000_000,
    eval_frequency=50_000,
    num_eval_episodes=10,
    output_dir=".",
    games=None,
    pilot=False,
    log_every=1000,
    tracking=None,
    individual_ids=None,
    config_overrides=None,
    recipe="shared",
    eval_limit="raise",
):
    """
    Train one RCT arm (control or treatment) across games and individuals.

    Results are written to JSON files and can be resumed if a cluster job
    is interrupted. Text logging goes to stdout (captured by SLURM .out files),
    with optional W&B/JSONL metrics in a separate tracking directory.

    Follow-up runs may train only the cohort members in individual_ids (each
    keeps its own seed), replace config values with config_overrides, and use
    another recipe or evaluation-limit behaviour (see rct.constants.RECIPES).
    """
    import os

    if variant not in VARIANT_LABELS:
        raise ValueError(f"Unknown variant: {variant}")
    if num_individuals < 1 or total_frames < 1 or eval_frequency < 1 or log_every < 1:
        raise ValueError("Individual count, frame budget and logging/evaluation intervals must be positive")
    if total_frames % eval_frequency:
        raise ValueError("total_frames must be divisible by eval_frequency so the final policy is evaluated")
    from rct.constants import EVAL_SEEDS
    if not 1 <= num_eval_episodes <= len(EVAL_SEEDS):
        raise ValueError("num_eval_episodes must be between 1 and the number of fixed evaluation seeds")
    tracking = tracking or TrackingSettings()
    tracking.validate()
    os.makedirs(output_dir, exist_ok=True)
    games = games if games is not None else GAMES
    settings = training_settings(recipe, eval_limit)
    unknown = set(config_overrides or {}) - set(POPULATION_SPACE)
    if unknown:
        raise ValueError(f"Unknown config override keys: {sorted(unknown)}")
    if individual_ids is not None:
        individual_ids = sorted(set(individual_ids))
        if not individual_ids or individual_ids[0] < 0:
            raise ValueError("individual_ids must be non-negative and non-empty")
        selected = set(individual_ids)
        individuals = [ind for ind in sample_individuals(individual_ids[-1] + 1)
                       if ind["individual_id"] in selected]
        num_individuals = len(individuals)
    else:
        individuals = sample_individuals(num_individuals)
    for individual in individuals:
        individual["config"] = {**individual["config"], **(config_overrides or {})}

    print(f"Variant: {VARIANT_LABELS[variant]}")
    print(f"Using device: {DEVICE}")

    stopped_early = False

    for game_id in games:
        print(f"\n{'=' * 50}")
        print(f"RCT — {VARIANT_LABELS[variant]} — {game_id}")
        print(f"{'=' * 50}")

        if should_stop_for_time_limit():
            print("Approaching SLURM time limit — stopping before next game.")
            stopped_early = True
            break

        init_env = gym.make(game_id)
        in_channels = init_env.observation_space.shape[-1]
        num_actions = init_env.action_space.n
        init_env.close()

        out_path = result_path(output_dir, variant, game_id)
        metadata = build_metadata(
            variant, game_id, total_frames, eval_frequency, num_eval_episodes,
            recipe=recipe, eval_limit=eval_limit, config_overrides=config_overrides,
        )
        game_results, completed_ids = load_existing_results(out_path, metadata, individuals)

        if completed_ids:
            print(
                f"Resuming: {len(completed_ids)}/{num_individuals} individuals "
                f"already completed for {game_id}"
            )

        for position, individual in enumerate(individuals, 1):
            j = individual["individual_id"]

            if j in completed_ids:
                print(f"--- Skipping Individual {position}/{num_individuals} (id {j}; already done) ---")
                continue

            if should_stop_for_time_limit():
                print(
                    "Approaching SLURM time limit — stopping before next individual "
                    f"(completed {len(completed_ids)}/{num_individuals} for {game_id})."
                )
                stopped_early = True
                break

            print(f"\n--- Individual {position}/{num_individuals} (id {j}) ---")
            print(f"Seed: {individual['seed']} | Settings: {individual['config']}")

            tracker = AttemptTracker(tracking, metadata, individual, pilot=pilot)
            eval_truncations = [] if eval_limit == "truncate" else None
            try:
                eval_trajectory, finished = train_individual(
                    variant=variant,
                    game_id=game_id,
                    individual=individual,
                    in_channels=in_channels,
                    num_actions=num_actions,
                    total_frames=total_frames,
                    eval_frequency=eval_frequency,
                    num_eval_episodes=num_eval_episodes,
                    log_every=log_every,
                    metrics_callback=tracker.log if tracking.mode != "disabled" else None,
                    settings=settings,
                    eval_limit=eval_limit,
                    eval_truncations=eval_truncations,
                )
                if not finished:
                    tracker.finish("interrupted")
                    print(
                        f"Individual id {j} incomplete (time limit) — "
                        "progress saved only for completed individuals."
                    )
                    stopped_early = True
                    break

                computed_auc = compute_auc(eval_trajectory, eval_frequency)
                game_results.append({
                    "individual_id": j,
                    "seed": individual["seed"],
                    "config": individual["config"],
                    "eval_trajectory": eval_trajectory,
                    "auc": computed_auc,
                    "final_return": eval_trajectory[-1],
                    "attempt_id": tracker.attempt_id,
                })
                if eval_truncations is not None:
                    # Episodes per checkpoint that hit the step limit (partial returns).
                    game_results[-1]["eval_truncated_episodes"] = eval_truncations
                # Only mark W&B complete after the authoritative result is saved.
                save_results(out_path, metadata, game_results)
                completed_ids.add(j)
                tracker.finish("completed", {
                    "auc": computed_auc, "final_return": eval_trajectory[-1],
                    "last_10_mean_return": sum(eval_trajectory[-10:]) / len(eval_trajectory[-10:]),
                    "completed_frames": total_frames,
                })
                print(f"Completed {variant}. AUC: {computed_auc:.4f}")
            except BaseException:
                try:
                    tracker.finish("failed")
                except Exception:
                    print("W&B could not finalize the failed attempt; inspect its local journal.")
                raise

        print(f"\nSaved results to {out_path}")
        if stopped_early:
            break

    return stopped_early
