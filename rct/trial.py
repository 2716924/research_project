"""Run the full RCT trial loop for one algorithmic variant."""

import gymnasium as gym

from rct.constants import GAMES, VARIANT_LABELS
from rct.evaluation import compute_auc
from rct.results import build_metadata, load_existing_results, result_path, save_results
from rct.sampling import sample_individuals
from rct.slurm_utils import should_stop_for_time_limit
from rct.torch_setup import DEVICE
from rct.trainer import train_individual


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
):
    """
    Train one RCT arm (control or treatment) across games and individuals.

    Results are written to JSON files and can be resumed if a cluster job
    is interrupted. All logging goes to stdout (captured by SLURM .out files).
    """
    import os

    os.makedirs(output_dir, exist_ok=True)
    games = games if games is not None else GAMES
    individuals = sample_individuals(num_individuals)

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
            variant, game_id, total_frames, eval_frequency, num_eval_episodes
        )
        game_results, completed_ids = load_existing_results(out_path)

        if completed_ids:
            print(
                f"Resuming: {len(completed_ids)}/{num_individuals} individuals "
                f"already completed for {game_id}"
            )

        for individual in individuals:
            j = individual["individual_id"]

            if j in completed_ids:
                print(f"--- Skipping Individual {j + 1}/{num_individuals} (already done) ---")
                continue

            if should_stop_for_time_limit():
                print(
                    "Approaching SLURM time limit — stopping before next individual "
                    f"(completed {len(completed_ids)}/{num_individuals} for {game_id})."
                )
                stopped_early = True
                break

            print(f"\n--- Individual {j + 1}/{num_individuals} ---")
            print(f"Seed: {individual['seed']} | Settings: {individual['config']}")

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
            )

            if not finished:
                print(
                    f"Individual {j + 1} incomplete (time limit) — "
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
            })
            completed_ids.add(j)
            print(f"Completed {variant}. AUC: {computed_auc:.4f}")

            save_results(out_path, metadata, game_results)

        print(f"\nSaved results to {out_path}")
        if stopped_early:
            break

    return stopped_early
