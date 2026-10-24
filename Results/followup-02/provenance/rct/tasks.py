"""Single source of truth for the variant-major SLURM array."""

import argparse

from rct.constants import GAMES, VARIANTS


def task_assignment(task_id):
    if not 0 <= task_id < len(VARIANTS) * len(GAMES):
        raise ValueError("Array task must be between 0 and 24")
    variant = VARIANTS[task_id // len(GAMES)]
    game = GAMES[task_id % len(GAMES)]
    slug = game.split("/")[1].removesuffix("-v1").lower()
    return variant, game, slug


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_id", type=int)
    args = parser.parse_args()
    try:
        print(*task_assignment(args.task_id))
    except ValueError as exc:
        parser.error(str(exc))
