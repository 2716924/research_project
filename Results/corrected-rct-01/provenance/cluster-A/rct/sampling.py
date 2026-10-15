"""Draw hyperparameter individuals from the shared population space."""

import random

from rct.constants import POPULATION_SEED, POPULATION_SPACE, TRAINING_SEED_BASE


def sample_individuals(num_individuals, population_seed=POPULATION_SEED):
    """
    Sample N individuals (hyperparameter configs) for the RCT.

    Each individual is one random draw from Table 4.1 in the proposal.
    The same seed always produces the same cohort — so every treatment arm
    trains on identical individuals (paired design).
    """
    rng = random.Random(population_seed)
    individuals = []
    for j in range(num_individuals):
        config = {param: rng.choice(values) for param, values in POPULATION_SPACE.items()}
        individuals.append({
            "individual_id": j,
            "seed": TRAINING_SEED_BASE + j,
            "config": config,
        })
    return individuals
