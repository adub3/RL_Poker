"""
Check the greedy best-response estimator against OpenSpiel's exact
exploitability on games small enough to solve exactly.

    .venv/bin/python -B tools/evaluation/test_best_response.py
"""

import sys
from pathlib import Path

import pyspiel
from open_spiel.python import policy as policy_lib
from open_spiel.python.algorithms import cfr, exploitability

sys.path.insert(0, str(Path(__file__).resolve().parent))

from best_response import GreedyBestResponse  # noqa: E402


def blueprint_dict(tabular):
    """OpenSpiel TabularPolicy -> {infostate: {action_key: prob}}."""
    blueprint = {}
    for infostate, index in tabular.state_lookup.items():
        probs = tabular.action_probability_array[index]
        mask = tabular.legal_actions_mask[index]
        blueprint[infostate] = {
            str(action): float(probs[action]) for action in range(len(probs)) if mask[action]
        }
    return blueprint


def describe(state):
    return state.information_state_string(), state.legal_actions()


def check(game_name, tabular, train_iterations, eval_hands, min_fraction=1.0):
    """The estimate is a lower bound: it may fall short of the exact value by
    up to (1 - min_fraction) of it, and may exceed it only by sampling error."""
    game = pyspiel.load_game(game_name)
    exact = exploitability.exploitability(game, tabular)
    result = GreedyBestResponse(describe, blueprint_dict(tabular), seed=1).compute(
        game, train_iterations, eval_hands
    )
    estimate, stderr = result["exploitability"], result["stderr"]
    print(f"{game_name:12s} exact {exact:.4f}  estimate {estimate:.4f} ± {1.96 * stderr:.4f} (95%)")
    assert estimate < exact + 4 * stderr, (game_name, exact, result)
    assert estimate > min_fraction * exact - 4 * stderr, (game_name, exact, result)
    for seat in result["seats"]:
        assert seat["blueprint_missing_rate"] == 0.0
    return result


def test_kuhn_uniform():
    game = pyspiel.load_game("kuhn_poker")
    check("kuhn_poker", policy_lib.TabularPolicy(game), 2_000, 40_000)


def test_kuhn_near_equilibrium():
    game = pyspiel.load_game("kuhn_poker")
    solver = cfr.CFRSolver(game)
    for _ in range(200):
        solver.evaluate_and_update_policy()
    check("kuhn_poker", solver.average_policy(), 4_000, 40_000)


def test_leduc_partially_trained():
    game = pyspiel.load_game("leduc_poker")
    solver = cfr.CFRSolver(game)
    for _ in range(20):
        solver.evaluate_and_update_policy()
    # Leduc has deep lines the responder reaches rarely, so it converges more
    # slowly: 3k iterations gives ~70% of the exact value, 30k gives ~90%.
    check("leduc_poker", solver.average_policy(), 30_000, 30_000, min_fraction=0.85)


def test_missing_blueprint_entries_are_counted():
    game = pyspiel.load_game("kuhn_poker")
    result = GreedyBestResponse(describe, {}, seed=2).compute(game, 200, 2_000)
    for seat in result["seats"]:
        assert seat["blueprint_missing_rate"] == 1.0


if __name__ == "__main__":
    test_kuhn_uniform()
    test_kuhn_near_equilibrium()
    test_leduc_partially_trained()
    test_missing_blueprint_entries_are_counted()
    print("best response tests passed")
