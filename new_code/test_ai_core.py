import os
import tempfile

import numpy as np

from abstraction import abstractioncards_street_aware, preflop_betting_context
from ai import (
    ActionAbstractor,
    DecisionAction,
    LinearMCCFRTrainer,
    PreflopActionAbstractor,
    StrategyTable,
    TrainingCheckpointer,
    _default_path,
    bet_sizing_for_checkpoint,
    game_config_for_checkpoint,
    linear_weight,
    linear_weight_total,
    load_table,
    merge_strategy_tables,
    preflop_hand_strength,
    save_table,
    table_metrics,
)
from abstraction import parse_poker_string
from const import game_config
from preflop import hand_matrix_position


class FakeGame:
    def num_players(self):
        return 2

    def new_initial_state(self):
        return FakeTrainingState()


class FakeState:
    def __init__(self, round_id=0, terminal_after_action=None):
        self.round_id = round_id
        self.terminal_after_action = terminal_after_action

    def information_state_string(self):
        return (
            f"[Round {self.round_id}]"
            "[Player: 0]"
            "[Pot: 100]"
            "[Money: 20000 20000]"
            "[Private: AcKd]"
            "[Public: ]"
            "[Sequences: c]"
        )

    def clone(self):
        return FakeState(self.round_id, self.terminal_after_action)

    def apply_action(self, action):
        self.applied_action = action

    def is_terminal(self):
        return getattr(self, "applied_action", None) == self.terminal_after_action


class FakeActionState(FakeState):
    def __init__(self):
        super().__init__()
        self.labels = {
            0: "player=0 move=Fold",
            1: "player=0 move=Call",
            200: "player=0 move=Bet200",
            250: "player=0 move=Bet250",
            300: "player=0 move=Bet300",
            500: "player=0 move=Bet500",
            4975: "player=0 move=Bet4975",
            9950: "player=0 move=Bet9950",
            14925: "player=0 move=Bet14925",
            20000: "player=0 move=Bet20000",
        }

    def current_player(self):
        return 0

    def information_state_string(self):
        return (
            "[Round 0]"
            "[Player: 0]"
            "[Pot: 100]"
            "[Money: 20000 20000]"
            "[Private: AcKd]"
            "[Public: ]"
            "[Sequences: ]"
        )

    def action_to_string(self, player, action):
        return self.labels[int(action)]


class FakeTrainingState:
    def __init__(self, applied_action=None):
        self.applied_action = applied_action

    def clone(self):
        return FakeTrainingState(self.applied_action)

    def apply_action(self, action):
        self.applied_action = action

    def is_terminal(self):
        return self.applied_action is not None

    def is_chance_node(self):
        return False

    def current_player(self):
        return 0

    def legal_actions(self):
        return [1, 2]

    def returns(self):
        if self.applied_action == 1:
            return [1.0, -1.0]
        return [-1.0, 1.0]

    def information_state_string(self):
        return "root"


class FakePostflopState(FakeState):
    def current_player(self):
        return 0

    def information_state_string(self, player=None):
        player = 0 if player is None else int(player)
        private = "AcAd" if player == 0 else "7c2d"
        return (
            "[Round 1]"
            f"[Player: {player}]"
            "[Pot: 600]"
            "[Money: 9700 9700]"
            f"[Private: {private}]"
            "[Public: 2c7dJh]"
            "[Sequences: r300c/]"
        )


def test_strategy_table_regret_matching():
    table = StrategyTable()
    infoset = "root"
    table.add_regret(infoset, 1, 3)
    table.add_regret(infoset, 2, 1)

    strategy = table.regret_matching(infoset, [1, 2])

    assert strategy == [0.75, 0.25]


def test_average_strategy_uses_linear_weight():
    table = StrategyTable()
    table.add_average_strategy("root", [1, 2], [0.25, 0.75], weight=2)
    table.add_average_strategy("root", [1, 2], [0.75, 0.25], weight=4)

    average = table.average_strategy()["root"]

    assert np.isclose(average["1"], 7 / 12)
    assert np.isclose(average["2"], 5 / 12)


def test_strategy_table_accepts_labeled_action_keys():
    table = StrategyTable()
    infoset = "root"
    actions = [
        DecisionAction("open_2_5bb", 250),
        DecisionAction("open_3bb", 300),
    ]
    table.add_regret(infoset, actions[0], 3)
    table.add_regret(infoset, actions[1], 1)

    strategy = table.regret_matching(infoset, actions)

    assert strategy == [0.75, 0.25]
    assert set(table.data[infoset]["regret"]) == {"open_2_5bb", "open_3bb"}


def test_worker_table_merge_sums_sparse_fields_exactly():
    first = StrategyTable()
    first.add_regret("root", 1, 3)
    first.add_average_strategy("root", [1, 2], [0.25, 0.75], weight=2)

    second = StrategyTable()
    second.add_regret("root", 1, 4)
    second.add_regret("root", 2, -1)
    second.add_average_strategy("root", [1, 2], [0.50, 0.50], weight=4)

    merged = merge_strategy_tables([first, second])
    node = merged.data["root"]

    assert node["regret"]["1"] == 7
    assert node["regret"]["2"] == -1
    assert node["strategy_sum"]["1"] == 2.5
    assert node["strategy_sum"]["2"] == 3.5
    assert node["visits"] == 2


def test_save_and_load_table_roundtrip():
    table = StrategyTable()
    table.add_regret("root", 1, 3)

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "table.json")
        save_table(table, path)
        loaded = load_table(path)

    assert loaded.data == table.data


def test_save_and_load_compressed_table_roundtrip():
    table = StrategyTable()
    table.add_regret("root", 1, 3)

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "table.json.gz")
        save_table(table, path)
        loaded = load_table(path)

    assert loaded.data == table.data


def test_pruning_keeps_terminal_actions_and_drops_bad_regrets():
    table = StrategyTable()
    trainer = LinearMCCFRTrainer(
        FakeGame(),
        table=table,
        prune_threshold=-10,
        prune_after=0,
    )
    infoset = (
        "[1323][00][c]"
    )
    table.ensure_infoset(infoset, [1, 2, 3])
    table.add_regret(infoset, 1, -20)
    table.add_regret(infoset, 2, -20)
    table.add_regret(infoset, 3, 1)

    kept = trainer._traverser_actions(
        FakeState(terminal_after_action=2),
        infoset,
        [1, 2, 3],
        allow_pruning=True,
    )

    assert kept == [2, 3]


def test_action_abstractor_selects_interpretable_buckets():
    abstractor = ActionAbstractor(
        pot_fractions=(1.0,),
        stack_fractions=(0.5,),
        random_probe_count=0,
        max_actions=None,
    )
    state = FakeActionState()

    selected = abstractor.select_actions(
        state,
        [0, 1, 200, 250, 500, 4975, 9950, 14925, 20000],
    )
    details = abstractor.describe_actions(state, [], selected)

    assert 0 in selected
    assert 1 in selected
    assert 200 in selected
    assert 9950 in selected
    assert 20000 in selected
    assert details["pot"] == 100
    assert details["active_stack"] == 20000
    assert details["opponent_stack"] == 20000
    assert details["effective_stack"] == 20000


def test_preflop_action_abstractor_returns_legal_stiff_menu():
    abstractor = PreflopActionAbstractor(big_blind=100)
    state = FakeActionState()
    legal = [0, 1, 200, 250, 500, 4975, 9950, 14925, 20000]

    selected = abstractor.select_actions(state, legal)

    assert all(action in legal for action in selected)
    assert 0 in selected
    assert 1 in selected
    assert 20000 in selected
    assert 200 in selected
    assert 250 in selected
    assert 9950 in selected


def test_preflop_action_abstractor_returns_labeled_specs():
    abstractor = PreflopActionAbstractor(big_blind=100)
    state = FakeActionState()
    legal = [0, 1, 200, 250, 300, 4975, 9950, 20000]

    specs = abstractor.select_action_specs(state, legal)
    keys = [spec.key for spec in specs]

    assert all(int(spec) in legal for spec in specs)
    assert "fold" in keys
    assert "call" in keys
    assert "min_raise" in keys
    assert "open_2_5bb" in keys
    assert "open_3bb" in keys
    assert "jam" in keys


def test_preflop_abstraction_is_lossless_by_rank_pair_and_suitedness():
    assert abstractioncards_street_aware({"Private": ["Ac", "Kd"], "Public": []}) == "[PF:AKo]"
    assert abstractioncards_street_aware({"Private": ["Ad", "Kc"], "Public": []}) == "[PF:AKo]"
    assert abstractioncards_street_aware({"Private": ["Ac", "Kc"], "Public": []}) == "[PF:AKs]"
    assert abstractioncards_street_aware({"Private": ["7d", "7s"], "Public": []}) == "[PF:77]"


def test_preflop_betting_context_buckets_raise_sizes():
    context = preflop_betting_context(
        {
            "Player": 1,
            "Money": [9750, 9950],
            "Sequences": "r250r1000",
        },
        big_blind=100,
    )

    assert "[pos:P1]" in context
    assert "[tc:2bb]" in context
    assert "[eff:100bb]" in context
    assert "[seq:r:2_5bb|r:4x]" in context


def test_preflop_hand_matrix_maps_suited_offsuit_and_pairs():
    assert hand_matrix_position("AKs") == (0, 1)
    assert hand_matrix_position("AKo") == (1, 0)
    assert hand_matrix_position("77") == (7, 7)


def test_preflop_cutoff_values_stronger_hands_higher():
    trainer = LinearMCCFRTrainer(FakeGame(), cutoff_street="preflop")
    state = FakePostflopState()

    aa_value = trainer._cutoff_value(state, 0)
    weak_value = trainer._cutoff_value(state, 1)

    assert preflop_hand_strength(["Ac", "Ad"]) > preflop_hand_strength(["7c", "2d"])
    assert aa_value > weak_value


def test_postflop_abstraction_remains_lossy_bucket():
    bucket = abstractioncards_street_aware(
        {"Private": ["Ac", "Kd"], "Public": ["2c", "7d", "Jh"]}
    )

    assert bucket.startswith("[")
    assert not bucket.startswith("[PF:")


def test_checkpointer_writes_sparse_metrics_and_compressed_tables():
    with tempfile.TemporaryDirectory() as directory:
        trainer = LinearMCCFRTrainer(
            FakeGame(),
            infoset_fn=lambda state: "root",
        )
        checkpointer = TrainingCheckpointer(
            directory,
            checkpoint_iterations=[1],
        )

        trainer.train(2, checkpointer=checkpointer)

        metrics_path = os.path.join(directory, "metrics.csv")
        checkpoint_path = os.path.join(
            directory,
            "mccfr_table_iter_00000001.json.gz",
        )

        assert os.path.exists(metrics_path)
        assert os.path.exists(checkpoint_path)
        assert load_table(checkpoint_path).data

        with open(metrics_path, "r") as metrics_file:
            lines = metrics_file.read().splitlines()

        assert lines[0].startswith("iteration,")
        assert len(lines) == 2
        assert lines[1].startswith("1,")
        assert table_metrics(trainer.table)["infosets"] == 1


def _random_openspiel_decisions(hands, seed):
    """Yield (state, parsed) at every decision of random full-game hands."""
    import pyspiel

    game = pyspiel.load_game("universal_poker", game_config)
    rng = np.random.default_rng(seed)
    # v2 has no raise cap, so walking with it reaches long raise chains.
    walker = PreflopActionAbstractor(big_blind=100, starting_stack=10_000, bet_sizing="v2")
    for _ in range(hands):
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                actions, probs = zip(*state.chance_outcomes())
                state.apply_action(int(rng.choice(actions, p=probs)))
                continue
            parsed = parse_poker_string(state.information_state_string())
            yield state, parsed
            # Walk with small bets so hands reach the turn and river.
            specs = [s for s in walker.select_action_specs_direct(parsed) if s.key != "fold"]
            small = [s for s in specs if s.key != "jam"] or specs
            state.apply_action(int(small[int(rng.integers(len(small)))]))


def test_postflop_bet_sizing_is_legal_and_pot_relative():
    try:
        import pyspiel  # noqa: F401
    except ImportError:
        print("skipped: pyspiel not installed")
        return
    v2 = PreflopActionAbstractor(big_blind=100, starting_stack=10_000, bet_sizing="v2")
    v3 = PreflopActionAbstractor(big_blind=100, starting_stack=10_000, bet_sizing="v3")
    legacy = PreflopActionAbstractor(big_blind=100, starting_stack=10_000, bet_sizing="legacy")
    postflop_checked = 0
    for state, parsed in _random_openspiel_decisions(hands=300, seed=7):
        legal = set(state.legal_actions())
        for abstractor in (v2, v3, legacy):
            for spec in abstractor.select_action_specs_direct(parsed):
                assert int(spec) in legal, (abstractor.bet_sizing, spec, parsed)
        specs = {s.key: int(s) for s in v2.select_action_specs_direct(parsed)}
        raises = sorted(a for a in legal if a > 1)
        if raises and raises[0] < 10_000:
            assert specs["min_raise"] == raises[0], (specs, raises[0], parsed)
        if int(parsed["Round"]) == 0:
            continue
        money = parsed["Money"]
        if money[0] == money[1] and "bet_pot" in specs:
            committed = 10_000 - money[0]
            pot = 2 * committed
            if committed + pot < 10_000:
                assert specs["bet_pot"] - committed == pot, (specs, parsed)
            if committed + round(pot / 3) > specs["min_raise"]:
                assert specs["bet_33"] - committed == round(pot / 3), (specs, parsed)
        postflop_checked += 1
    assert postflop_checked > 200, postflop_checked


def test_bet_sizing_for_checkpoint_reads_manifest():
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = os.path.join(tmp, "mccfr_table_iter_00000100.json.gz")
        open(checkpoint, "w").close()
        assert bet_sizing_for_checkpoint(checkpoint) == "legacy"
        with open(os.path.join(tmp, "run_manifest.json"), "w") as manifest:
            manifest.write('{"schema_version": "preflop_parallel_v3"}')
        assert bet_sizing_for_checkpoint(checkpoint) == "legacy"
        with open(os.path.join(tmp, "run_manifest.json"), "w") as manifest:
            manifest.write('{"bet_sizing": "v2"}')
        assert bet_sizing_for_checkpoint(checkpoint) == "v2"
        assert bet_sizing_for_checkpoint(tmp) == "v2"


def test_small_blind_acts_first_preflop_and_big_blind_first_postflop():
    try:
        import pyspiel
    except ImportError:
        print("skipped: pyspiel not installed")
        return
    game = pyspiel.load_game("universal_poker", game_config)
    state = game.new_initial_state()
    while state.is_chance_node():
        state.apply_action(state.chance_outcomes()[0][0])
    parsed = parse_poker_string(state.information_state_string())
    # P1 posted the small blind (50), P0 the big blind (100).
    assert parsed["Player"] == 1 and parsed["Money"] == [9900, 9950]
    assert 0 in state.legal_actions()  # the small blind can fold
    state.apply_action(1)  # small blind calls
    assert state.current_player() == 0  # big blind has the option
    state.apply_action(1)  # big blind checks
    while state.is_chance_node():
        state.apply_action(state.chance_outcomes()[0][0])
    assert state.current_player() == 0  # big blind acts first on the flop


def test_game_config_for_checkpoint_keeps_the_rules_a_run_used():
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = os.path.join(tmp, "mccfr_table_iter_00000100.json.gz")
        open(checkpoint, "w").close()
        assert game_config_for_checkpoint(checkpoint)["firstPlayer"] == "1"
        with open(os.path.join(tmp, "run_manifest.json"), "w") as manifest:
            manifest.write('{"game_config": {"firstPlayer": "2 1 1 1", "stack": "5000 5000"}}')
        config = game_config_for_checkpoint(checkpoint)
        assert config["firstPlayer"] == "2 1 1 1"
        assert config["stack"] == "5000 5000"


def test_linear_weight_total_matches_the_sum_of_weights():
    for n in (0, 1, 99, 100, 101, 250, 1_000):
        for start in (0, 7, 99, 100, 5_000):
            for stride in (1, 3, 4, 150):
                expected = sum(linear_weight(start + t * stride) for t in range(1, n + 1))
                assert linear_weight_total(n, start, stride) == expected, (n, start, stride)


def test_table_metrics_regret_bound():
    table = StrategyTable()
    table.ensure_infoset("a", [1, 2])
    table.ensure_infoset("b", [1, 2])
    table.add_regret("a", 1, 30.0)
    table.add_regret("a", 2, 10.0)
    table.add_regret("b", 1, -5.0)
    metrics = table_metrics(table, weight_total=10)
    assert metrics["max_positive_regret_sum"] == 30.0
    assert metrics["avg_regret_bound"] == 3.0
    assert metrics["avg_regret_per_infoset"] == 1.5
    assert "avg_regret_bound" not in table_metrics(table)


def test_default_path_points_into_new_code():
    assert _default_path("x.json") == os.path.join(os.path.dirname(os.path.realpath(__file__)), "x.json")


def test_trainer_converges_like_openspiel_reference_on_leduc():
    """OpenSpiel's external-sampling MCCFR reaches ~0.55 after 4k iterations."""
    try:
        import pyspiel
        from open_spiel.python import policy as policy_lib
        from open_spiel.python.algorithms import exploitability
    except ImportError:
        print("skipped: pyspiel not installed")
        return
    game = pyspiel.load_game("leduc_poker")
    trainer = LinearMCCFRTrainer(
        game, infoset_fn=lambda state: state.information_state_string(),
        rng=np.random.default_rng(0),
    )
    trainer.train(4_000)
    policy = policy_lib.TabularPolicy(game)
    average = trainer.table.average_strategy()
    for infostate, index in policy.state_lookup.items():
        probs = average.get(infostate)
        if not probs:
            continue
        row = np.array([probs.get(str(a), 0.0) for a in range(game.num_distinct_actions())])
        row = row * policy.legal_actions_mask[index]
        if row.sum() > 0:
            policy.action_probability_array[index] = row / row.sum()
    assert exploitability.exploitability(game, policy) < 0.7


def test_v3_caps_raises_per_street():
    try:
        import pyspiel  # noqa: F401
    except ImportError:
        print("skipped: pyspiel not installed")
        return
    from ai import POSTFLOP_RAISE_CAP, PREFLOP_RAISE_CAP, _raise_amounts

    v3 = PreflopActionAbstractor(big_blind=100, starting_stack=10_000, bet_sizing="v3")
    capped_seen = 0
    for _, parsed in _random_openspiel_decisions(hands=400, seed=5):
        street = (parsed.get("Sequences") or "").rsplit("|", 1)[-1]
        raises = len(_raise_amounts(street))
        keys = {spec.key for spec in v3.select_action_specs_direct(parsed)}
        cap = PREFLOP_RAISE_CAP if int(parsed["Round"]) == 0 else POSTFLOP_RAISE_CAP
        if raises >= cap:
            assert keys <= {"fold", "call", "jam"}, (keys, parsed)
            capped_seen += 1
        elif int(parsed["Round"]) > 0 and raises > 0:
            # Facing a postflop bet: a pot-sized raise or a jam, no min-raises.
            assert "min_raise" not in keys, (keys, parsed)
    assert capped_seen > 20, capped_seen


if __name__ == "__main__":
    test_strategy_table_regret_matching()
    test_average_strategy_uses_linear_weight()
    test_strategy_table_accepts_labeled_action_keys()
    test_worker_table_merge_sums_sparse_fields_exactly()
    test_save_and_load_table_roundtrip()
    test_save_and_load_compressed_table_roundtrip()
    test_pruning_keeps_terminal_actions_and_drops_bad_regrets()
    test_action_abstractor_selects_interpretable_buckets()
    test_preflop_action_abstractor_returns_legal_stiff_menu()
    test_preflop_action_abstractor_returns_labeled_specs()
    test_preflop_abstraction_is_lossless_by_rank_pair_and_suitedness()
    test_preflop_betting_context_buckets_raise_sizes()
    test_preflop_hand_matrix_maps_suited_offsuit_and_pairs()
    test_preflop_cutoff_values_stronger_hands_higher()
    test_postflop_abstraction_remains_lossy_bucket()
    test_checkpointer_writes_sparse_metrics_and_compressed_tables()
    test_postflop_bet_sizing_is_legal_and_pot_relative()
    test_bet_sizing_for_checkpoint_reads_manifest()
    test_small_blind_acts_first_preflop_and_big_blind_first_postflop()
    test_game_config_for_checkpoint_keeps_the_rules_a_run_used()
    test_linear_weight_total_matches_the_sum_of_weights()
    test_table_metrics_regret_bound()
    test_default_path_points_into_new_code()
    test_trainer_converges_like_openspiel_reference_on_leduc()
    test_v3_caps_raises_per_street()
    print("ai core tests passed")
