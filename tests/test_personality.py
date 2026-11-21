import json
import math
import os
import sys

import chess
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import validate_dimensions as V  # noqa: E402
from test_echo import learn_args, v1_file, write_random_games  # noqa: E402

import echo  # noqa: E402
import personality as P  # noqa: E402


def sig(name, fen, uci):
    board = chess.Board(fen)
    return P.signal_of(name, board, board.parse_uci(uci))


# ------------------------------------------------------------------ what each signal means
def test_aggression_is_capture_or_check():
    fen = "4k3/8/8/3n4/8/1B6/8/4K3 w - - 0 1"
    assert sig("aggression", fen, "b3d5") == 1            # capture
    assert sig("aggression", fen, "b3c4") == 0            # quiet
    assert sig("aggression", "4k3/8/8/8/8/8/8/R3K3 w - - 0 1", "a1a8") == 1   # check


def test_risk_counts_the_position_after_the_move():
    undefended = "4r1k1/8/8/8/8/8/6N1/3K4 w - - 0 1"       # Ne3 is hit by the rook and nothing defends it
    defended = "4r1k1/8/8/8/8/8/3K2N1/8 w - - 0 1"         # the same square, but the king defends it
    pawn_hit = "4k3/8/8/3p4/8/6N1/8/4K3 w - - 0 1"         # Ne4 is attacked by a cheaper piece, a pawn
    assert sig("risk_tolerance", undefended, "g2e3") == 1
    assert sig("risk_tolerance", defended, "g2e3") == 0    # attacked by a rook (worth more) but defended
    assert sig("risk_tolerance", pawn_hit, "g3e4") == 1
    assert sig("risk_tolerance", pawn_hit, "g3f5") == 0


def test_tactical_appetite_needs_a_new_threat():
    fen = "4k3/8/8/3q4/8/8/8/1N2K3 w - - 0 1"
    assert sig("tactical_appetite", fen, "b1c3") == 1      # the knight now attacks the queen
    assert sig("tactical_appetite", fen, "b1a3") == 0
    # an attack that already existed before the move is not new
    already = "4k3/8/8/3q4/8/2N5/8/4K3 w - - 0 1"
    assert sig("tactical_appetite", already, "c3b1") == 0


def test_simplification_is_a_piece_for_piece_capture():
    assert sig("simplification", "4k3/8/8/3n4/8/1B6/8/4K3 w - - 0 1", "b3d5") == 1
    assert sig("simplification", "4k3/8/8/3n4/4P3/8/8/4K3 w - - 0 1", "e4d5") == 0   # a pawn takes: not a trade-down
    assert sig("simplification", "4k3/8/8/3r4/8/1B6/8/4K3 w - - 0 1", "b3d5") == 0   # bishop for rook is unequal


def test_initiative_is_a_pawn_advance_to_the_fourth_rank():
    assert sig("initiative", chess.STARTING_FEN, "e2e4") == 1
    assert sig("initiative", chess.STARTING_FEN, "e2e3") == 0
    black = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    assert sig("initiative", black, "e7e5") == 1
    assert sig("initiative", black, "e7e6") == 0


def test_positional_is_classical_development():
    assert sig("positional", chess.STARTING_FEN, "g1f3") == 1
    assert sig("positional", chess.STARTING_FEN, "e2e4") == 1     # occupies the centre
    assert sig("positional", chess.STARTING_FEN, "a2a3") == 0
    assert sig("positional", chess.STARTING_FEN, "h2h3") == 0
    assert sig("positional", "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1") == 1   # castling


def test_scopes_exclude_what_the_dimension_is_not_about():
    board = chess.Board("4k3/8/8/3n4/8/1B6/8/4K3 w - - 0 1")
    assert not P.in_scope("risk_tolerance", board, board.parse_uci("b3d5"))     # captures are out of scope for risk
    assert not P.in_scope("tactical_appetite", board, board.parse_uci("b3d5"))
    assert not P.in_scope("positional", board, board.parse_uci("b3d5"))        # forcing moves are not "quiet"
    assert not P.in_scope("endgame_activity", chess.Board(), chess.Move.from_uci("e2e4"))   # not an endgame


def test_endgame_king_activity():
    fen = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
    assert P._is_endgame(chess.Board(fen)) and not P._is_endgame(chess.Board())
    assert sig("endgame_activity", fen, "e1d2") == 1
    assert sig("endgame_activity", fen, "e1f1") == 0


# ------------------------------------------------------------------ the statistic
def test_one_decision_gives_the_textbook_numbers():
    board = chess.Board("4k3/8/8/3n4/8/1B6/8/4K3 w - - 0 1")
    moves = list(board.legal_moves)
    k = len(moves)
    forcing = sum(1 for m in moves if board.is_capture(m) or board.gives_check(m))
    p = forcing / k
    tally = P.new_tally()
    P.observe(board, board.parse_uci("b3d5"), moves, tally)
    n, observed, expected, variance = tally["aggression"][0]
    assert (n, observed) == (1, 1.0)
    assert expected == pytest.approx(p) and variance == pytest.approx(p * (1 - p))
    assert P.summarize(tally)["aggression"]["z"] == pytest.approx((1 - p) / math.sqrt(p * (1 - p)))


def test_decisions_that_cannot_differ_are_ignored():
    tally = P.new_tally()
    board = chess.Board()
    P.observe(board, chess.Move.from_uci("e2e4"), list(board.legal_moves), tally)
    assert tally["aggression"][0][0] == 0                   # no capture or check exists, so nothing to learn
    assert tally["endgame_activity"][0][0] == 0             # not an endgame


def test_played_move_outside_the_scope_is_not_counted():
    board = chess.Board("4k3/8/8/3n4/8/1B6/8/4K3 w - - 0 1")
    tally = P.new_tally()
    P.observe(board, board.parse_uci("b3d5"), list(board.legal_moves), tally)    # a capture
    assert tally["risk_tolerance"][0][0] == 0 and tally["tactical_appetite"][0][0] == 0


def test_merge_and_metadata_round_trip():
    rng = np.random.default_rng(3)
    a, b = V.play_games(rng, 4, 30), V.play_games(rng, 4, 30)
    merged = P.merge_tally(a, b)
    for n in P.NAMES:
        for h in (0, 1):
            assert merged[n][h][0] == a[n][h][0] + b[n][h][0]
            assert merged[n][h][3] == pytest.approx(a[n][h][3] + b[n][h][3])
    back, games = P.from_meta(json.loads(json.dumps(P.to_meta(merged, 8))))     # as stored in the profile
    assert games == 8 and back == merged


def test_unreadable_metadata_is_treated_as_missing():
    assert P.from_meta(None) == (None, 0)
    assert P.from_meta({"version": 99, "tally": {}}) == (None, 0)


@pytest.mark.parametrize("n,z,halves,expected", [
    (10, 9.0, [5, 5], "too little data"),
    (200, 3.4, [2.5, 2.4], "clear: more than random"),
    (200, -3.4, [-2.5, -2.4], "clear: less than random"),
    (200, 3.4, [4.0, -0.3], "not distinguishable from random"),                       # the halves disagree
    (200, 2.4, [1.8, 1.6], "weak: more than random"),
    (200, 1.9, [1.5, 1.4], "not distinguishable from random"),
])
def test_verdict_thresholds(n, z, halves, expected):
    assert P.verdict(n, z, halves) == expected


# ------------------------------------------------------------------ does it measure what it says?
def test_random_players_are_not_flagged():
    rng = np.random.default_rng(11)
    zs = {n: [] for n in P.NAMES}
    clear = 0
    for _ in range(12):
        s = P.summarize(V.play_games(rng, 10, 40))
        clear += sum(r["verdict"].startswith("clear") for r in s.values())
        for n, r in s.items():
            if r["n"] >= P.MIN_POSITIONS:
                zs[n].append(r["z"])
    assert clear <= 1                                       # 12 players x 7 dimensions, expect ~0.3
    for n, z in zs.items():
        if len(z) >= 8:
            assert abs(np.mean(z)) < 1.0 and 0.5 < np.std(z) < 1.6, n


@pytest.mark.parametrize("dim", ["aggression", "risk_tolerance", "tactical_appetite", "initiative", "positional",
                                 "endgame_activity"])
def test_planted_preference_is_recovered_in_the_right_direction(dim):
    rng = np.random.default_rng(5)
    up = P.summarize(V.play_games(rng, 12, 40, dim, +2.0))[dim]
    down = P.summarize(V.play_games(rng, 12, 40, dim, -2.0))[dim]
    assert up["z"] > 3 and up["verdict"].startswith("clear: more")
    assert down["z"] < 0 and down["z"] < up["z"] - 5


def test_measured_effect_rises_with_the_planted_strength():
    rng = np.random.default_rng(9)
    effects = []
    for beta in (-2.0, -1.0, 0.0, 1.0, 2.0):
        row = P.summarize(V.play_games(rng, 12, 40, "aggression", beta))["aggression"]
        effects.append(row["observed_rate"] - row["expected_rate"])
    assert effects == sorted(effects)


# ------------------------------------------------------------------ in the profile
def test_learn_stores_dimensions_and_prints_them(tmp_path, capsys):
    pgn = tmp_path / "g.pgn"
    write_random_games(pgn, 20, seed=2)
    out = tmp_path / "s.npz"
    echo.learn(learn_args(pgn, out))
    printed = capsys.readouterr().out
    _, meta = echo.load_model(out)
    tally, games = P.from_meta(meta["dimensions"])
    assert games == 20 and set(tally) == set(P.NAMES)
    assert tally["aggression"][0][0] + tally["aggression"][1][0] > 0
    assert "Behavior, measured from 20 games" in printed
    assert all(d.label in printed for d in P.DIMENSIONS)
    assert "## Behavior" in echo.export_markdown(meta)


def test_resume_adds_new_games_to_the_measurements(tmp_path):
    a, b = tmp_path / "a.pgn", tmp_path / "b.pgn"
    write_random_games(a, 12, seed=1)
    write_random_games(b, 12, seed=2)

    def counts(out):
        _, meta = echo.load_model(out)
        tally, games = P.from_meta(meta["dimensions"])
        return {n: tally[n][0][0] + tally[n][1][0] for n in P.NAMES}, games

    echo.learn(learn_args(a, tmp_path / "a.npz"))
    echo.learn(learn_args(b, tmp_path / "b.npz"))
    echo.learn(learn_args(b, tmp_path / "ab.npz", resume=str(tmp_path / "a.npz")))
    (ca, _), (cb, _), (cab, games) = counts(tmp_path / "a.npz"), counts(tmp_path / "b.npz"), counts(tmp_path / "ab.npz")
    assert games == 24
    assert all(cab[n] == ca[n] + cb[n] for n in P.NAMES)


def test_resuming_an_older_profile_says_the_measurements_are_partial(tmp_path, capsys):
    pgn = tmp_path / "g.pgn"
    write_random_games(pgn, 12, seed=3)
    old = v1_file(tmp_path, echo.StyleNet(hidden=8), games=40, positions=400)
    out = tmp_path / "new.npz"
    echo.learn(learn_args(pgn, out, resume=str(old)))
    assert "cover only the games added now" in capsys.readouterr().out
    _, meta = echo.load_model(out)
    assert meta["games"] == 52 and meta["dimensions"]["games"] == 12     # honest about what is covered


def test_profiles_without_dimensions_still_work(tmp_path, capsys):
    path = v1_file(tmp_path, echo.StyleNet(hidden=8))
    _, meta = echo.load_model(path)
    assert echo.dimension_rows(meta) is None
    echo.print_profile(meta)
    assert "Behavior: not measured" in capsys.readouterr().out
    assert "Not measured" in echo.export_markdown(meta)
