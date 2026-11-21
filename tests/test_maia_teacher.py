"""Tests for the optional Maia teacher.

Most tests use a fake teacher, so they need neither lc0 nor Maia weights. The integration tests at the bottom run a
real lc0 + Maia-1 and are skipped unless ECHOSTAR_LC0 and ECHOSTAR_MAIA_WEIGHTS point at them.
"""
import json
import os
import random
import sys
import zlib
from argparse import Namespace

import chess
import chess.engine
import chess.pgn
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_echo import FakeEngine, StubBrain, learn_args, write_random_games  # noqa: E402

import echo  # noqa: E402
import maia_teacher as mt  # noqa: E402


class FakeTeacher:
    """Deterministic pseudo-probabilities per (position, move); counts calls."""
    name = "fake:maia-test"
    elo = 1500

    def __init__(self, favor=None, strength=0.0):
        self.calls, self.favor, self.strength = 0, favor, strength

    def probs(self, board, moves=None):
        moves = list(board.legal_moves) if moves is None else moves
        self.calls += 1
        rng = np.random.default_rng(zlib.crc32(mt.position_key(board).encode()))
        raw = rng.random(board.legal_moves.count())
        by_uci = dict(zip((m.uci() for m in board.legal_moves), raw))
        if self.favor:
            by_uci[self.favor] = by_uci.get(self.favor, 0) + self.strength
        return mt._normalize([by_uci[m.uci()] for m in moves])

    def close(self):
        pass


# ---------------------------------------------------------------- parsing and file names
LC0_LINES = [
    "e1e2  (106 ) N:       0 (+ 0) (P:  0.02%) (WL:  -.-----) (D: -.---) (M:  -.-) (Q:  0.05519) (U: 0.00039) (S:  0.05558) (V:  -.----) ",
    "f1c4  (139 ) N:       0 (+ 0) (P: 14.28%) (WL:  -.-----) (D: -.---) (M:  -.-) (Q:  0.05519) (U: 0.24924) (S:  0.30443) (V:  -.----) ",
    "g1f3  (159 ) N:       0 (+ 0) (P: 46.94%) (WL:  -.-----) (D: -.---) (M:  -.-) (Q:  0.05519) (U: 0.81908) (S:  0.87427) (V:  -.----) ",
    "node  (  29) N:       1 (+ 0) (P:  0.00%) (WL:  0.05244) (D: 0.038) (M:  0.0) (Q:  0.05244) (V:  0.0552) ",
    "some unrelated info string",
]


def test_parse_verbose_move_stats_reads_moves_and_skips_the_node_line():
    p = mt.parse_verbose_move_stats(LC0_LINES)
    assert p == {"e1e2": pytest.approx(0.0002), "f1c4": pytest.approx(0.1428), "g1f3": pytest.approx(0.4694)}


def test_maia1_rating_from_file_name_and_nearest_weights(tmp_path):
    assert mt.maia1_elo("/x/maia-1500.pb.gz") == 1500
    assert mt.maia1_elo("maia-1100.pb") == 1100
    assert mt.maia1_elo("net.pb.gz") is None
    for elo in (1100, 1500, 1900):
        (tmp_path / f"maia-{elo}.pb.gz").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("not weights")
    assert mt.nearest_maia1_weights(1620, str(tmp_path)).endswith("maia-1500.pb.gz")
    assert mt.nearest_maia1_weights(1750, str(tmp_path)).endswith("maia-1900.pb.gz")
    assert mt.nearest_maia1_weights(800, str(tmp_path)).endswith("maia-1100.pb.gz")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(mt.TeacherError, match="no maia"):
        mt.nearest_maia1_weights(1500, str(empty))


def test_missing_lc0_weights_is_a_clear_error(tmp_path):
    with pytest.raises(mt.TeacherError, match="weights not found"):
        mt.Lc0Teacher("lc0", str(tmp_path / "nope.pb.gz"))


def test_unstartable_lc0_is_a_clear_error(tmp_path):
    weights = tmp_path / "maia-1500.pb.gz"
    weights.write_bytes(b"x")
    teacher = mt.Lc0Teacher(str(tmp_path / "no-such-lc0"), str(weights))
    with pytest.raises(mt.TeacherError, match="could not start lc0"):
        teacher.probs(chess.Board())


# ---------------------------------------------------------------- cache
def test_cache_roundtrip_is_exact_and_independent_of_move_order(tmp_path):
    path = str(tmp_path / "maia.jsonl")
    live = FakeTeacher()
    cache = mt.CachingTeacher(path, live)
    board = chess.Board()
    moves = list(board.legal_moves)
    first = cache.probs(board, moves)
    assert cache.misses == 1 and live.calls == 1
    cache.probs(board, moves)
    assert cache.hits == 1 and live.calls == 1          # second call did not reach the teacher
    cache.flush()

    replay = mt.CachingTeacher(path)                    # no live teacher: read-only replay
    assert replay.name == "fake:maia-test" and replay.elo == 1500
    assert np.allclose(replay.probs(board, moves), first, atol=1e-12)
    shuffled = moves[::-1]
    assert np.allclose(replay.probs(board, shuffled), first[::-1], atol=1e-12)
    assert replay.probs(board, moves).sum() == pytest.approx(1.0)

    board.push_san("e4")
    with pytest.raises(mt.TeacherError, match="not in the Maia cache"):
        replay.probs(board)


def test_cache_refuses_a_different_teacher_and_a_missing_file(tmp_path):
    path = str(tmp_path / "maia.jsonl")
    cache = mt.CachingTeacher(path, FakeTeacher())
    cache.probs(chess.Board())
    cache.flush()

    other = FakeTeacher()
    other.name = "lc0:maia-1100.pb.gz"
    with pytest.raises(mt.TeacherError, match="written by"):
        mt.CachingTeacher(path, other)
    with pytest.raises(mt.TeacherError, match="not found"):
        mt.CachingTeacher(str(tmp_path / "missing.jsonl"))


def test_cache_file_is_json_lines_with_a_header(tmp_path):
    path = tmp_path / "maia.jsonl"
    cache = mt.CachingTeacher(str(path), FakeTeacher())
    cache.probs(chess.Board())
    cache.close()
    lines = path.read_text().splitlines()
    assert json.loads(lines[0]) == {"cache_version": 1, "teacher": "fake:maia-test", "elo": 1500}
    row = json.loads(lines[1])
    assert set(row) == {"k", "p"} and sum(row["p"].values()) == pytest.approx(1.0)


def test_terminal_position_gives_no_probabilities(tmp_path):
    mate = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert not list(mate.legal_moves)
    live = FakeTeacher()
    cache = mt.CachingTeacher(str(tmp_path / "maia.jsonl"), live)
    assert cache.probs(mate).shape == (0,)
    assert live.calls == 0                                  # nothing to ask, nothing cached


# ---------------------------------------------------------------- the training update
def sample_for(board, move_uci):
    moves = list(board.legal_moves)
    t = [m.uci() for m in moves].index(move_uci)
    F = np.array([echo.features(board, m) for m in moves])
    return moves, (echo.encode(board), np.array([echo.move_index(board, m) for m in moves]), F, t)


def test_step_without_a_teacher_is_unchanged_and_mix_zero_ignores_it():
    board = chess.Board()
    moves, base = sample_for(board, "e2e4")
    q = FakeTeacher().probs(board, moves)
    a, b, c = (echo.StyleNet(hidden=8) for _ in range(3))
    la = a.step(base, 0.1)
    lb = b.step(base + (q,), 0.1)                 # teacher attached but mix = 0
    lc = c.step(base + (q,), 0.1, 0.0)
    assert la == lb == lc
    for k in ("W1", "b1", "W2T", "b2", "w"):
        assert np.array_equal(a.params()[k], b.params()[k]) and np.array_equal(a.params()[k], c.params()[k])


def test_step_gradient_is_softmax_minus_the_mixed_target():
    board = chess.Board()
    moves, base = sample_for(board, "e2e4")
    act, midx, F, t = base
    q = FakeTeacher(favor="d2d4", strength=3.0).probs(board, moves)
    net, lr, mix = echo.StyleNet(hidden=8), 0.05, 0.4
    h = np.maximum(net.W1[act].sum(0) + net.b1, 0)
    p = echo.softmax(net.W2T[midx] @ h + net.b2[midx] + F @ net.w)
    target = mix * q
    target[t] += 1 - mix
    before = net.b2.copy()
    loss = net.step(base + (q,), lr, mix)
    assert np.allclose(-(net.b2[midx] - before[midx]) / lr, p - target, atol=1e-12)
    assert loss == pytest.approx(-np.log(p[t] + 1e-12))     # reported loss stays the plain cross-entropy on the move played


def test_teacher_pulls_probability_toward_what_it_favors():
    board = chess.Board()
    moves, base = sample_for(board, "e2e4")
    ucis = [m.uci() for m in moves]
    q = FakeTeacher(favor="d2d4", strength=50.0).probs(board, moves)     # teacher is nearly certain of d4
    plain, taught = echo.StyleNet(hidden=8), echo.StyleNet(hidden=8)
    for _ in range(60):
        plain.step(base, 0.3)
        taught.step(base + (q,), 0.3, 0.5)
    d4 = ucis.index("d2d4")
    p_plain = echo.softmax(plain.logits(board, moves))
    p_taught = echo.softmax(taught.logits(board, moves))
    assert p_taught[d4] > 0.3 > p_plain[d4]                     # the teacher's move gains weight
    assert p_taught[ucis.index("e2e4")] > p_taught[d4] * 0.5     # but the player's own move still keeps real weight


def test_evaluate_accepts_samples_with_and_without_a_teacher():
    board = chess.Board()
    moves, base = sample_for(board, "e2e4")
    q = FakeTeacher().probs(board, moves)
    net = echo.StyleNet(hidden=8)
    assert net.evaluate([base]) == net.evaluate([base + (q,)])


# ---------------------------------------------------------------- learn() with a teacher
def make_cache_for(pgn, cache_path, player="Me"):
    """Fill a Maia cache with the fake teacher over every position `learn` will ask about."""
    cache = mt.CachingTeacher(str(cache_path), FakeTeacher())
    games, _ = echo.read_games(str(pgn), player)
    for game, me in games:
        board = game.board()
        for move in game.mainline_moves():
            if board.turn == me:
                cache.probs(board)
            board.push(move)
    cache.close()


def train(pgn, out, seed=3, **kw):
    random.seed(seed)
    echo.learn(learn_args(pgn, out, epochs=2, **kw))
    d = np.load(out, allow_pickle=False)
    return {k: d[k] for k in ("W1", "b1", "W2T", "b2", "w")}, json.loads(str(d["meta"]))


def test_learn_without_maia_flags_is_unchanged(tmp_path):
    pgn = tmp_path / "g.pgn"
    write_random_games(pgn, 12, seed=1, plies=24)
    a, meta_a = train(pgn, tmp_path / "a.npz")
    b, meta_b = train(pgn, tmp_path / "b.npz", maia_mix=0.0)         # explicit zero is the same as absent
    assert "teacher" not in meta_a and "teacher" not in meta_b
    for k in a:
        assert np.array_equal(a[k], b[k])


def test_learn_with_a_cached_teacher_records_provenance_and_changes_the_model(tmp_path, capsys):
    pgn = tmp_path / "g.pgn"
    write_random_games(pgn, 12, seed=1, plies=24)
    cache = tmp_path / "maia.jsonl"
    make_cache_for(pgn, cache)
    base, _ = train(pgn, tmp_path / "base.npz")
    taught, meta = train(pgn, tmp_path / "taught.npz", maia_cache=str(cache), maia_mix=0.4)
    assert meta["teacher"] == {"version": 1, "name": "fake:maia-test", "elo": 1500, "mix": 0.4, "games": 12}
    assert meta["profile_version"] == echo.PROFILE_VERSION
    assert not np.array_equal(base["W2T"], taught["W2T"])
    assert "soft-target weight 0.4" in capsys.readouterr().out
    # the teacher record survives resuming without a teacher
    _, resumed = train(pgn, tmp_path / "resumed.npz", resume=str(tmp_path / "taught.npz"))
    assert resumed["teacher"]["mix"] == 0.4


@pytest.mark.parametrize("kwargs,message", [
    (dict(maia_mix=0.3), "needs a teacher"),
    (dict(maia_mix=1.0, maia_cache="x"), "below 1"),
    (dict(maia_mix=0.0, maia_cache="x"), "no effect"),
])
def test_learn_rejects_inconsistent_maia_flags(tmp_path, kwargs, message):
    pgn = tmp_path / "g.pgn"
    write_random_games(pgn, 4, seed=1, plies=10)
    with pytest.raises(SystemExit) as exc:
        echo.learn(learn_args(pgn, tmp_path / "s.npz", **kwargs))
    assert message in str(exc.value)


def test_learn_reports_a_cache_miss_clearly(tmp_path):
    pgn = tmp_path / "g.pgn"
    write_random_games(pgn, 4, seed=1, plies=10)
    other = tmp_path / "other.pgn"
    write_random_games(other, 4, seed=99, plies=10)
    cache = tmp_path / "maia.jsonl"
    make_cache_for(other, cache)                       # a cache for different games
    with pytest.raises(SystemExit) as exc:
        echo.learn(learn_args(pgn, tmp_path / "s.npz", maia_cache=str(cache), maia_mix=0.3))
    assert "not in the Maia cache" in str(exc.value)


def test_taught_model_exports_to_the_same_echo_layout(tmp_path):
    """No tensor or feature changed, so the portable file is the same format version and size as before."""
    pgn = tmp_path / "g.pgn"
    write_random_games(pgn, 12, seed=1, plies=24)
    cache = tmp_path / "maia.jsonl"
    make_cache_for(pgn, cache)
    train(pgn, tmp_path / "base.npz")
    train(pgn, tmp_path / "taught.npz", maia_cache=str(cache), maia_mix=0.4)
    sizes, headers = [], []
    for name in ("base", "taught"):
        net, meta = echo.load_model(str(tmp_path / f"{name}.npz"))
        out = tmp_path / f"{name}.echo"
        echo.export_echo(net, meta, str(out))
        raw = out.read_bytes()
        assert raw[:8] == b"ECHOSTYL" and int.from_bytes(raw[8:12], "little") == 2
        n = int.from_bytes(raw[12:16], "little")
        headers.append(json.loads(raw[16:16 + n]))
        sizes.append(len(raw) - n)                      # tensor bytes only
    assert sizes[0] == sizes[1]
    assert headers[0]["dims"] == headers[1]["dims"] and "teacher" in headers[1] and "teacher" not in headers[0]


# ---------------------------------------------------------------- safety: Maia cannot bypass the band or the gate
class Favors:
    """A teacher that is certain of one move."""
    name = "fake:favors"

    def __init__(self, uci):
        self.uci = uci

    def probs(self, board, moves=None):
        moves = list(board.legal_moves) if moves is None else moves
        return mt._normalize([1.0 if m.uci() == self.uci else 0.0 for m in moves])


def test_maia_blend_cannot_pick_outside_the_safety_band():
    cps = {"e2e4": 50, "d2d4": 45, "a2a3": -200}               # a3 is a blunder the teacher adores
    brain = mt.MaiaBlendBrain(StubBrain({}), Favors("a2a3"), weight=1000.0)
    e = echo.Echo(FakeEngine(cps), brain, "sharp")
    for _ in range(5):
        assert e.choose(chess.Board()).uci() in ("e2e4", "d2d4")


def test_maia_blend_still_goes_through_the_blunder_gate():
    cps = {"e2e4": 50, "d2d4": 45}
    deep = {"d2d4": -300}                                      # looks fine at first, refuted by the deeper search
    brain = mt.MaiaBlendBrain(StubBrain({}), Favors("d2d4"), weight=1000.0)
    e = echo.Echo(FakeEngine(cps, deep), brain, "sharp")
    assert e.choose(chess.Board()).uci() == "e2e4"


def test_maia_blend_cannot_override_a_forced_mate():
    cps = {"d1h5": echo.MATE - 3, "e2e4": 30}
    brain = mt.MaiaBlendBrain(StubBrain({}), Favors("e2e4"), weight=1000.0)
    e = echo.Echo(FakeEngine(cps), brain, "faithful")
    assert e.choose(chess.Board()).uci() == "d1h5"             # band is 0 on a forced mate; the mate stays


def test_blend_with_zero_weight_equals_the_plain_brain_and_never_calls_the_teacher():
    teacher = FakeTeacher()
    stub = StubBrain({"e2e4": 1.5})
    board = chess.Board()
    moves = list(board.legal_moves)
    assert mt.MaiaBlendBrain(stub, teacher, weight=0).scores(board, moves) == stub.scores(board, moves)
    assert teacher.calls == 0


def test_blend_adds_the_log_probability_for_the_moves_it_was_given():
    board = chess.Board()
    given = [chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")]
    teacher = FakeTeacher()
    q = dict(zip((m.uci() for m in board.legal_moves), teacher.probs(board)))
    out = mt.MaiaBlendBrain(StubBrain({"e2e4": 2.0}), teacher, weight=0.5).scores(board, given)
    assert out[0] == pytest.approx(2.0 + 0.5 * np.log(q["e2e4"]))
    assert out[1] == pytest.approx(0.0 + 0.5 * np.log(q["d2d4"]))


# ---------------------------------------------------------------- maia-compare command
def test_maia_compare_prints_a_summary_and_needs_a_teacher(tmp_path, capsys):
    pgn = tmp_path / "g.pgn"
    write_random_games(pgn, 6, seed=2, plies=20)
    cache = tmp_path / "maia.jsonl"
    make_cache_for(pgn, cache)
    echo.maia_compare(Namespace(pgn=str(pgn), player="Me", maia_cache=str(cache), maia_weights=None, lc0=None,
                                style=None, speed=None, examples=3))
    out = capsys.readouterr().out
    assert "Maia comparison for Me" in out and "top move is the one you played" in out
    assert out.count("Maia gave it") == 3
    with pytest.raises(SystemExit, match="needs a Maia teacher"):
        echo.maia_compare(Namespace(pgn=str(pgn), player="Me", maia_cache=None, maia_weights=None, lc0=None,
                                    style=None, speed=None, examples=3))


# ---------------------------------------------------------------- real lc0 + Maia-1 (skipped when not installed)
LC0, WEIGHTS = os.environ.get("ECHOSTAR_LC0"), os.environ.get("ECHOSTAR_MAIA_WEIGHTS")
needs_lc0 = pytest.mark.skipif(not (LC0 and WEIGHTS and os.path.exists(LC0) and os.path.exists(WEIGHTS)),
                               reason="set ECHOSTAR_LC0 and ECHOSTAR_MAIA_WEIGHTS to run against real Maia-1")


@needs_lc0
def test_real_maia1_gives_a_normalized_human_distribution():
    with mt.Lc0Teacher(LC0, WEIGHTS) as t:
        board = chess.Board()
        moves = list(board.legal_moves)
        p = t.probs(board, moves)
        assert p.shape == (20,) and p.sum() == pytest.approx(1.0)
        assert moves[int(np.argmax(p))].uci() in ("e2e4", "d2d4")
        assert p.max() > 0.3                                    # humans concentrate on a few openings
        for fen in ("r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 4 8",   # castling available
                    "8/P6k/8/8/8/8/6K1/8 w - - 0 1",                                           # promotions
                    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"):          # en passant
            b = chess.Board(fen)
            ms = list(b.legal_moves)
            q = t.probs(b, ms)
            assert q.shape == (len(ms),) and q.sum() == pytest.approx(1.0) and q.min() > 0
        promo = chess.Board("8/P6k/8/8/8/8/6K1/8 w - - 0 1")
        ms = list(promo.legal_moves)
        assert ms[int(np.argmax(t.probs(promo, ms)))].uci() == "a7a8q"


@needs_lc0
def test_real_maia1_castling_gets_its_real_probability():
    """Regression: lc0 prints castling as e1h1 while python-chess says e1g1; mapping it wrongly floored every castling move."""
    with mt.Lc0Teacher(LC0, WEIGHTS) as t:
        for fen, castle in (("r1b1k2r/q2nbpp1/pp2pn1p/8/1PPP4/P1NB4/3BNPPP/R2QK2R w KQkq - 1 14", "e1g1"),
                            ("r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R b KQkq - 4 8", "e8g8"),
                            ("r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R b KQkq - 4 8", "e8c8")):
            b = chess.Board(fen)
            moves = list(b.legal_moves)
            p = dict(zip((m.uci() for m in moves), t.probs(b, moves)))
            assert castle in p
            if fen.startswith("r1b1k2r"):
                assert p[castle] > 0.5                 # Maia-1500 gives castling about 66% here; the bug gave 0.0001
        # on any castling-heavy position the full distribution must now be covered by real mass
        b = chess.Board("r1b1k2r/q2nbpp1/pp2pn1p/8/1PPP4/P1NB4/3BNPPP/R2QK2R w KQkq - 1 14")
        assert max(t.probs(b)) > 0.5


def test_castling_spelling_is_translated(tmp_path):
    """lc0 spells castling king-takes-rook; both spellings must resolve to the same legal move (no lc0 needed)."""
    b = chess.Board("r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 4 8")
    castle = chess.Move.from_uci("e1g1")
    assert castle in b.legal_moves and b.uci(castle, chess960=True) == "e1h1"
    parsed = mt.parse_verbose_move_stats(["e1h1  (103 ) N: 0 (+ 0) (P: 66.16%) (WL: -.-----)"])
    assert parsed == {"e1h1": pytest.approx(0.6616)}


@needs_lc0
def test_real_maia1_uses_raw_policy_not_lc0_default_temperature():
    """lc0's default softmax temperature (1.36) flattens the distribution; the teacher must ask for 1.0.
    Entropy of the reference position at temperature 1.0 is 1.64 nats (1.36 gives 2.14), measured with a fresh process."""
    board = chess.Board("rnbqkbnr/ppp2ppp/8/3pp3/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3")
    with mt.Lc0Teacher(LC0, WEIGHTS) as t:
        p = t.probs(board)
    entropy = float(-(p * np.log(p)).sum())
    assert entropy < 1.9, entropy


@needs_lc0
def test_real_maia1_distills_into_learn(tmp_path):
    pgn = tmp_path / "g.pgn"
    write_random_games(pgn, 6, seed=5, plies=16)
    random.seed(0)
    echo.learn(learn_args(pgn, tmp_path / "s.npz", maia_weights=WEIGHTS, lc0=LC0, maia_mix=0.3,
                          maia_cache=str(tmp_path / "c.jsonl")))
    meta = json.loads(str(np.load(tmp_path / "s.npz", allow_pickle=False)["meta"]))
    assert meta["teacher"]["name"] == f"lc0:{os.path.basename(WEIGHTS)}" and meta["teacher"]["elo"] == 1500
    assert os.path.getsize(tmp_path / "c.jsonl") > 100
