import os
import random
import sys
from argparse import Namespace

import chess
import chess.engine
import chess.pgn
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import echo  # noqa: E402


# ---------------------------------------------------------------- encoding / features
def test_encode_is_perspective_symmetric():
    board = chess.Board()
    for san in ("e4", "e5", "Nf3"):
        board.push_san(san)                       # black to move
    mirrored = board.mirror()                     # same position, white to move
    assert sorted(echo.encode(board)) == sorted(echo.encode(mirrored))


def test_move_index_is_perspective_symmetric():
    black = chess.Board()
    black.push_san("e4")
    white = black.mirror()
    assert echo.move_index(black, chess.Move.from_uci("e7e5")) == \
        echo.move_index(white, chess.Move.from_uci("e2e4"))


def test_features_castle_and_trade():
    board = chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
    f = echo.features(board, chess.Move.from_uci("e1g1"))
    assert f[echo.FEATURES.index("castle")] == 1

    board = chess.Board()
    for san in ("e4", "d5"):
        board.push_san(san)
    f = echo.features(board, board.parse_san("exd5"))
    assert f[echo.FEATURES.index("capture")] == 1
    assert f[echo.FEATURES.index("even_or_winning_trade")] == 1


def test_softmax_sums_to_one():
    assert echo.softmax(np.array([1.0, 2.0, 3.0])).sum() == pytest.approx(1.0)


# ---------------------------------------------------------------- PGN helpers
@pytest.mark.parametrize("tc,expected", [("60+0", "bullet"), ("180+0", "blitz"), ("600+5", "rapid"),
                                         ("1800+0", "classical"), ("-", None), (None, None)])
def test_classify_speed(tc, expected):
    assert echo.classify_speed(tc) == expected


def write_random_games(path, n, seed=0, plies=40):
    rng = random.Random(seed)
    with open(path, "w") as fh:
        for i in range(n):
            board, game = chess.Board(), chess.pgn.Game()
            game.headers.update({"White": "Me", "Black": "Opp", "TimeControl": "600+0", "BlackElo": "1500"})
            node = game
            for _ in range(plies):
                if board.is_game_over():
                    break
                move = rng.choice(list(board.legal_moves))
                board.push(move)
                node = node.add_variation(move)
            print(game, file=fh, end="\n\n")


def learn_args(pgn, out, **kw):
    base = dict(pgn=str(pgn), player="Me", out=str(out), epochs=1, lr=0.02,
                speed=None, min_opp_rating=None, resume=None)
    base.update(kw)
    return Namespace(**base)


def test_read_games_filters(tmp_path):
    pgn = tmp_path / "g.pgn"
    write_random_games(pgn, 3)
    assert len(echo.read_games(pgn, "Me")[0]) == 3
    assert len(echo.read_games(pgn, "me")[0]) == 3                      # case-insensitive
    assert len(echo.read_games(pgn, "Someone")[0]) == 0
    assert echo.read_games(pgn, "Me", speeds={"rapid"})[0].__len__() == 3
    games, skipped = echo.read_games(pgn, "Me", speeds={"bullet"})
    assert games == [] and skipped == 3
    assert echo.read_games(pgn, "Me", min_opp_rating=1600)[0] == []


def test_learn_roundtrip_and_resume(tmp_path):
    pgn = tmp_path / "g.pgn"
    write_random_games(pgn, 12)
    out = tmp_path / "style.npz"
    echo.learn(learn_args(pgn, out))
    net, meta = echo.load_model(out)
    assert meta["games"] == 12 and meta["validation"] is not None
    assert 0 <= meta["validation"]["top1"] <= 1

    resumed = tmp_path / "style2.npz"
    echo.learn(learn_args(pgn, resumed, resume=str(out)))
    _, meta2 = echo.load_model(resumed)
    assert meta2["games"] == 24 and meta2["positions"] == 2 * meta["positions"]

    with pytest.raises(SystemExit):
        echo.learn(learn_args(pgn, resumed, player="Opp", resume=str(out)))

    brain = echo.StyleBrain(str(out))
    board = chess.Board()
    moves = list(board.legal_moves)
    assert len(brain.scores(board, moves)) == len(moves)


def test_repertoire_and_export(tmp_path):
    pgn = tmp_path / "g.pgn"
    write_random_games(pgn, 5)
    out = tmp_path / "s.npz"
    echo.learn(learn_args(pgn, out))
    meta = echo.load_model(out)[1]
    assert echo.repertoire(meta)[0][0] == "As White, first move"
    md = echo.export_markdown(meta)
    assert "# Style profile: Me" in md and "| Move type |" in md


# ---------------------------------------------------------------- board rendering
def test_render_orientation_and_plain_mode():
    board = chess.Board()
    white = echo.render(board, chess.WHITE, color=False).splitlines()
    black = echo.render(board, chess.BLACK, color=False).splitlines()
    assert white[0].startswith(" 8 ") and black[0].startswith(" 1 ")
    assert "a  b  c" in white[8] and "h  g  f" in black[8]
    assert "Material: even" in white[-1]


# ---------------------------------------------------------------- chooser logic (no Stockfish needed)
class FakeEngine:
    """Returns canned evals: `cps` maps first-move UCI -> centipawns (mover's view); `deep` overrides verify."""
    def __init__(self, cps, deep=None):
        self.cps, self.deep = cps, deep or {}

    def analyse(self, board, limit, multipv=None):
        if multipv:
            ranked = sorted(self.cps.items(), key=lambda kv: -kv[1])
            return [{"pv": [chess.Move.from_uci(u)], "score": chess.engine.PovScore(chess.engine.Cp(cp), board.turn)}
                    for u, cp in ranked]
        uci = board.peek().uci()   # verify(): position after the candidate move
        cp = self.deep.get(uci, self.cps[uci])
        return {"score": chess.engine.PovScore(chess.engine.Cp(-cp), board.turn)}


class StubBrain:
    def __init__(self, prefs):
        self.prefs = prefs

    def scores(self, board, moves):
        return [self.prefs.get(m.uci(), 0.0) for m in moves]


def make_echo(cps, prefs, dial="sharp", deep=None):
    return echo.Echo(FakeEngine(cps, deep), StubBrain(prefs), dial)


def test_style_picks_among_safe_moves_only():
    cps = {"e2e4": 50, "d2d4": 45, "a2a3": -200}
    bot = make_echo(cps, {"d2d4": 5.0, "a2a3": 9.0})        # style loves a3, but a3 is a blunder
    assert bot.choose(chess.Board()).uci() == "d2d4"


def test_blunder_gate_falls_back_when_deep_search_disagrees():
    cps = {"e2e4": 50, "d2d4": 45}
    bot = make_echo(cps, {"d2d4": 5.0}, deep={"d2d4": -300})  # looks fine shallow, loses deeper
    assert bot.choose(chess.Board()).uci() == "e2e4"


def test_forced_mate_is_always_taken():
    cps = {"e2e4": echo.MATE - 1, "d2d4": 100}
    bot = make_echo(cps, {"d2d4": 9.0}, dial="faithful")
    assert bot.choose(chess.Board()).uci() == "e2e4"


def test_summary_reports_cost():
    bot = make_echo({"e2e4": 50, "d2d4": 45}, {"d2d4": 5.0})
    bot.choose(chess.Board())
    assert "0/1" in bot.summary() or "1/1" in bot.summary()


def test_save_game_roundtrip(tmp_path):
    board = chess.Board()
    board.push_san("e4")
    path = echo.save_game(board, "Me", "Echo", str(tmp_path))
    with open(path) as fh:
        game = chess.pgn.read_game(fh)
    assert game.headers["White"] == "Me" and game.headers["Black"] == "Echo"
    assert [m.uci() for m in game.mainline_moves()] == ["e2e4"]
