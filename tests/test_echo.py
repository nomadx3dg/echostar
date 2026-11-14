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


# ---------------------------------------------------------------- polish additions
def test_custom_dial_dict_and_movetime():
    bot = echo.Echo(FakeEngine({"e2e4": 50, "d2d4": 45}), StubBrain({"d2d4": 5.0}), {"band": 20, "temp": 0.0})
    assert bot.choose(chess.Board(), movetime=1.0).uci() == "d2d4"


def test_gate_checks_are_capped():
    # five safe moves, style ranks the engine's best last and every deep check fails -> at most 3 verifications
    cps = {"e2e4": 50, "d2d4": 49, "c2c4": 48, "g1f3": 47, "b1c3": 46}
    deep = {u: -500 for u in cps if u != "e2e4"}
    calls = []

    class CountingEngine(FakeEngine):
        def analyse(self, board, limit, multipv=None):
            if not multipv:
                calls.append(board.peek().uci())
            return super().analyse(board, limit, multipv)

    bot = echo.Echo(CountingEngine(cps, deep), StubBrain({"d2d4": 5, "c2c4": 4, "g1f3": 3, "b1c3": 2}),
                    {"band": 100, "temp": 0.0})
    assert bot.choose(chess.Board()).uci() == "e2e4"
    assert len(calls) == echo.MAX_GATE_CHECKS


def test_resign_result_is_saved(tmp_path):
    board = chess.Board()
    board.push_san("e4")
    path = echo.save_game(board, "Me", "Echo", str(tmp_path), result="0-1")
    with open(path) as fh:
        assert chess.pgn.read_game(fh).headers["Result"] == "0-1"


def test_export_echo_format(tmp_path):
    import json
    import struct
    net = echo.StyleNet(hidden=8)
    meta = {"player": "Me", "games": 1, "positions": 1, "book": {}}
    path = tmp_path / "m.echo"
    echo.export_echo(net, meta, str(path))
    data = path.read_bytes()
    assert data[:8] == echo.FORMAT_MAGIC
    version, json_len = struct.unpack("<II", data[8:16])
    header = json.loads(data[16:16 + json_len])
    assert version == 2 and header["dims"]["hidden"] == 8 and header["dims"]["moves"] == echo.N_MOVES
    floats = (768 * 8) + 8 + (echo.N_MOVES * 8) + echo.N_MOVES + len(echo.FEATURES)
    assert len(data) == 16 + json_len + 4 * floats


def test_export_legacy_v1_layout(tmp_path):
    import json
    import struct
    net = echo.StyleNet(hidden=8)
    path = tmp_path / "old.echo"
    echo.export_echo(net, {"player": "Me", "games": 1, "positions": 1, "book": {}}, str(path), version=1)
    data = path.read_bytes()
    version, json_len = struct.unpack("<II", data[8:16])
    assert version == 1 and json.loads(data[16:16 + json_len])["dims"]["moves"] == 4096
    floats = (768 * 8) + 8 + (4096 * 8) + 4096 + len(echo.FEATURES)
    assert len(data) == 16 + json_len + 4 * floats


# ---------------------------------------------------------------- downloading games (mock servers, no internet)
@pytest.mark.parametrize("tc,expected", [("600", "rapid"), ("180", "blitz"), ("1/86400", "classical"), ("180+2", "blitz")])
def test_classify_speed_chesscom_formats(tc, expected):
    assert echo.classify_speed(tc) == expected


@pytest.fixture
def mock_server():
    import http.server
    import threading

    routes = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = routes.get(self.path)
            if body is None:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body.encode())

        def log_message(self, *a):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", routes
    server.shutdown()


GAME = '[Event "x"]\n[White "alice"]\n[Black "bob"]\n[Result "1-0"]\n\n1. e4 e5 1-0\n\n'


def test_fetch_lichess_builds_query(mock_server):
    base, routes = mock_server
    routes["/api/games/user/alice?max=5&tags=true&clocks=false&evals=false&opening=false&perfType=blitz,rapid&rated=true"] = GAME
    text = echo.fetch_lichess("alice", 5, ["blitz", "rapid"], True, base=base)
    assert text.count("[Event ") == 1


def test_fetch_chesscom_walks_archives_newest_first(mock_server):
    import json
    base, routes = mock_server
    routes["/pub/player/bob/games/archives"] = json.dumps({"archives": [f"{base}/m/2026/09", f"{base}/m/2026/10"]})
    routes["/m/2026/09/pgn"] = GAME.replace("alice", "bob_old")
    routes["/m/2026/10/pgn"] = GAME + GAME
    text = echo.fetch_chesscom("bob", 2, base=base)
    assert text.count("[Event ") == 2 and "bob_old" not in text      # stopped after the newest month


def numbered(n, tc="600"):
    return "".join(f'[Event "g{i}"]\n[White "bob"]\n[Black "x"]\n[TimeControl "{tc}"]\n[Result "1-0"]\n\n1. e4 e5 1-0\n\n'
                   for i in range(n))


def test_chesscom_returns_exactly_max_newest_first(mock_server):
    import json
    base, routes = mock_server
    routes["/pub/player/bob/games/archives"] = json.dumps({"archives": [f"{base}/m/1", f"{base}/m/2"]})
    routes["/m/1/pgn"] = numbered(5)                 # older month
    routes["/m/2/pgn"] = numbered(4)                 # newer month; its last game (g3) is the newest
    games = echo.split_games(echo.fetch_chesscom("bob", 6, base=base))
    assert len(games) == 6
    assert 'Event "g3"' in games[0] and 'Event "g0"' in games[3]      # newest month, reversed
    assert 'Event "g4"' in games[4]                                   # then the older month, newest first


def test_chesscom_speed_filter(mock_server):
    import json
    base, routes = mock_server
    routes["/pub/player/bob/games/archives"] = json.dumps({"archives": [f"{base}/m/1"]})
    routes["/m/1/pgn"] = numbered(3, "60") + numbered(2, "600")
    games = echo.split_games(echo.fetch_chesscom("bob", 10, {"rapid"}, base=base))
    assert len(games) == 2 and all(echo.game_speed(g) == "rapid" for g in games)


def test_unknown_user_is_a_clear_error(mock_server):
    base, _ = mock_server
    with pytest.raises(SystemExit) as exc:
        echo.fetch_lichess("nobody", 5, base=base)
    assert "Not found" in str(exc.value)


def test_fetch_cmd_writes_file(mock_server, tmp_path, monkeypatch):
    base, routes = mock_server
    routes["/api/games/user/alice?max=3&tags=true&clocks=false&evals=false&opening=false"] = GAME
    monkeypatch.setattr(echo, "LICHESS_URL", base)
    monkeypatch.setattr(echo, "fetch_lichess",
                        lambda *a, **k: echo.http_get(f"{base}/api/games/user/alice?max=3&tags=true&clocks=false&evals=false&opening=false"))
    out = tmp_path / "g.pgn"
    echo.fetch_cmd(Namespace(user="alice", site="lichess", max=3, speed=None, rated=False, out=str(out)))
    assert out.read_text().count("[Event ") == 1


# ---------------------------------------------------------------- robustness
def test_no_engine_candidates_falls_back_to_a_legal_move():
    bot = make_echo({}, {})                         # the engine returns no lines at all
    board = chess.Board()
    assert bot.choose(board) in board.legal_moves
    assert bot.history == [(False, 0)]


def test_verify_skips_the_engine_when_the_move_mates():
    # Fool's mate: after 1.f3 e5 2.g4 black plays Qh4#
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2")

    class NeverCalled:
        def analyse(self, *args, **kwargs):
            raise AssertionError("the engine should not be asked about a checkmate")

    bot = echo.Echo(NeverCalled(), StubBrain({}), "sharp")
    assert bot.verify(board, chess.Move.from_uci("d8h4")) == echo.MATE
    assert not board.move_stack                     # the board is left as it was


def test_learn_with_no_player_moves_is_a_clear_error(tmp_path):
    pgn = tmp_path / "empty.pgn"
    pgn.write_text('[White "Me"]\n[Black "Opp"]\n[Result "*"]\n\n*\n\n')
    with pytest.raises(SystemExit) as exc:
        echo.learn(learn_args(pgn, tmp_path / "s.npz"))
    assert "none contain a move" in str(exc.value)


def play_args(tmp_path, **kw):
    base = dict(style=None, stockfish=None, threads=None, hash=None, seed=None, dial="balanced", band=None,
                temp=None, depth=14, color="white", show=False, plain=True, movetime=None, no_save=True,
                save_dir=str(tmp_path))
    base.update(kw)
    return Namespace(**base)


class DummyEngine:
    options = {}

    def __init__(self):
        self.quit_called = False

    def configure(self, opts):
        pass

    def quit(self):
        self.quit_called = True


def test_play_stops_cleanly_on_eof_and_reports_nothing_to_undo(tmp_path, monkeypatch, capsys):
    engine = DummyEngine()
    monkeypatch.setattr(echo, "find_stockfish", lambda path: "stockfish")
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", classmethod(lambda cls, path: engine))
    inputs = iter(["undo"])

    def fake_input(prompt=""):
        try:
            return next(inputs)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr("builtins.input", fake_input)
    echo.play(play_args(tmp_path))                  # must return normally, not raise EOFError
    assert "Nothing to undo yet." in capsys.readouterr().out
    assert engine.quit_called


@pytest.mark.parametrize("fetch,args", [(echo.fetch_lichess, ("a b/c", 1)), (echo.fetch_chesscom, ("a b/c", 1))])
def test_usernames_are_url_quoted(mock_server, fetch, args):
    import json
    base, routes = mock_server
    routes["/api/games/user/a%20b%2Fc?max=1&tags=true&clocks=false&evals=false&opening=false"] = GAME
    routes["/pub/player/a%20b%2Fc/games/archives"] = json.dumps({"archives": [f"{base}/m/1"]})
    routes["/m/1/pgn"] = GAME
    assert fetch(*args, base=base).count("[Event ") == 1


# ---------------------------------------------------------------- promotion-aware moves (profile v2)
PROMO_FENS = [
    "4k3/P6P/8/8/8/8/p6p/4K3 w - - 0 1",             # straight promotions, both sides
    "4k3/P6P/8/8/8/8/p6p/4K3 b - - 0 1",
    "1n2k3/P7/8/8/8/8/7p/4K3 w - - 0 1",             # white promotes by capturing as well
    "4k3/8/8/8/8/8/7p/4K1R1 b - - 0 1",              # black promotes by capturing too
]


def test_promotions_get_distinct_units_and_queen_keeps_its_v1_unit():
    board = chess.Board(PROMO_FENS[0])
    promos = [board.parse_uci("a7a8" + p) for p in "qrbn"]
    idx = [echo.move_index(board, m) for m in promos]
    assert len(set(idx)) == 4
    assert idx[0] == chess.A7 * 64 + chess.A8                # the queen promotion is the unchanged v1 unit
    assert all(i < echo.N_MOVES for i in idx) and all(i >= echo.N_BASE_MOVES for i in idx[1:])


@pytest.mark.parametrize("fen", PROMO_FENS + [chess.STARTING_FEN, "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"])
def test_move_index_is_injective_over_legal_moves(fen):
    board = chess.Board(fen)
    moves = list(board.legal_moves)
    assert len({echo.move_index(board, m) for m in moves}) == len(moves)


def test_promotion_mirroring_agrees_between_colors():
    white = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    black = white.mirror()                                    # same position, colors swapped
    for p in "qrbn":
        a = echo.move_index(white, white.parse_uci("a7a8" + p))
        b = echo.move_index(black, black.parse_uci("a2a1" + p))
        assert a == b


def promo_sample(board, move):
    moves = list(board.legal_moves)
    F = np.array([echo.features(board, m) for m in moves])
    return (echo.encode(board), np.array([echo.move_index(board, m) for m in moves]), F, moves.index(move)), moves


def test_model_can_learn_to_prefer_a_knight_promotion():
    board = chess.Board("4k3/P7/8/8/8/8/8/3K4 w - - 0 1")   # Na8 and Ba8 are identical quiet moves in v1
    net = echo.StyleNet(hidden=8)
    sample, moves = promo_sample(board, board.parse_uci("a7a8n"))
    for _ in range(60):
        net.step(sample, 0.3)
    p = echo.softmax(net.logits(board, moves))
    by_uci = {m.uci(): p[i] for i, m in enumerate(moves)}
    assert by_uci["a7a8n"] > 0.5 and by_uci["a7a8n"] > by_uci["a7a8b"]


def v1_file(tmp_path, net, **meta_extra):
    """Write a profile exactly as v1 did: 4096 move units and no profile_version field."""
    import json
    meta = {"player": "Me", "games": 3, "positions": 30, "played_rate": {n: 0.1 for n in echo.FEATURES},
            "baseline_rate": {n: 0.1 for n in echo.FEATURES}, "book": {}, "validation": None}
    meta.update(meta_extra)
    path = tmp_path / "v1.npz"
    np.savez_compressed(path, meta=json.dumps(meta), W1=net.W1, b1=net.b1, W2T=net.W2T[:4096], b2=net.b2[:4096],
                        w=net.w)
    return path


def v1_logits(net, board, moves):
    """The v1 scoring formula, written out independently of echo.StyleNet."""
    h = np.maximum(net.W1[echo.encode(board)].sum(0) + net.b1, 0)
    out = []
    for m in moves:
        f, t = m.from_square, m.to_square
        if board.turn == chess.BLACK:
            f, t = chess.square_mirror(f), chess.square_mirror(t)
        i = f * 64 + t
        out.append(net.W2T[i] @ h + net.b2[i] + echo.features(board, m) @ net.w)
    return np.array(out)


def test_v1_profile_migrates_without_changing_any_score(tmp_path, capsys):
    rng = np.random.default_rng(5)
    net = echo.StyleNet(rng=rng, hidden=8)
    net.W2T[:4096] = rng.normal(0, 0.5, (4096, 8)); net.b2[:4096] = rng.normal(0, 0.3, 4096)
    net.W1 = rng.normal(0, 0.5, net.W1.shape); net.w = rng.normal(0, 0.8, net.w.shape)
    path = v1_file(tmp_path, net)

    migrated, meta = echo.load_model(path)
    assert "v1 profile" in capsys.readouterr().err                 # never silent
    assert meta["migrated_from"] == 1 and meta["profile_version"] == 2
    assert migrated.W2T.shape[0] == echo.N_MOVES
    assert np.allclose(migrated.W2T[:4096], net.W2T[:4096])

    for fen in PROMO_FENS + [chess.STARTING_FEN, "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1"]:
        board = chess.Board(fen)
        moves = list(board.legal_moves)
        assert np.allclose(migrated.logits(board, moves), v1_logits(net, board, moves)), fen


def test_future_profile_version_is_refused(tmp_path):
    path = v1_file(tmp_path, echo.StyleNet(hidden=8), profile_version=99)
    with pytest.raises(SystemExit) as exc:
        echo.load_model(path)
    assert "99" in str(exc.value)


def test_resume_from_v1_profile_writes_v2(tmp_path):
    pgn = tmp_path / "g.pgn"
    write_random_games(pgn, 12)
    path = v1_file(tmp_path, echo.StyleNet(hidden=8), games=12, positions=100)
    out = tmp_path / "v2.npz"
    echo.learn(learn_args(pgn, out, resume=str(path)))
    net, meta = echo.load_model(out)                                # no migration note: already v2
    assert "migrated_from" not in meta and meta["profile_version"] == 2 and net.W2T.shape[0] == echo.N_MOVES
