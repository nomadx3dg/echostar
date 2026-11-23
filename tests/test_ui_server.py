import http.client
import json
import threading
from http.server import ThreadingHTTPServer

import chess
import chess.engine
import pytest

import echo
import ui_server

VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
CODE = "reykjavik72"


def material(board):
    return sum(v * (len(board.pieces(p, chess.WHITE)) - len(board.pieces(p, chess.BLACK))) for p, v in VALUES.items())


class MaterialEngine:
    """Stands in for Stockfish: scores a position by material, one ply deep. Good enough to drive the app."""
    def analyse(self, board, limit, multipv=None):
        rows = []
        for mv in list(board.legal_moves):
            board.push(mv)
            white_cp = material(board) * 100
            board.pop()
            rows.append((mv, white_cp))
        rows.sort(key=lambda r: -r[1] if board.turn == chess.WHITE else r[1])
        score = lambda cp: chess.engine.PovScore(chess.engine.Cp(cp), chess.WHITE)  # noqa: E731
        if multipv:
            return [{"pv": [mv], "score": score(cp)} for mv, cp in rows[:multipv]]
        return {"score": score(material(board) * 100)}


class Brain:
    def __init__(self, player="Tester"):
        self.player = player

    def scores(self, board, moves):
        return [0.0] * len(moves)


def make_app(tmp_path, fischer_available=True, save=False):
    engine = MaterialEngine()
    bots = {
        "echo": ui_server.Bot("echo", "Echo", "style bot", "echo", Brain("Tester"), "balanced"),
        "fischer": ui_server.Bot("fischer", "Fischer", "style model", "fischer", Brain("Fischer, Robert James"),
                                 "sharp", available=fischer_available, note="Train it first."),
    }
    return ui_server.GameApp(engine, lambda bot, dial: echo.Echo(engine, bot.brain, dial, depth=1), bots,
                             save_dir=str(tmp_path), save=save, seed=1, save_fn=echo.save_game)


def test_unlock_word_matching():
    assert ui_server.unlock_matches(CODE)
    assert ui_server.unlock_matches("zzz REYKJAVIK-72")           # case, punctuation and leading noise are ignored
    assert ui_server.unlock_matches("hello" + CODE)
    assert not ui_server.unlock_matches("reykjavik")
    assert not ui_server.unlock_matches("reykjavik73")
    assert not ui_server.unlock_matches(CODE + "x")                # must be the end of what was typed
    assert not ui_server.unlock_matches("")


def test_source_does_not_contain_the_unlock_word():
    with open(ui_server.__file__, encoding="utf-8") as fh:
        assert CODE not in fh.read().lower()


def test_new_game_and_move_flow(tmp_path):
    app = make_app(tmp_path)
    st = app.state()
    assert st["turn"] == "w" and "e2e4" in st["legal"] and not st["pending"]
    assert [b["id"] for b in st["bots"]] == ["echo"]               # the extra opponent is hidden
    st = app.move("e2e4")
    assert st["pending"] and st["legal"] == [] and st["san"] == ["e4"]
    st = app.reply()
    assert not st["pending"] and len(st["san"]) == 2 and st["legal"]
    assert app.reply()["san"] == st["san"]                        # a repeated reply() does not play twice
    assert st["chat"][0]["ply"] == 0 and st["chat"][0]["text"]


def test_rejects_bad_moves_and_wrong_turn(tmp_path):
    app = make_app(tmp_path)
    for bad in ("e2e5", "zzzz", "", "e7e5"):
        with pytest.raises(ValueError):
            app.move(bad)
    app.move("e2e4")
    with pytest.raises(ValueError):                                # bot's turn
        app.move("d2d4")


def test_undo_takes_back_a_full_move_and_trims_chat(tmp_path):
    app = make_app(tmp_path)
    with pytest.raises(ValueError):
        app.undo()
    app.move("e2e4")
    app.reply()
    st = app.undo()
    assert st["san"] == [] and st["legal"] and all(c["ply"] == 0 for c in st["chat"])


def test_bot_plays_first_when_human_is_black(tmp_path):
    app = make_app(tmp_path)
    st = app.new_game("echo", "black")
    assert st["human"] == "b" and st["pending"]
    st = app.reply()
    assert len(st["san"]) == 1 and not st["pending"] and st["legal"]


def test_resign_ends_game_and_saves_pgn_with_profile_name(tmp_path):
    app = make_app(tmp_path, save=True)
    app.move("e2e4")
    app.reply()
    st = app.resign()
    assert st["result"] == "0-1" and st["resigned"] and st["saved"]
    text = open(st["saved"]).read()
    assert '[White "Tester"]' in text and '[Black "Echo"]' in text
    with pytest.raises(ValueError):
        app.move("d2d4")


def test_unlock_reveals_fischer_and_its_games_are_not_labelled_as_his(tmp_path):
    app = make_app(tmp_path, save=True)
    with pytest.raises(ValueError):
        app.new_game("fischer")                                    # locked
    assert app.unlock("nothing to see")["unlocked"] is False
    assert app.unlock(CODE)["unlocked"] is True
    assert [b["id"] for b in app.bot_list()] == ["echo", "fischer"]
    st = app.new_game("fischer", "white")
    assert st["bot"]["persona"] == "fischer" and st["dial"] == "sharp"
    app.move("e2e4")
    app.reply()
    text = open(app.resign()["saved"]).read()
    assert '[White "You"]' in text and '[Black "Fischer"]' in text
    assert "Fischer, Robert James" not in text          # these are not Fischer's games


def test_unavailable_fischer_model_gives_a_helpful_error(tmp_path):
    app = make_app(tmp_path, fischer_available=False)
    app.unlock(CODE)
    with pytest.raises(ValueError, match="Train it first"):
        app.new_game("fischer")


def test_checkmate_sets_result_and_ends_the_turn(tmp_path):
    app = make_app(tmp_path)
    app.new_game("echo", "white")
    for san in ["f3", "e5"]:
        app.board.push_san(san)
    app.evals = [0] * (len(app.board.move_stack) + 1)
    app.move("g2g4")
    app.board.push_san("Qh4#")                          # Black mates
    assert app._finished() and app.result == "0-1"
    st = app.state()
    assert st["result"] == "0-1" and st["legal"] == [] and not st["pending"]


@pytest.fixture
def server(tmp_path):
    app = make_app(tmp_path)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), ui_server.make_handler(app))
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv.server_address[1], app
    srv.shutdown()
    srv.server_close()


def call(port, method, path, body=None, headers=None, raw=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    hdrs = {"Content-Type": "application/json"} if method == "POST" else {}
    hdrs.update(headers or {})
    payload = raw if raw is not None else (json.dumps(body) if body is not None else None)
    conn.request(method, path, body=payload, headers=hdrs)
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, data


def test_http_page_and_api(server):
    port, _ = server
    status, page = call(port, "GET", "/")
    assert status == 200 and b"<title>Echo" in page
    status, data = call(port, "POST", "/api/move", {"uci": "e2e4"})
    assert status == 200 and json.loads(data)["pending"]
    status, data = call(port, "POST", "/api/reply", {})
    assert status == 200 and len(json.loads(data)["san"]) == 2
    status, data = call(port, "POST", "/api/move", {"uci": "a1a8"})
    assert status == 400 and "error" in json.loads(data)
    assert call(port, "GET", "/nope")[0] == 404
    assert call(port, "POST", "/api/nope", {})[0] == 404


def test_http_refuses_other_sites_wrong_types_and_big_bodies(server):
    port, _ = server
    assert call(port, "POST", "/api/new", {}, headers={"Origin": "http://evil.example"})[0] == 403
    assert call(port, "POST", "/api/new", {}, headers={"Origin": f"http://127.0.0.1:{port}"})[0] == 200
    assert call(port, "POST", "/api/new", {}, headers={"Content-Type": "text/plain"})[0] == 415
    assert call(port, "POST", "/api/new", raw="x" * 10000)[0] == 413
    assert call(port, "POST", "/api/new", raw="not json")[0] == 400
    assert call(port, "POST", "/api/new", raw="[1,2]")[0] == 400


def test_http_unlock_roundtrip(server):
    port, _ = server
    _, data = call(port, "POST", "/api/unlock", {"typed": "abc"})
    assert json.loads(data)["unlocked"] is False
    _, data = call(port, "POST", "/api/unlock", {"typed": CODE})
    out = json.loads(data)
    assert out["unlocked"] and [b["id"] for b in out["bots"]] == ["echo", "fischer"]
