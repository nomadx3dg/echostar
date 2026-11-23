import random

import chess

import commentary
from commentary import LINES, Commentator


def line_in(text, persona, key):
    return text in LINES[persona][key]


def push_all(sans):
    board = chess.Board()
    for san in sans[:-1]:
        board.push_san(san)
    return board, board.parse_san(sans[-1])


def test_swing_clamps_mate_scores_and_signs_by_mover():
    assert commentary.swing_for_mover(0, -400, True) == -400
    assert commentary.swing_for_mover(0, -400, False) == 400
    assert commentary.swing_for_mover(0, commentary.MATE, True) == 1500      # not 100000
    assert commentary.swing_for_mover(None, 10, True) is None


def test_mate_line_for_each_side():
    board, mv = push_all(["f3", "e5", "g4", "Qh4#"])
    talk = Commentator("echo", random.Random(1))
    assert line_in(talk.on_move(board, mv, True), "echo", "mate_bot")
    assert line_in(Commentator("echo", random.Random(1)).on_move(board, mv, False), "echo", "mate_you")


def test_human_blunder_and_mistake_need_an_evaluation():
    board, mv = push_all(["e4", "e5", "Qh5"])
    talk = Commentator("echo", random.Random(2))
    assert line_in(talk.on_move(board, mv, False, 20, -380), "echo", "blunder_you")
    assert line_in(Commentator("echo", random.Random(2)).on_move(board, mv, False, 20, -180), "echo", "mistake_you")
    assert Commentator("echo", random.Random(2)).on_move(board, mv, False) is None   # no eval, nothing notable


def test_blunder_judged_from_blacks_point_of_view():
    board, mv = push_all(["e4", "e5", "Nf3", "Nf6", "Nxe5", "Nxe4"])
    talk = Commentator("echo", random.Random(3))
    # Black moved; White's eval rising by 400 is a 400 cp loss for Black
    assert line_in(talk.on_move(board, mv, False, 0, 400), "echo", "blunder_you")


def test_capture_castle_check_and_queen_lines():
    talk = Commentator("fischer", random.Random(4))
    board, mv = push_all(["e4", "d5", "exd5"])
    # a pawn capture this early is not remarked on; a piece capture is
    assert talk.on_move(board, mv, True) is None
    board, mv = push_all(["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Bxc6"])
    assert Commentator("fischer", random.Random(4)).on_move(board, mv, True) in \
        [s.format(victim="knight") for s in LINES["fischer"]["capture_bot"]]
    board, mv = push_all(["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "O-O"])
    assert line_in(Commentator("echo", random.Random(4)).on_move(board, mv, True), "echo", "castle_bot")
    board, mv = push_all(["e4", "e5", "Qh5", "Nc6", "Qxf7+"])
    # an early pawn capture is not remarked on, but the check it gives is
    assert line_in(Commentator("echo", random.Random(4)).on_move(board, mv, False), "echo", "check_you")


def test_opening_lines_and_start():
    board, mv = push_all(["e4"])
    assert line_in(Commentator("fischer", random.Random(5)).on_move(board, mv, True), "fischer", "open_e4")
    board, mv = push_all(["d4"])
    assert line_in(Commentator("echo", random.Random(5)).on_move(board, mv, True), "echo", "open_d4")
    assert Commentator("echo", random.Random(5)).start(True) in LINES["echo"]["start_white"]
    assert Commentator("echo", random.Random(5)).start(False) in LINES["echo"]["start_black"]


def test_mood_remarks_do_not_repeat_and_are_spaced_out():
    talk = Commentator("echo", random.Random(6))
    board = chess.Board()
    for san in ["Nf3", "Nf6", "Nc3", "Nc6", "g3", "g6", "Bg2", "Bg7"]:
        board.push_san(san)
    mv = board.parse_san("h3")                       # nothing notable, and the bot is clearly winning
    first = talk.on_move(board, mv, True, 250, 260)
    assert line_in(first, "echo", "winning_bot")
    assert talk._mood_line(260) is None              # same mood: stay quiet
    assert talk.on_move(board, mv, True, 250, 260) is None   # also too soon after the last remark


def test_results_and_unknown_persona_fall_back():
    talk = Commentator("nobody", random.Random(7))
    assert talk.persona == "echo"
    assert talk.on_result("1/2-1/2", True) in LINES["echo"]["draw"]
    assert talk.on_result("0-1", True, resigned=False) in LINES["echo"]["win_you"]
    assert talk.on_result("1-0", True, resigned=True) in LINES["echo"]["resign_you"]
    assert talk.on_result("1-0", True) in LINES["echo"]["mate_bot"]


def test_every_persona_has_every_line_key():
    keys = set(LINES["echo"])
    for persona, table in LINES.items():
        assert set(table) == keys, persona
        for key, lines in table.items():
            assert lines and all(isinstance(x, str) and x for x in lines), (persona, key)


def test_play_chat_prints_commentary_in_the_terminal(monkeypatch, capsys, tmp_path):
    """`play --chat` with a fake engine and scripted input: the bot speaks at the start and after a notable move."""
    from argparse import Namespace

    from test_ui_server import MaterialEngine

    import echo

    class Engine(MaterialEngine):
        options = {}

        def configure(self, _opts):
            pass

        def quit(self):
            pass

    monkeypatch.setattr(echo, "find_stockfish", lambda _p: "fake")
    monkeypatch.setattr(echo.chess.engine.SimpleEngine, "popen_uci", staticmethod(lambda _path: Engine()))
    moves = iter(["e4", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(moves))
    args = Namespace(style=None, stockfish=None, threads=None, hash=None, seed=3, dial="balanced", band=None, temp=None,
                     depth=1, color="white", movetime=None, show=False, plain=True, no_save=True,
                     save_dir=str(tmp_path), chat=True)
    echo.play(args)
    out = capsys.readouterr().out
    assert 'Echo: "' in out
    spoken = [ln for ln in out.splitlines() if ln.startswith('Echo: "')]
    assert spoken[0][7:-1] in LINES["echo"]["start_black"]          # you are White, so the bot is Black
    assert any(ln[7:-1] in LINES["echo"]["open_e4"] for ln in spoken)
