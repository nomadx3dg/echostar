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
        [s.format(victim="knight", san="Bxc6", square="c6") for s in LINES["fischer"]["capture_bot"]]
    board, mv = push_all(["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "O-O"])
    assert line_in(Commentator("echo", random.Random(4)).on_move(board, mv, True), "echo", "castle_bot")
    board, mv = push_all(["e4", "d6", "Bb5+"])
    assert Commentator("echo", random.Random(4)).on_move(board, mv, False) in \
        [t.format(san="Bb5", square="b5") for t in LINES["echo"]["check_you"]]


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


def play(sans):
    board = chess.Board()
    for san in sans:
        board.push_san(san)
    return board


def feed(talk, sans, bot_white=True, evals=None):
    """Play SANs through the commentator; the bot moves on its colour. Returns [(ply, line)] for the moves that spoke."""
    board, out = chess.Board(), []
    for i, san in enumerate(sans):
        mv = board.parse_san(san)
        e0, e1 = (evals.get(i, (None, None)) if evals else (None, None))
        line = talk.on_move(board, mv, (i % 2 == 0) == bot_white, e0, e1)
        board.push(mv)
        if line:
            out.append((i, line))
    return out


def test_second_human_blunder_is_called_out_differently():
    board, mv = push_all(["e4", "e5", "Qh5"])
    talk = Commentator("echo", random.Random(5))
    first = talk.on_move(board, mv, False, 20, -380)
    second = talk.on_move(board, mv, False, 20, -380)
    assert first in LINES["echo"]["blunder_you"] and second in LINES["echo"]["blunder_again"]


def test_bot_admits_its_own_blunder_and_celebrates_a_find():
    board, mv = push_all(["e4", "e5", "Qh5"])
    assert line_in(Commentator("echo", random.Random(6)).on_move(board, mv, True, 20, -300), "echo", "oops_bot")
    board, mv = push_all(["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Bxc6"])
    line = Commentator("echo", random.Random(6)).on_move(board, mv, True, 0, 250)
    assert line in LINES["echo"]["found_bot"]


def test_capture_and_check_lines_can_name_the_move_and_square():
    board, mv = push_all(["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Bxc6"])
    seen = {Commentator("echo", random.Random(i)).on_move(board, mv, True) for i in range(40)}
    assert any("Bxc6" in text and "c6" in text for text in seen)


def test_material_change_is_announced_once_on_a_quiet_move():
    board = chess.Board("r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")     # Black is a knight down
    talk = Commentator("echo", random.Random(7))
    first = talk.on_move(board, board.parse_san("Nf3"), True)
    assert first in LINES["echo"]["ahead_bot"]
    board.push_san("Nf3")
    board.push_san("d5")
    assert talk.on_move(board, board.parse_san("d4"), True) is None                   # no repeat while nothing changed


def test_material_is_judged_from_the_bots_side():
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/R1BQKBNR w KQkq - 0 1")     # White, the human, is a knight down
    talk = Commentator("echo", random.Random(7))
    assert talk.on_move(board, board.parse_san("Nf3"), False) in LINES["echo"]["ahead_bot"]      # so the bot is ahead
    board = chess.Board("r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")     # Black is a knight down
    talk = Commentator("echo", random.Random(7))
    assert talk.on_move(board, board.parse_san("Nf3"), False) in LINES["echo"]["behind_bot"]


def test_opening_is_named_at_most_twice_and_only_once_per_line():
    talk = Commentator("echo", random.Random(8))
    board = chess.Board()
    names = []
    for i, san in enumerate(["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6"]):
        mv = board.parse_san(san)
        line = talk.on_move(board, mv, i % 2 == 0, opening="Sicilian Defense" if i >= 3 else None)
        board.push(mv)
        if line and "Sicilian Defense" in line:
            names.append(i)
    assert len(names) == 1


def test_endgame_remark_comes_once():
    board = chess.Board("r3k3/4pp2/8/8/8/8/4PP2/R3K3 w - - 0 1")          # level material, four pieces' worth of rooks
    talk = Commentator("echo", random.Random(9))
    first = talk.on_move(board, board.parse_san("Rd1"), True)
    assert first in LINES["echo"]["endgame"]
    board.push_san("Rd1")
    board.push_san("Rd8")
    talk.last_spoke = -10
    assert talk.on_move(board, board.parse_san("Rc1"), True) not in LINES["echo"]["endgame"]


def test_comeback_and_slipping_moods():
    talk = Commentator("fischer", random.Random(10))
    board = chess.Board()
    mv = board.parse_san("Nf3")
    assert talk._mood_line(-300) in LINES["fischer"]["losing_bot"]
    assert talk._mood_line(0) in LINES["fischer"]["recover_bot"]
    assert talk._mood_line(300) in LINES["fischer"]["winning_bot"]
    assert talk._mood_line(0) in LINES["fischer"]["slipping_bot"]
    assert mv


def test_quiet_stretch_gets_a_phase_remark_not_before_move_four():
    talk = Commentator("echo", random.Random(11))
    out = feed(talk, ["Nf3", "Nf6", "g3", "g6", "Bg2", "Bg7", "O-O", "O-O", "d3", "d6"])
    assert all(i >= 6 for i, _ in out if _ not in LINES["echo"]["castle_bot"] + LINES["echo"]["castle_you"])


def test_new_situations_exist_for_both_personas():
    for persona in ("echo", "fischer"):
        for key in ("blunder_again", "oops_bot", "found_bot", "sac_bot", "sac_you", "ahead_bot", "behind_bot", "level_bot",
                    "endgame", "opening_name", "long_game", "recover_bot", "slipping_bot", "quiet_open", "quiet_middle",
                    "quiet_end"):
            assert len(LINES[persona][key]) >= 2, (persona, key)


def test_same_line_is_not_repeated_back_to_back():
    for persona in commentary.LINES:
        talk = commentary.Commentator(persona, random.Random(3))
        said = [talk._say("check_bot", san="Qh5", square="h5") for _ in range(60)]
        assert all(a != b for a, b in zip(said, said[1:]))


def test_every_situation_has_several_lines():
    for persona, table in commentary.LINES.items():
        assert all(len(lines) >= 3 for lines in table.values()), persona
