import chess
import pytest

import openings
import review
from review import classify, is_sacrifice, win_pct


@pytest.fixture(scope="module")
def book():
    return openings.Openings()


def board_after(*sans):
    b = chess.Board()
    for s in sans:
        b.push_san(s)
    return b


def judge(sans, move, before_cp, after_cp, best=None, second=None, book_after=False):
    b = board_after(*sans)
    mv = b.parse_san(move)
    return classify(b, mv, {"cp": before_cp, "best": best, "second": second}, {"cp": after_cp}, book_after)


def test_win_pct_is_symmetric_monotonic_and_clamped():
    assert win_pct(0) == pytest.approx(50)
    assert win_pct(200) > win_pct(100) > 50 > win_pct(-100) > win_pct(-200)
    assert win_pct(300) + win_pct(-300) == pytest.approx(100)
    assert win_pct(100000) == win_pct(1500)                               # mate scores are clamped
    assert win_pct(0) < win_pct(100000) < 100


def test_loss_ladder_for_white():
    base = dict(sans=["e4", "e5"], move="Nf3", before_cp=0)
    assert judge(**base, after_cp=-10)["label"] == "excellent"
    assert judge(**base, after_cp=-40)["label"] == "good"
    assert judge(**base, after_cp=-90)["label"] == "inaccuracy"
    assert judge(**base, after_cp=-150)["label"] == "mistake"
    assert judge(**base, after_cp=-200)["label"] == "blunder"            # two pawns is already a blunder on Lichess's curve
    assert judge(**base, after_cp=-600)["label"] == "blunder"


def test_loss_is_measured_from_blacks_side():
    # Black moves; White's evaluation rising from 0 to +600 is a blunder for Black
    assert judge(["e4"], "e5", 0, 600)["label"] == "blunder"
    assert judge(["e4"], "e5", 0, -10)["label"] in ("excellent", "good", "best")


def test_playing_the_engine_move_is_best_even_if_scores_wobble():
    out = judge(["e4", "e5"], "Nf3", 0, -150, best="g1f3")             # noise between two searches
    assert out["label"] == "best" and out["loss"] == 0


def test_bad_moves_carry_the_better_move_in_words():
    out = judge(["e4", "e5"], "Qh5", 0, -300, best="g1f3")
    assert out["label"] in review.BAD and out["best"] == "g1f3" and out["best_san"] == "Nf3"


def test_forced_and_book_take_priority():
    b = chess.Board("k7/8/8/8/8/8/1q6/K7 w - - 0 1")                   # Kxb2 is the only legal move
    assert b.legal_moves.count() == 1
    assert classify(b, chess.Move.from_uci("a1b2"), {"cp": -900}, {"cp": -900})["label"] == "forced"
    assert judge(["e4", "e5"], "Nf3", 0, -5, book_after=True)["label"] == "book"
    # a book move that loses a lot is not praised
    assert judge(["e4", "e5"], "Qh5", 0, -500, book_after=True)["label"] == "blunder"


def test_great_is_an_only_move():
    out = judge(["e4", "e5"], "Nf3", 40, 40, best="g1f3", second=-300)
    assert out["label"] == "great"
    assert judge(["e4", "e5"], "Nf3", 40, 40, best="g1f3", second=20)["label"] == "best"
    assert judge(["e4", "e5"], "Nf3", 40, 40, best="g1f3", second=None)["label"] == "best"
    # not "great" to find the only move when the game is already decided
    assert judge(["e4", "e5"], "Nf3", 900, 900, best="g1f3", second=100)["label"] == "best"


def test_sacrifice_detection():
    # Bxf7+ gives a bishop for a pawn
    b = board_after("e4", "e5", "Bc4", "Nc6")
    assert is_sacrifice(b, b.parse_san("Bxf7+"))
    # an even trade is not a sacrifice
    b = board_after("e4", "e5", "Nf3", "Nc6", "Bb5", "a6")
    assert not is_sacrifice(b, b.parse_san("Bxc6"))
    # a quiet developing move is not
    assert not is_sacrifice(chess.Board(), chess.Move.from_uci("g1f3"))
    # kings and pawns are never "sacrificed" by this test
    b = board_after("e4", "d5")
    assert not is_sacrifice(b, b.parse_san("exd5"))


def test_brilliant_needs_sacrifice_best_play_and_a_sound_position():
    sans, move = ["e4", "e5", "Bc4", "Nc6"], "Bxf7+"
    assert judge(sans, move, 20, 30, best="c4f7")["label"] == "brilliant"
    assert judge(sans, move, 20, -400, best="c4f7")["label"] == "best"      # loses the plot afterwards: not brilliant
    # a sacrifice that holds the evaluation level is as good as the engine's move, even if that was a different one
    assert judge(sans, move, 20, 25, best="g1f3")["label"] == "brilliant"
    assert judge(sans, move, 20, -60, best="g1f3")["label"] != "brilliant"      # but not if it costs something
    assert judge(sans, move, 800, 800, best="c4f7")["label"] == "best"      # already crushing: not brilliant


def test_opening_names_and_book_positions(book):
    b = board_after("e4", "e5", "Nf3", "Nc6", "Bb5")
    assert book.lookup(b) == ("C60", "Ruy Lopez") and book.is_book(b)
    b = board_after("e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6")
    assert book.lookup(b) == ("B90", "Sicilian Defense: Najdorf Variation")
    assert not book.is_book(board_after("a4", "h5", "Ra3", "Rh6", "Rg3", "Rg6"))
    assert book.lookup(board_after("e4")) is not None
    assert len(book) > 3000


def test_book_is_found_through_transpositions(book):
    a = board_after("d4", "d5", "c4", "e6", "Nc3", "Nf6")
    b = board_after("c4", "e6", "Nc3", "d5", "d4", "Nf6")
    assert a.epd() == b.epd() and book.is_book(a) and book.is_book(b)


def test_missing_openings_file_is_not_fatal(tmp_path):
    assert openings.load(str(tmp_path / "nope.tsv")) is None
