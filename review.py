"""Move signs for practice mode: brilliant, great, best, excellent, good, book, forced, inaccuracy, mistake, blunder.

Pure logic. The caller supplies engine evaluations (centipawns from White's point of view) for the position before and
after the move; nothing here talks to an engine.

How a move is judged: the *loss* is how many points of winning chance the move gave up compared with the engine's best
move, using Lichess's win-percentage curve. Blunder/mistake/inaccuracy use Lichess's own cut-offs (15, 10 and 5 points).
Brilliant and great are judged on top of that and are deliberately conservative, since both are easy to over-award:

  brilliant  a near-best move that gives up a piece for less than it takes (a sacrifice) and keeps a sound position;
  great      the best move when the next-best move is much worse (an only move).

"Book" means the resulting position lies on a named opening line in data/openings.tsv. It is not an engine verdict.
"""
import math

import chess

VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}

# Loss thresholds in win-percentage points (Lichess: 0.1 / 0.2 / 0.3 on its -1..1 scale = 5 / 10 / 15 points).
EXCELLENT, GOOD, INACCURACY, MISTAKE = 2.0, 5.0, 10.0, 15.0
ONLY_MOVE_GAP = 15.0        # win-percentage points between the best and the second-best move
BAD = ("inaccuracy", "mistake", "blunder")


def win_pct(cp):
    """Lichess's win-percentage curve for a centipawn score from the side's own point of view."""
    cp = max(-1500, min(1500, cp))
    return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * cp)) - 1)


def is_sacrifice(board_before, move):
    """True if the move puts a piece (a knight or more) where it can be taken for less than the move itself wins."""
    piece = board_before.piece_at(move.from_square)
    if piece is None or piece.piece_type == chess.KING:
        return False
    victim = board_before.piece_at(move.to_square)
    if board_before.is_en_passant(move):
        victim = chess.Piece(chess.PAWN, not piece.color)
    gain = VALUES.get(victim.piece_type, 0) if victim else 0
    after = board_before.copy(stack=False)
    after.push(move)
    moved = after.piece_at(move.to_square)
    if moved is None or VALUES.get(moved.piece_type, 0) < 3:
        return False
    attackers = after.attackers(not piece.color, move.to_square)
    if not attackers:
        return False
    cheapest = min(VALUES.get(after.piece_type_at(sq), 100) for sq in attackers)   # a king attacker counts as costly
    value = VALUES[moved.piece_type]
    if not after.attackers(piece.color, move.to_square):
        risk = value                       # nobody defends it
    elif cheapest < value:
        risk = value - cheapest            # it can be taken by something cheaper, even if recaptured
    else:
        return False
    return risk - gain >= 2


def classify(board_before, move, before, after, book_after=False):
    """Judge `move`, played from `board_before`.

    `before` and `after` are dicts with "cp" (White's view, centipawns); `before` may also have "best" (the engine's
    best move, UCI) and "second" (White's-view score of the second-best move). Returns {"label", "loss", "best",
    "best_san"}; "best" and "best_san" are None when unknown.
    """
    white = board_before.turn == chess.WHITE
    pov = (lambda cp: cp) if white else (lambda cp: -cp)
    w_before, w_after = win_pct(pov(before["cp"])), win_pct(pov(after["cp"]))
    best = before.get("best")
    played_best = best == move.uci()
    loss = 0.0 if played_best else max(0.0, w_before - w_after)
    best_san = None
    if best and not played_best:
        try:
            best_san = board_before.san(chess.Move.from_uci(best))
        except ValueError:
            best_san = None

    only_move = False
    if played_best and before.get("second") is not None:
        only_move = w_before - win_pct(pov(before["second"])) >= ONLY_MOVE_GAP and w_before < 90

    if board_before.legal_moves.count() == 1:
        label = "forced"
    elif loss < GOOD and book_after:
        label = "book"
    elif loss < EXCELLENT and is_sacrifice(board_before, move) and w_after >= 40 and w_before <= 85:
        label = "brilliant"
    elif only_move:
        label = "great"
    elif played_best:
        label = "best"
    elif loss < EXCELLENT:
        label = "excellent"
    elif loss < GOOD:
        label = "good"
    elif loss < INACCURACY:
        label = "inaccuracy"
    elif loss < MISTAKE:
        label = "mistake"
    else:
        label = "blunder"
    return {"label": label, "loss": round(loss, 1), "best": best, "best_san": best_san}
