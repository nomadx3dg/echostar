"""Behavior dimensions measured from the moves a player actually chose.

Nothing here is a hand-set personality score. Each dimension is a yes/no property of a move (its `signal`), looked at
only in the positions and among the moves where that property is meaningful (its `applies` and `scope`). At every
position the player's choice is compared with a uniformly random choice among the legal moves in scope:

    observed  O = signal(chosen move)
    expected  E = mean of signal over the scope moves
    variance  V = variance of signal over the scope moves

Summed over positions, z = (O - E) / sqrt(V) is a standardized effect: it is about 0 for a player who picks uniformly
at random, and positive (negative) for a player who picks the property more (less) often than chance would.

Limits, stated up front:
  * The reference is a random legal move, not a typical player. Any sensible player scores below random on "risk".
    Compare two profiles with each other, not one profile with zero.
  * The signals are move-local proxies (see each description). They cannot see recaptures or long tactics.
  * Context (material, king safety) is not used yet, so one number per dimension is an average over all contexts.

This module depends only on python-chess so `echo.py` can import it.
"""
import math

import chess

VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
ATTACKER_VALUE = {**VALUE, chess.KING: 100}           # a king can only capture an undefended piece
CENTER = {chess.D4, chess.E4, chess.D5, chess.E5}
ENDGAME_MATERIAL = 24      # both sides' non-pawn material at or below this (the start position has 62)
MIN_POSITIONS = 50         # fewer informative positions than this: no verdict
SUMMARY_VERSION = 1


def _rel_rank(color, square):
    rank = chess.square_rank(square)
    return rank if color == chess.WHITE else 7 - rank


def _forcing(board, move):
    return board.is_capture(move) or board.gives_check(move)


def _en_prise_after(board, move):
    """The moved piece can be taken for free, or by a cheaper piece, in the position after the move."""
    color = board.turn
    board.push(move)
    try:
        to = move.to_square
        moved = board.piece_at(to)
        attackers = board.attackers(not color, to)
        if not attackers or moved is None:
            return 0.0
        defended = board.is_attacked_by(color, to)
        cheapest = min(ATTACKER_VALUE[board.piece_type_at(s)] for s in attackers)
        return float((not defended) or cheapest < VALUE[moved.piece_type])
    finally:
        board.pop()


def _creates_threat(board, move):
    """After the move, the moved piece newly attacks an enemy piece that is undefended or worth more than itself."""
    color = board.turn
    enemy = board.occupied_co[not color]
    before = board.attacks_mask(move.from_square) & enemy
    board.push(move)
    try:
        to = move.to_square
        moved = board.piece_at(to)
        newly = (board.attacks_mask(to) & enemy) & ~before
        for sq in chess.scan_forward(newly):
            victim = board.piece_at(sq)
            if victim.piece_type == chess.KING:
                continue
            if VALUE[victim.piece_type] > VALUE[moved.piece_type] or not board.is_attacked_by(not color, sq):
                return 1.0
        return 0.0
    finally:
        board.pop()


def _equal_piece_capture(board, move):
    """A piece takes a piece of the same value (N or B for N or B, R for R, Q for Q)."""
    if not board.is_capture(move) or board.is_en_passant(move):
        return 0.0
    mover = board.piece_at(move.from_square).piece_type
    victim = board.piece_at(move.to_square).piece_type
    return float(mover != chess.PAWN and VALUE[mover] == VALUE[victim])


def _space_pawn_advance(board, move):
    piece = board.piece_at(move.from_square)
    return float(piece.piece_type == chess.PAWN and not board.is_capture(move)
                 and _rel_rank(board.turn, move.to_square) >= 3)


def _classical_development(board, move):
    piece = board.piece_at(move.from_square)
    back_rank = 0 if board.turn == chess.WHITE else 7
    develops = piece.piece_type in (chess.KNIGHT, chess.BISHOP) and chess.square_rank(move.from_square) == back_rank
    return float(board.is_castling(move) or develops or move.to_square in CENTER)


def _non_pawn_material(board):
    return sum(VALUE[p.piece_type] for p in board.piece_map().values() if p.piece_type not in (chess.PAWN, chess.KING))


def _is_endgame(board):
    return _non_pawn_material(board) <= ENDGAME_MATERIAL


def _king_dist(square):
    return max(abs(chess.square_file(square) - 3.5), abs(chess.square_rank(square) - 3.5))


def _centralizes_king(board, move):
    piece = board.piece_at(move.from_square)
    return float(piece.piece_type == chess.KING and not board.is_castling(move)
                 and _king_dist(move.to_square) < _king_dist(move.from_square))


def _always(board):
    return True


def _any_move(board, move):
    return True


def _quiet_non_king(board, move):
    return not board.is_capture(move) and board.piece_at(move.from_square).piece_type != chess.KING


def _non_capture(board, move):
    return not board.is_capture(move)


def _non_forcing(board, move):
    return not _forcing(board, move)


class Dimension:
    def __init__(self, name, label, description, signal, scope=_any_move, applies=_always):
        self.name, self.label, self.description = name, label, description
        self.signal, self.scope, self.applies = signal, scope, applies


DIMENSIONS = [
    Dimension("aggression", "Aggression",
              "plays a capture or a check (forcing moves)",
              _forcing),
    Dimension("risk_tolerance", "Risk tolerance",
              "makes a non-capturing move after which the piece can be taken for free or by a cheaper piece",
              _en_prise_after, scope=_quiet_non_king),
    Dimension("tactical_appetite", "Tactical appetite",
              "makes a non-capturing move that newly attacks an enemy piece that is undefended or worth more",
              _creates_threat, scope=_non_capture),
    Dimension("simplification", "Simplification preference",
              "captures a piece of equal value (piece for piece), the usual way to trade down",
              _equal_piece_capture),
    Dimension("initiative", "Initiative preference",
              "advances a pawn to its 4th rank or beyond without capturing (space gaining; a proxy)",
              _space_pawn_advance),
    Dimension("positional", "Positional preference",
              "among quiet moves: castles, develops a minor piece from the back rank, or occupies d4/e4/d5/e5",
              _classical_development, scope=_non_forcing),
    Dimension("endgame_activity", "Endgame king activity",
              "in endgames (non-pawn material <= 24): moves the king closer to the centre",
              _centralizes_king, applies=_is_endgame),
]
NAMES = [d.name for d in DIMENSIONS]
_BY_NAME = {d.name: d for d in DIMENSIONS}


# --------------------------------------------------------------------------- measuring
def new_tally():
    """Per dimension, two halves (games alternate between them) of [positions, observed, expected, variance]."""
    return {n: [[0, 0.0, 0.0, 0.0], [0, 0.0, 0.0, 0.0]] for n in NAMES}


def observe(board, played, moves, tally, half=0):
    """Add one decision (the player chose `played` among the legal `moves`) to `tally`."""
    for dim in DIMENSIONS:
        if not dim.applies(board):
            continue
        scoped = [m for m in moves if dim.scope(board, m)]
        if len(scoped) < 2 or played not in scoped:
            continue
        signals = [dim.signal(board, m) for m in scoped]
        mean = sum(signals) / len(signals)
        var = sum((s - mean) ** 2 for s in signals) / len(signals)
        if var <= 1e-12:
            continue                      # every in-scope move has the same property: this choice says nothing
        cell = tally[dim.name][half]
        cell[0] += 1
        cell[1] += signals[scoped.index(played)]
        cell[2] += mean
        cell[3] += var


def merge_tally(a, b):
    out = new_tally()
    for n in NAMES:
        for h in (0, 1):
            out[n][h] = [x + y for x, y in zip(a[n][h], b[n][h])]
    return out


def _z(cell):
    n, observed, expected, variance = cell
    return (observed - expected) / math.sqrt(variance) if variance > 0 else 0.0


def verdict(n, z, z_halves):
    """A plain-language reading. Thresholds are deliberately strict: there are 7 dimensions, so |z| >= 3 is required
    for 'clear', and both halves of the games (alternate games) must point the same way as the whole."""
    if n < MIN_POSITIONS:
        return "too little data"
    consistent = all(h * z > 0 for h in z_halves)
    direction = "more" if z > 0 else "less"
    if abs(z) >= 3 and consistent:
        return f"clear: {direction} than random"
    if abs(z) >= 2 and consistent:
        return f"weak: {direction} than random"
    return "not distinguishable from random"


def summarize(tally):
    """name -> {n, observed_rate, expected_rate, z, z_halves, verdict}; rates are over informative positions."""
    out = {}
    for n in NAMES:
        total = [sum(tally[n][h][i] for h in (0, 1)) for i in range(4)]
        halves = [_z(tally[n][0]), _z(tally[n][1])]
        z = _z(total)
        count = int(total[0])
        out[n] = {"n": count,
                  "observed_rate": total[1] / count if count else 0.0,
                  "expected_rate": total[2] / count if count else 0.0,
                  "z": z, "z_halves": halves, "verdict": verdict(count, z, halves)}
    return out


def to_meta(tally, games):
    return {"version": SUMMARY_VERSION, "games": games, "tally": tally}


def from_meta(block):
    """Rebuild a tally from profile metadata. Returns (tally, games) or (None, 0) when absent or unreadable."""
    if not block or block.get("version") != SUMMARY_VERSION:
        return None, 0
    tally = new_tally()
    for n in NAMES:
        if n in block.get("tally", {}):
            tally[n] = [list(map(float, block["tally"][n][h])) for h in (0, 1)]
            for h in (0, 1):
                tally[n][h][0] = int(tally[n][h][0])
    return tally, int(block.get("games", 0))


def signal_of(name, board, move):
    """The raw yes/no property of one move, for tests and the validation script."""
    return _BY_NAME[name].signal(board, move)


def in_scope(name, board, move):
    dim = _BY_NAME[name]
    return dim.applies(board) and dim.scope(board, move)
