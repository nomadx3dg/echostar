"""Chat-style commentary for a game against an Echo bot.

Pure logic, no engine and no I/O: the caller passes the board before the move, the move, and (optionally) engine
evaluations in centipawns from White's point of view. That keeps it easy to test and lets the terminal player and the
web UI share it.

Personas are voices, not people. `fischer` is a bot that plays in the style learned from a player's games; its lines
are written for this program and are not quotes from anyone.
"""
import random

import chess

MATE = 100000
PIECE_NAMES = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
               chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king"}

LINES = {
    "echo": {
        "start_white": ["Let's play. I'll start.", "Good luck. White to move, and that's me."],
        "start_black": ["Good luck. Your move.", "You have the first move. Show me something."],
        "open_e4": ["e4. A good start.", "King's pawn. Classic."],
        "open_d4": ["d4. Let's keep it solid.", "Queen's pawn. I like it."],
        "castle_bot": ["King is safe. Now we can play.", "Castled. That's one worry fewer."],
        "castle_you": ["You castled. Sensible.", "Tucking the king away. Fair enough."],
        "promote_bot": ["A new queen. This is going well.", "Promoted. That should help."],
        "promote_you": ["You got a queen. Well played.", "Promotion. I didn't want that."],
        "mate_bot": ["Checkmate. Good game.", "That's mate. Thanks for playing."],
        "mate_you": ["Checkmate. Well played, you got me.", "That's mate. I'll remember this one."],
        "check_bot": ["Check.", "Check! Mind your king."],
        "check_you": ["Check. Let me think.", "You're giving check. Fine."],
        "queens_off": ["Queens are off. This is an endgame now.", "No more queens. Let's see who knows the endgame."],
        "capture_queen_bot": ["I'll take your queen.", "Your queen is gone."],
        "capture_queen_you": ["You took my queen. Ouch.", "My queen! Well, that hurts."],
        "capture_bot": ["I'll take that {victim}.", "{victim} captured.", "That {victim} was loose."],
        "capture_you": ["You took my {victim}.", "Nice capture.", "I'll miss that {victim}."],
        "blunder_you": ["Thank you. That's a big gift.", "I think that was a mistake.", "That changes things in my favor."],
        "mistake_you": ["Hmm, I like that for me.", "That gives me something to work with."],
        "great_you": ["Strong move. I didn't like that.", "Good one. That's hard to meet."],
        "winning_bot": ["I like my position.", "Things are going my way."],
        "losing_bot": ["You've got me under pressure.", "I'm in trouble here."],
        "draw": ["A draw. Fair result.", "Drawn. Good fight."],
        "resign_you": ["Good game. Want a rematch?", "You resigned. Thanks for playing."],
        "win_you": ["You won. Good game.", "Well played. You outplayed me."],
    },
    "fischer": {
        "start_white": ["White. e4 is the plan.", "I move first. Let's see what you have."],
        "start_black": ["Your move. I'll be waiting.", "Go ahead. I'm ready."],
        "open_e4": ["e4. Simple and direct.", "Pawn to e4. It's where I usually start."],
        "open_d4": ["d4. A little different today.", "d4. We'll see how you handle it."],
        "castle_bot": ["King safe. Now the real work.", "Castled."],
        "castle_you": ["You castled. Good.", "King safe on your side too."],
        "promote_bot": ["Promotion. That decides it.", "Queen. The rest is technique."],
        "promote_you": ["You promoted. I should have stopped that.", "A new queen for you. Careful play."],
        "mate_bot": ["Mate. Good game.", "Checkmate. That's the game."],
        "mate_you": ["Mate. You earned it.", "Checkmate. I missed it."],
        "check_bot": ["Check.", "Check. Your king has to move."],
        "check_you": ["Check. Noted.", "A check. I have an answer."],
        "queens_off": ["Queens off. Endgames suit me.", "No queens. Now it comes down to precision."],
        "capture_queen_bot": ["Queen taken.", "Your queen is gone."],
        "capture_queen_you": ["You took my queen. That's a problem.", "My queen is gone. Now I have to find something."],
        "capture_bot": ["{victim} taken.", "That {victim} was undefended.", "Free {victim}."],
        "capture_you": ["You took my {victim}. Fair.", "Good capture.", "I can live without that {victim}."],
        "blunder_you": ["That was a mistake. I'll use it.", "You gave it away.", "That loses material or worse."],
        "mistake_you": ["Not the best. I'll take the advantage.", "Inaccurate. I'm happy with that."],
        "great_you": ["Good move. That's tough.", "Well found. That's the move I feared."],
        "winning_bot": ["The position is mine.", "I'm better, and I know it."],
        "losing_bot": ["You're doing well. I need a plan.", "I'm worse here. We'll see."],
        "draw": ["A draw. Respectable.", "Drawn. I wanted more."],
        "resign_you": ["Good game. Play again whenever.", "You resigned. A clean result."],
        "win_you": ["You won. That was better chess.", "Well played. I'll study this game."],
    },
}


def swing_for_mover(eval_before, eval_after, mover_is_white):
    """Change in the mover's evaluation (centipawns, positive = the move helped them). None if unknown."""
    if eval_before is None or eval_after is None:
        return None
    clamp = lambda v: max(-1500, min(1500, v))  # noqa: E731  (mate scores must not dominate the swing)
    delta = clamp(eval_after) - clamp(eval_before)
    return delta if mover_is_white else -delta


def mover_eval(eval_white, color_is_white):
    if eval_white is None:
        return None
    return eval_white if color_is_white else -eval_white


class Commentator:
    def __init__(self, persona="echo", rng=None):
        self.lines = LINES.get(persona, LINES["echo"])
        self.persona = persona if persona in LINES else "echo"
        self.rng = rng or random.Random()
        self.last_spoke = -10
        self.mood = None                      # "winning" / "losing" / None, to avoid repeating the same remark

    def _say(self, key, **fmt):
        return self.rng.choice(self.lines[key]).format(**fmt)

    def start(self, bot_is_white):
        return self._say("start_white" if bot_is_white else "start_black")

    def on_move(self, board_before, move, mover_is_bot, eval_before=None, eval_after=None):
        """Comment on `move` (played from `board_before`), or return None to stay quiet."""
        ply = len(board_before.move_stack)
        mover_white = board_before.turn == chess.WHITE
        who = "bot" if mover_is_bot else "you"
        piece = board_before.piece_at(move.from_square)
        victim = board_before.piece_at(move.to_square)
        if board_before.is_en_passant(move):
            victim = chess.Piece(chess.PAWN, not board_before.turn)
        board_after = board_before.copy(stack=False)
        board_after.push(move)

        line = None
        swing = swing_for_mover(eval_before, eval_after, mover_white)
        if board_after.is_checkmate():
            line = self._say(f"mate_{who}")
        elif move.promotion:
            line = self._say(f"promote_{who}")
        elif not mover_is_bot and swing is not None and swing <= -300:
            line = self._say("blunder_you")
        elif not mover_is_bot and swing is not None and swing <= -120:
            line = self._say("mistake_you")
        elif not mover_is_bot and swing is not None and swing >= 150 and (victim or board_after.is_check()):
            line = self._say("great_you")
        elif board_before.is_castling(move):
            line = self._say(f"castle_{who}")
        elif victim and victim.piece_type == chess.QUEEN:
            line = self._say(f"capture_queen_{who}")
        elif victim and not _queens(board_after) and _queens(board_before):
            line = self._say("queens_off")
        elif victim and (piece is None or PIECE_NAMES[victim.piece_type] != "pawn" or ply > 12):
            line = self._say(f"capture_{who}", victim=PIECE_NAMES[victim.piece_type])
        elif board_after.is_check():
            line = self._say(f"check_{who}")
        elif ply == 0 and piece and board_before.san(move) == "e4":
            line = self._say("open_e4")
        elif ply == 0 and piece and board_before.san(move) == "d4":
            line = self._say("open_d4")
        elif mover_is_bot and ply - self.last_spoke >= 6:
            line = self._mood_line(mover_eval(eval_after, mover_white))
        if line:
            self.last_spoke = ply
        return line

    def _mood_line(self, bot_eval):
        if bot_eval is None:
            return None
        mood = "winning" if bot_eval >= 200 else "losing" if bot_eval <= -200 else None
        if mood == self.mood:
            return None
        self.mood = mood
        return self._say(f"{mood}_bot") if mood else None

    def on_result(self, result, bot_is_white, resigned=False):
        """Closing line for a finished game. `result` is '1-0', '0-1' or '1/2-1/2'."""
        if result == "1/2-1/2":
            return self._say("draw")
        bot_won = (result == "1-0") == bot_is_white
        if resigned and not bot_won:
            return self._say("resign_you")
        return self._say("mate_bot" if bot_won else "win_you") if not resigned else self._say("resign_you")


def _queens(board):
    return bool(board.pieces(chess.QUEEN, chess.WHITE) or board.pieces(chess.QUEEN, chess.BLACK))
