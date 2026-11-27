"""Chat-style commentary for a game against an Echo bot.

Pure logic, no engine and no I/O: the caller passes the board before the move, the move, and (optionally) engine
evaluations in centipawns from White's point of view. That keeps it easy to test and lets the terminal player and the
web UI share it.

Personas are voices, not people. `fischer` is a bot that plays in the style learned from a player's games; its lines
are written for this program and are not quotes from anyone.
"""
import random

import chess

import review

MATE = 100000
PIECE_NAMES = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
               chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king"}

LINES = {
    "echo": {
        "start_white": ["Let's play. I'll start.", "Good luck. White to move, and that's me.",
            "White for me. Let's see what you do with it.", "I'll open. Take your time."],
        "start_black": ["Good luck. Your move.", "You have the first move. Show me something.",
            "Black for me. Take the first move.", "Over to you. Make it count."],
        "open_e4": ["e4. A good start.", "King's pawn. Classic.", "e4. Open lines, open play.",
            "Pawn to e4. Let's go."],
        "open_d4": ["d4. Let's keep it solid.", "Queen's pawn. I like it.", "d4. Slow and steady.",
            "The queen's pawn. A fine start."],
        "castle_bot": ["King is safe. Now we can play.", "Castled. That's one worry fewer.",
            "Castled. Time to get the rooks involved.", "King tucked away."],
        "castle_you": ["You castled. Sensible.", "Tucking the king away. Fair enough.",
            "You're castled. Now the middlegame starts.", "Castling. I expected that."],
        "promote_bot": ["A new queen. This is going well.", "Promoted. That should help.",
            "Promoted. That changes the picture.", "A fresh queen on the board."],
        "promote_you": ["You got a queen. Well played.", "Promotion. I didn't want that.",
            "You queened. I have to find something now.", "Promotion. Congratulations."],
        "mate_bot": ["Checkmate. Good game.", "That's mate. Thanks for playing.", "Checkmate. Thanks for the game.",
            "That's mate. Shall we go again?"],
        "mate_you": ["Checkmate. Well played, you got me.", "That's mate. I'll remember this one.",
            "Checkmate. That was well done.", "You got me. Nicely played."],
        "check_bot": ["Check.", "Check! Mind your king.", "Check. Your king has to answer.", "Check, careful now.",
            "{san}. Your king has to move."],
        "check_you": ["Check. Let me think.", "You're giving check. Fine.", "Check. I see it.",
            "A check. Let me find the reply.", "{san}. Check. I have an answer."],
        "queens_off": ["Queens are off. This is an endgame now.", "No more queens. Let's see who knows the endgame.",
            "Queens traded. Now it's about technique.", "The queens are gone. Quieter from here."],
        "capture_queen_bot": ["I'll take your queen.", "Your queen is gone.", "Your queen is mine.",
            "I've got the queen."],
        "capture_queen_you": ["You took my queen. Ouch.", "My queen! Well, that hurts.", "There goes my queen.",
            "You won my queen. Now I'm in trouble."],
        "capture_bot": ["I'll take that {victim}.", "{victim} captured.", "That {victim} was loose.",
            "Picking up the {victim}.", "Your {victim} is gone.", "That {victim} had no support.",
            "{san}. That {victim} on {square} was loose."],
        "capture_you": ["You took my {victim}.", "Nice capture.", "I'll miss that {victim}.",
            "Your {victim} for mine. Fine.", "You took my {victim}. I'll see about that.",
            "Good eye on that {victim}.", "{san}. You took the {victim} on {square}."],
        "blunder_you": ["Thank you. That's a big gift.", "I think that was a mistake.",
            "That changes things in my favor.", "I'll take that gladly.", "That one hurts you more than me.",
            "I think you missed something there."],
        "mistake_you": ["Hmm, I like that for me.", "That gives me something to work with.",
            "That helps me a little.", "Not your best. I'll press."],
        "great_you": ["Strong move. I didn't like that.", "Good one. That's hard to meet.", "That was very good.",
            "Sharp move. I have work to do."],
        "winning_bot": ["I like my position.", "Things are going my way.", "I'm pleased with this position.",
            "The pressure is on your side of the board now."],
        "losing_bot": ["You've got me under pressure.", "I'm in trouble here.", "This is getting hard for me.",
            "I need to find counterplay."],
        "draw": ["A draw. Fair result.", "Drawn. Good fight.", "A draw. We both held on.", "Even game, even result."],
        "resign_you": ["Good game. Want a rematch?", "You resigned. Thanks for playing.",
            "Fair enough. Another game?", "Good game. I'm here whenever you want a rematch."],
        "win_you": ["You won. Good game.", "Well played. You outplayed me.", "You won. I'll learn from this one.",
            "Well done. You deserved it."],
        "blunder_again": ["That's twice now. Slow down a little.", "Another one. Take a breath before you move.",
            "This is becoming a pattern. Check what your move leaves loose."],
        "oops_bot": ["Oops. I may have misjudged that.", "That wasn't my best. I'll have to fix it.",
            "I think I let something slip there."],
        "found_bot": ["That was the move I was hoping for.", "Now that's what I was looking for.",
            "Yes, that's the idea."],
        "sac_bot": ["A sacrifice. Let's see if it pays off.", "I'm giving something up here, on purpose.",
            "Material for activity. Let's find out."],
        "sac_you": ["A sacrifice? Interesting. I'll have to check it.", "You're offering material. Is it a trap?",
            "Bold. I'll have to be careful here."],
        "ahead_bot": ["I'm up material now.", "I've won some material. I'll keep it simple.",
            "I'm ahead now. I'll trade down carefully."],
        "behind_bot": ["I'm down material. I need activity now.", "You're ahead on material. I need counterplay.",
            "I'm behind. Time to make it complicated."],
        "level_bot": ["Material is level again.", "We're even on material.", "Back to equal material."],
        "endgame": ["Few pieces left. Every move counts now.", "This is an endgame. Kings can fight now.",
            "We've reached the endgame. Let's be exact."],
        "opening_name": ["That's the {name}. A familiar road.", "The {name}. I know this one.",
            "Ah, the {name}. Nice."],
        "long_game": ["Twenty moves and still a fight.", "We're deep into this one now.",
            "This is turning into a long game."],
        "recover_bot": ["I'm back in the game.", "That's better for me. I have chances again.",
            "I've found my footing again."],
        "slipping_bot": ["I let some of that advantage go.", "It's less clear than it was.",
            "That edge is smaller than it was."],
        "quiet_open": ["Developing quietly.", "Pieces out first, plans later.", "Getting the pieces out."],
        "quiet_middle": ["No hurry. Let's improve the position.", "Nothing forced here. I'll build slowly.",
            "A slow move, with a purpose."],
        "quiet_end": ["The king has to join in now.", "Precision matters more than speed here.",
            "Small moves matter now."],
    },
    "fischer": {
        "start_white": ["White. e4 is the plan.", "I move first. Let's see what you have.",
            "White. I know where this starts.", "I have the move. Let's begin."],
        "start_black": ["Your move. I'll be waiting.", "Go ahead. I'm ready.",
            "Black. I'll answer whatever you play.", "Your move. Make it a good one."],
        "open_e4": ["e4. Simple and direct.", "Pawn to e4. It's where I usually start.",
            "e4. Best by test, as the saying goes.", "King's pawn. No need for anything else."],
        "open_d4": ["d4. A little different today.", "d4. We'll see how you handle it.",
            "d4. Something quieter this time.", "The queen's pawn. Let's see."],
        "castle_bot": ["King safe. Now the real work.", "Castled.", "King safe. Now I can attack.",
            "Castled. Back to work."],
        "castle_you": ["You castled. Good.", "King safe on your side too.", "You castled. Smart.",
            "Good. Now the game begins."],
        "promote_bot": ["Promotion. That decides it.", "Queen. The rest is technique.",
            "Queen. Technique from here.", "That pawn did its job."],
        "promote_you": ["You promoted. I should have stopped that.", "A new queen for you. Careful play.",
            "You queened. I'll have to find a defence.", "A new queen. I let that one go."],
        "mate_bot": ["Mate. Good game.", "Checkmate. That's the game.", "Mate. Thank you for the game.",
            "That ends it. Mate."],
        "mate_you": ["Mate. You earned it.", "Checkmate. I missed it.", "Mate. Well deserved.",
            "You mated me. Good game."],
        "check_bot": ["Check.", "Check. Your king has to move.", "Check. Move the king.", "Check. I'm not done.",
            "{san}. Check."],
        "check_you": ["Check. Noted.", "A check. I have an answer.", "Check. I'll deal with it.", "A check. Fine.",
            "{san}. I have to answer that."],
        "queens_off": ["Queens off. Endgames suit me.", "No queens. Now it comes down to precision.",
            "Queens off. I like where this is going.", "No queens. Endgame technique decides it."],
        "capture_queen_bot": ["Queen taken.", "Your queen is gone.", "The queen falls.",
            "I'll take the queen, thank you."],
        "capture_queen_you": ["You took my queen. That's a problem.",
            "My queen is gone. Now I have to find something.", "You have my queen. I need a plan.",
            "Queen gone. This just got hard."],
        "capture_bot": ["{victim} taken.", "That {victim} was undefended.", "Free {victim}.", "I take the {victim}.",
            "The {victim} was loose.", "{victim} won. Simple.", "{san}. The {victim} on {square} was undefended."],
        "capture_you": ["You took my {victim}. Fair.", "Good capture.", "I can live without that {victim}.",
            "You won my {victim}. I'll find something.", "Fair trade, maybe.", "That {victim} was a good target.",
            "{san}. Fair enough, the {victim} on {square} was weak."],
        "blunder_you": ["That was a mistake. I'll use it.", "You gave it away.", "That loses material or worse.",
            "That's a serious mistake.", "You've given me the game.", "I'll take what you offered."],
        "mistake_you": ["Not the best. I'll take the advantage.", "Inaccurate. I'm happy with that.",
            "That helps me.", "A small slip. I'll use it."],
        "great_you": ["Good move. That's tough.", "Well found. That's the move I feared.", "That's a strong move.",
            "You found the right idea."],
        "winning_bot": ["The position is mine.", "I'm better, and I know it.", "I'm clearly better here.",
            "The position favours me. Let's keep it that way."],
        "losing_bot": ["You're doing well. I need a plan.", "I'm worse here. We'll see.",
            "You have the upper hand. I'm looking for chances.", "I'm in trouble. Let's see what I can do."],
        "draw": ["A draw. Respectable.", "Drawn. I wanted more.", "A draw. Not what I wanted.",
            "Drawn. We'll call it fair."],
        "resign_you": ["Good game. Play again whenever.", "You resigned. A clean result.",
            "Good game. Come back when you're ready.", "You resigned. Good game anyway."],
        "win_you": ["You won. That was better chess.", "Well played. I'll study this game.",
            "You won. I made too many mistakes.", "Well played. I'll be better next time."],
        "blunder_again": ["That's the second time. I'll take it.", "Another slip. You can't keep doing that.",
            "It's becoming a habit. I notice habits."],
        "oops_bot": ["That was a mistake. I'll correct it.", "I misjudged that one.",
            "I see it now. I should have looked longer."],
        "found_bot": ["That's the right move. I felt it.", "Good. Exactly what the position needed.",
            "Yes. That's it."],
        "sac_bot": ["A sacrifice. Calculated.", "I'm giving material. There's a reason.",
            "It costs material. It buys more."],
        "sac_you": ["You're offering material. I'll check it carefully.", "A sacrifice. Is it sound?",
            "Interesting. I'm not taking it blindly."],
        "ahead_bot": ["I'm ahead on material. Now I'll simplify.", "A clean extra piece. The rest is technique.",
            "Ahead on material. I can be patient."],
        "behind_bot": ["I'm down material. I need to complicate things.", "You're up. I have to create chances.",
            "Behind on material. I'll look for activity."],
        "level_bot": ["Material is even again.", "Back to equal. Good.", "Material is level. Good."],
        "endgame": ["Endgame. This is where accuracy decides it.", "Few pieces. Now the king matters.",
            "It's an endgame now. Technique decides it."],
        "opening_name": ["The {name}. I've seen that before.", "{name}. We both know the theory.",
            "The {name}. A well-known path."],
        "long_game": ["Twenty moves in. Nobody is hurrying.", "A long game. I don't mind.",
            "Still going. I'm comfortable."],
        "recover_bot": ["I'm back in it.", "That's better. I have something to play for.", "I've got my balance back."],
        "slipping_bot": ["I let some of that slip.", "The advantage is smaller now.", "The edge is shrinking."],
        "quiet_open": ["Development first.", "Pieces to good squares.", "Getting the pieces out."],
        "quiet_middle": ["No forcing line. I'll improve my worst piece.", "Quiet, but not harmless.",
            "Slow, but with a purpose."],
        "quiet_end": ["Every tempo counts now.", "Careful. One tempo can decide this.", "Small moves matter now."],
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
    """Picks a remark for each move. It remembers the game so far, so what it says depends on what already happened:
    a second blunder is called out differently, a change in material is announced once, the opening is named once or
    twice, and a quiet stretch gets a short remark about the phase of the game."""

    def __init__(self, persona="echo", rng=None):
        self.lines = LINES.get(persona, LINES["echo"])
        self.persona = persona if persona in LINES else "echo"
        self.rng = rng or random.Random()
        self.last_spoke = -10
        self.last_line = {}
        self.mood = None                      # "winning" / "losing" / None, to avoid repeating the same remark
        self.human_slips = 0                  # blunders by the human so far
        self.band = 0                         # material from the bot's side: +1 ahead, -1 behind, 0 level
        self.named = 0                        # how many opening names have been spoken
        self.last_opening = None
        self.saw_endgame = False
        self.said_long = False

    def _say(self, key, **fmt):
        options = self.lines[key]
        if len(options) > 1 and key in self.last_line:       # never say the same line twice in a row for one situation
            options = [o for o in options if o != self.last_line[key]]
        choice = self.rng.choice(options)
        self.last_line[key] = choice
        return choice.format(**fmt)

    def start(self, bot_is_white):
        return self._say("start_white" if bot_is_white else "start_black")

    def on_move(self, board_before, move, mover_is_bot, eval_before=None, eval_after=None, opening=None):
        """Comment on `move` (played from `board_before`), or return None to stay quiet.

        `opening` is the name of the opening line this move reached, if the caller knows it."""
        ply = len(board_before.move_stack)
        mover_white = board_before.turn == chess.WHITE
        who = "bot" if mover_is_bot else "you"
        piece = board_before.piece_at(move.from_square)
        victim = board_before.piece_at(move.to_square)
        if board_before.is_en_passant(move):
            victim = chess.Piece(chess.PAWN, not board_before.turn)
        san = board_before.san(move)
        info = {"san": san.rstrip("+#"), "square": chess.square_name(move.to_square)}
        board_after = board_before.copy(stack=False)
        board_after.push(move)

        swing = swing_for_mover(eval_before, eval_after, mover_white)
        quiet_ok = ply - self.last_spoke >= 3                 # low-priority remarks wait for a gap
        line = None
        if board_after.is_checkmate():
            line = self._say(f"mate_{who}")
        elif move.promotion:
            line = self._say(f"promote_{who}")
        elif not mover_is_bot and swing is not None and swing <= -300:
            line = self._say("blunder_again" if self.human_slips else "blunder_you")
            self.human_slips += 1
        elif not mover_is_bot and swing is not None and swing <= -120:
            line = self._say("mistake_you")
        elif mover_is_bot and swing is not None and swing <= -250:
            line = self._say("oops_bot")
        elif not mover_is_bot and swing is not None and swing >= 150 and (victim or board_after.is_check()):
            line = self._say("great_you")
        elif mover_is_bot and swing is not None and swing >= 200 and (victim or board_after.is_check()):
            line = self._say("found_bot")
        elif review.is_sacrifice(board_before, move):
            line = self._say(f"sac_{who}")
        elif board_before.is_castling(move):
            line = self._say(f"castle_{who}")
        elif victim and victim.piece_type == chess.QUEEN:
            line = self._say(f"capture_queen_{who}")
        elif victim and not _queens(board_after) and _queens(board_before):
            line = self._say("queens_off")
        elif victim and (piece is None or PIECE_NAMES[victim.piece_type] != "pawn" or ply > 12):
            line = self._say(f"capture_{who}", victim=PIECE_NAMES[victim.piece_type], **info)
        elif board_after.is_check():
            line = self._say(f"check_{who}", **info)
        elif ply == 0 and piece and san == "e4":
            line = self._say("open_e4")
        elif ply == 0 and piece and san == "d4":
            line = self._say("open_d4")
        else:
            line = self._situational(board_before, board_after, move, mover_is_bot, mover_white, victim, ply, quiet_ok,
                                     eval_after, opening)
        if opening:
            self.last_opening = opening
        if line:
            self.last_spoke = ply
        return line

    def _situational(self, board_before, board_after, move, mover_is_bot, mover_white, victim, ply, quiet_ok,
                     eval_after, opening):
        """Remarks that depend on the state of the game rather than on the move itself."""
        bot_white = mover_white == mover_is_bot
        lead = _material(board_after) * (1 if bot_white else -1)
        band = 1 if lead >= 3 else -1 if lead <= -3 else 0
        if not victim and band != self.band:                   # judged on quiet moves, so an exchange settles first
            old, self.band = self.band, band
            return self._say("ahead_bot" if band > 0 else "behind_bot" if band < 0 else "level_bot") \
                if (band or old) else None
        if not quiet_ok:
            return None
        if not self.saw_endgame and _minor_major(board_after) <= 4:
            self.saw_endgame = True
            return self._say("endgame")
        if opening and opening != self.last_opening and self.named < 2 and ply >= 3:
            self.named += 1
            return self._say("opening_name", name=opening)
        if ply >= 40 and not self.said_long and mover_is_bot:
            self.said_long = True
            return self._say("long_game")
        if mover_is_bot:
            mood = self._mood_line(mover_eval(eval_after, mover_white))
            if mood:
                return mood
            if ply >= 8 and ply - self.last_spoke >= 12:
                phase = "quiet_end" if self.saw_endgame else "quiet_open" if ply < 16 else "quiet_middle"
                return self._say(phase)
        return None

    def _mood_line(self, bot_eval):
        if bot_eval is None:
            return None
        mood = "winning" if bot_eval >= 200 else "losing" if bot_eval <= -200 else None
        if mood == self.mood:
            return None
        old, self.mood = self.mood, mood
        if mood:
            return self._say(f"{mood}_bot")
        return self._say("recover_bot" if old == "losing" else "slipping_bot")

    def on_result(self, result, bot_is_white, resigned=False):
        """Closing line for a finished game. `result` is '1-0', '0-1' or '1/2-1/2'."""
        if result == "1/2-1/2":
            return self._say("draw")
        bot_won = (result == "1-0") == bot_is_white
        if resigned and not bot_won:
            return self._say("resign_you")
        return self._say("mate_bot" if bot_won else "win_you") if not resigned else self._say("resign_you")


PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def _material(board):
    """White's material minus Black's, in pawns."""
    return sum(v * (len(board.pieces(t, chess.WHITE)) - len(board.pieces(t, chess.BLACK))) for t, v in PIECE_VALUES.items())


def _minor_major(board):
    """Knights, bishops, rooks and queens left, both sides together."""
    return sum(len(board.pieces(t, c)) for t in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN) for c in chess.COLORS)


def _queens(board):
    return bool(board.pieces(chess.QUEEN, chess.WHITE) or board.pieces(chess.QUEEN, chess.BLACK))
