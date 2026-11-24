"""Opening names and "book" positions, from data/openings.tsv (CC0, from lichess-org/chess-openings).

A position is *book* if it occurs on the way through any named line, so transpositions count. The name of a position
is that of the longest named line that ends exactly there.
"""
import csv
import os

import chess

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "openings.tsv")


def _tokens(pgn):
    for tok in pgn.replace("\n", " ").split():
        if tok[0].isdigit() and tok.rstrip(".").isdigit() or tok.endswith("."):
            continue
        yield tok


class Openings:
    def __init__(self, path=DEFAULT_PATH):
        self.book, self.names = set(), {}
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                board = chess.Board()
                try:
                    for san in _tokens(row["pgn"]):
                        board.push_san(san)
                        self.book.add(board.epd())
                except ValueError:
                    continue                                   # a malformed line must not take the whole file down
                key = board.epd()
                if key not in self.names or len(row["name"]) > len(self.names[key][1]):
                    self.names[key] = (row["eco"], row["name"])

    def __len__(self):
        return len(self.names)

    def is_book(self, board):
        return board.epd() in self.book

    def lookup(self, board):
        """(eco, name) if a named line ends at this position, else None."""
        return self.names.get(board.epd())


def load(path=DEFAULT_PATH):
    """The default opening data, or None if the file is missing (the UI then simply has no Book signs or names)."""
    return Openings(path) if os.path.exists(path) else None
