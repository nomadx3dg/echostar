"""
Optional Maia teacher: "what would a typical human of rating R play here?"

Maia models predict the move a human of a given rating would choose. EchoStar can use that as a *behavioral teacher*
(extra training signal, a benchmark) but never as the thing that decides the move. This module is
Python-only and optional: nothing in echo.py imports it unless a Maia flag is used, the Rust runtime never sees it,
and no Maia code or weights are bundled with EchoStar.

How Maia is reached
  The teacher talks to an external `lc0` process over UCI (the way the Maia authors say to run Maia-1: weights plus
  `go nodes 1`, no search). lc0 and the weights are separate programs under their own licenses; see docs/MAIA.md.
  A cache file lets any other source (for example Maia-3 run on a machine with a GPU) feed the same pipeline without
  lc0, and makes a training run reproducible without Maia installed.

Interface
  teacher.probs(board, moves) -> numpy array aligned with `moves` (the legal moves, in the caller's order), summing to 1.
"""
import json
import math
import os
import re

import chess
import chess.engine
import numpy as np

PROB_FLOOR = 1e-4          # lc0 prints percentages with two decimals; never let a legal move have probability 0
CACHE_VERSION = 1
_MOVE_LINE = re.compile(r"^(\S+)\s+\(\s*\d+\s*\)\s+N:.*?\(P:\s*([0-9.]+)%\)")
_MAIA1_NAME = re.compile(r"maia-(\d{4})\.pb(\.gz)?$")


class TeacherError(RuntimeError):
    """The teacher could not produce probabilities (missing binary, unsupported lc0, cache miss, ...)."""


def position_key(board: chess.Board) -> str:
    """Board, turn, castling, en passant: the same key the opening book uses (kept here to avoid importing echo)."""
    return " ".join(board.fen().split(" ")[:4])


def _normalize(raw: np.ndarray) -> np.ndarray:
    raw = np.maximum(np.asarray(raw, dtype=float), PROB_FLOOR)
    return raw / raw.sum()


def maia1_elo(weights_path: str):
    """Rating bucket from a file name like maia-1500.pb.gz, or None for other names."""
    m = _MAIA1_NAME.search(os.path.basename(weights_path))
    return int(m.group(1)) if m else None


def nearest_maia1_weights(elo: int, directory: str) -> str:
    """The Maia-1 weights file in `directory` whose rating bucket is closest to `elo`."""
    found = {}
    for name in os.listdir(directory):
        bucket = maia1_elo(name)
        if bucket is not None:
            found[bucket] = os.path.join(directory, name)
    if not found:
        raise TeacherError(f"no maia-NNNN.pb.gz files in {directory}")
    return found[min(found, key=lambda b: abs(b - elo))]


# ----------------------------------------------------------------------------------------------
# lc0 + Maia-1 weights
# ----------------------------------------------------------------------------------------------
def parse_verbose_move_stats(lines):
    """{uci: probability in 0..1} from lc0 `info string` lines produced with VerboseMoveStats.

    Lines look like `e2e4  (322 ) N: 0 (+ 0) (P: 20.15%) ...`; the summary line starting with `node` is skipped."""
    probs = {}
    for line in lines:
        m = _MOVE_LINE.match(line)
        if m and m.group(1) != "node":
            probs[m.group(1)] = float(m.group(2)) / 100.0
    return probs


class Lc0Teacher:
    """Maia-1 (any lc0-format weights) through a local lc0 binary, one forward pass per position (`go nodes 1`)."""

    def __init__(self, lc0_path, weights_path, backend="blas", threads=1, timeout=120):
        if not os.path.exists(weights_path):
            raise TeacherError(f"Maia weights not found: {weights_path}")
        self.lc0_path, self.weights_path = lc0_path, weights_path
        self.backend, self.threads, self.timeout = backend, threads, timeout
        self.elo = maia1_elo(weights_path)
        self.name = f"lc0:{os.path.basename(weights_path)}"
        self.engine = None
        self.calls = 0

    def _start(self):
        cmd = [self.lc0_path, f"--weights={self.weights_path}"]
        if self.backend:
            cmd.append(f"--backend={self.backend}")
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(cmd, timeout=self.timeout)
        except (OSError, chess.engine.EngineError) as exc:
            raise TeacherError(f"could not start lc0 ({self.lc0_path}): {exc}") from exc
        if "VerboseMoveStats" not in self.engine.options:
            self.close()
            raise TeacherError("this lc0 build has no VerboseMoveStats option; use a recent lc0 (v0.28 or newer)")
        opts = {"VerboseMoveStats": True, "PolicyTemperature": 1.0, "Threads": self.threads}
        self.engine.configure({k: v for k, v in opts.items() if k in self.engine.options})

    def probs(self, board, moves=None):
        moves = list(board.legal_moves) if moves is None else moves
        if not moves:
            return np.zeros(0)
        if self.engine is None:
            self._start()
        lines = []
        with self.engine.analysis(board, chess.engine.Limit(nodes=1)) as analysis:
            for info in analysis:
                if "string" in info:
                    lines.append(info["string"])
        self.calls += 1
        parsed = parse_verbose_move_stats(lines)
        # lc0 names castling as "king takes own rook" (e1h1), python-chess as e1g1; accept either spelling.
        raw = [parsed.get(m.uci(), parsed.get(board.uci(m, chess960=True), 0.0)) for m in moves]
        # lc0 prints percentages to two decimals, so over a whole legal-move set the total is within ~0.01 of 1;
        # a smaller total means some legal move was not understood, and silently flooring it would corrupt the signal.
        if len(moves) == board.legal_moves.count() and sum(raw) < 0.95:
            raise TeacherError(f"lc0 policy covers only {sum(raw):.2f} of the legal moves in {board.fen()} "
                               f"(got {len(parsed)} moves for {len(moves)} legal)")
        return _normalize(raw)

    def close(self):
        if self.engine is not None:
            try:
                self.engine.quit()
            except chess.engine.EngineError:
                pass
            self.engine = None

    def pid(self):
        """Process id of the lc0 child, for memory measurements (None before first use)."""
        return self.engine.transport.get_pid() if self.engine is not None else None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ----------------------------------------------------------------------------------------------
# Cache: a JSON-lines file, so any source of Maia probabilities can be replayed without Maia installed
# ----------------------------------------------------------------------------------------------
class CachingTeacher:
    """Wrap a teacher (or None) with a JSON-lines cache keyed by position.

    The first line names the teacher; a cache written by one teacher is refused by another so probabilities from
    different models are never mixed. With `inner=None` the cache is read-only and a miss is an error, which is
    what makes a distillation run reproducible on a machine without Maia."""

    def __init__(self, path, inner=None):
        self.path, self.inner = path, inner
        self.table, self.pending, self.hits, self.misses = {}, [], 0, 0
        self.name = getattr(inner, "name", None)
        self.elo = getattr(inner, "elo", None)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            self._load()
        elif inner is None:
            raise TeacherError(f"Maia cache not found (and no live teacher to fill it): {path}")

    def _load(self):
        with open(self.path, encoding="utf-8") as fh:
            header = json.loads(fh.readline())
            if header.get("cache_version") != CACHE_VERSION:
                raise TeacherError(f"{self.path}: unsupported cache version {header.get('cache_version')}")
            if self.name is not None and header.get("teacher") != self.name:
                raise TeacherError(f"{self.path} was written by {header.get('teacher')!r}, not {self.name!r}; "
                                   f"use a different cache file")
            self.name = header.get("teacher")
            self.elo = self.elo or header.get("elo")
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    self.table[row["k"]] = row["p"]

    def probs(self, board, moves=None):
        moves = list(board.legal_moves) if moves is None else moves
        if not moves:
            return np.zeros(0)
        key = position_key(board)
        row = self.table.get(key)
        if row is None:
            if self.inner is None:
                raise TeacherError(f"position not in the Maia cache {self.path}: {key}")
            self.misses += 1
            p = self.inner.probs(board, moves)
            row = {m.uci(): float(x) for m, x in zip(moves, p)}
            self.table[key] = row
            self.pending.append({"k": key, "p": row})
            return p
        self.hits += 1
        return _normalize([row.get(m.uci(), 0.0) for m in moves])

    def flush(self):
        """Append new entries to the file (writes the header first if the file is new)."""
        if not self.pending:
            return
        new = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
        with open(self.path, "a", encoding="utf-8") as fh:
            if new:
                fh.write(json.dumps({"cache_version": CACHE_VERSION, "teacher": self.name, "elo": self.elo}) + "\n")
            for row in self.pending:
                fh.write(json.dumps(row, separators=(",", ":")) + "\n")
        self.pending = []

    def close(self):
        self.flush()
        if self.inner is not None and hasattr(self.inner, "close"):
            self.inner.close()


# ----------------------------------------------------------------------------------------------
# Optional runtime use: a brain wrapper. It only changes the ORDER inside the safe set; it cannot widen the set.
# ----------------------------------------------------------------------------------------------
class MaiaBlendBrain:
    """Style score plus `weight * ln(maia probability)`, for `Echo(engine, MaiaBlendBrain(...))`.

    Echo only gives the brain moves that already passed the safety band, and the blunder gate re-checks the pick, so
    this cannot bypass either. It needs a live teacher at play time (lc0 + weights), which the Rust engine does not have."""

    def __init__(self, brain, teacher, weight=0.5):
        self.brain, self.teacher, self.weight = brain, teacher, float(weight)

    def __getattr__(self, name):          # player, net, book, ... come from the wrapped brain
        return getattr(self.brain, name)

    def scores(self, board, moves):
        base = self.brain.scores(board, moves)
        if self.weight == 0 or not moves:
            return base
        # Maia sees the whole position, so ask for the full legal-move distribution and read off the moves we were given
        legal = list(board.legal_moves)
        q = dict(zip((m.uci() for m in legal), self.teacher.probs(board, legal)))
        return [float(s + self.weight * math.log(q.get(m.uci(), PROB_FLOOR))) for s, m in zip(base, moves)]
