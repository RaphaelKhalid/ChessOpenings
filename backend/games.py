# Anonymous "play with a friend" game rooms — no accounts, no ratings.
# State lives in memory only: a backend restart clears all active games.
import asyncio
import secrets
import time
from dataclasses import dataclass, field

import chess
from fastapi import WebSocket

# FIDE-style presets. None means no time control (untimed game).
TIME_CONTROLS = {
    "bullet1": (60, 0),
    "blitz3": (180, 0),
    "blitz5": (300, 0),
    "rapid10": (600, 0),
    "rapid15": (900, 10),
    "classical30": (1800, 0),
    "untimed": None,
}


def new_id(n=8):
    alphabet = "abcdefghjkmnpqrstuvwxyz23456789"  # no ambiguous chars
    return "".join(secrets.choice(alphabet) for _ in range(n))


def new_token():
    return secrets.token_urlsafe(24)


@dataclass
class GameRoom:
    id: str
    time_control_key: str
    board: chess.Board = field(default_factory=chess.Board)
    tokens: dict = field(default_factory=dict)      # color -> token
    sockets: dict = field(default_factory=dict)      # color -> WebSocket
    clocks: dict = field(default_factory=lambda: {"white": None, "black": None})
    turn_started_at: float = field(default_factory=time.time)
    status: str = "waiting"   # waiting | active | finished
    result: str | None = None      # "1-0" | "0-1" | "1/2-1/2"
    reason: str | None = None      # "checkmate" | "resignation" | "timeout" | "stalemate" | "draw"
    created_at: float = field(default_factory=time.time)
    clock_task: asyncio.Task | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self):
        tc = TIME_CONTROLS[self.time_control_key]
        if tc is not None:
            initial, _increment = tc
            self.clocks = {"white": float(initial), "black": float(initial)}
        else:
            self.clocks = {"white": None, "black": None}

    @property
    def increment(self) -> int:
        tc = TIME_CONTROLS[self.time_control_key]
        return tc[1] if tc is not None else 0

    @property
    def is_timed(self) -> bool:
        return TIME_CONTROLS[self.time_control_key] is not None

    def open_color(self) -> str | None:
        for c in ("white", "black"):
            if c not in self.tokens:
                return c
        return None

    def color_for_token(self, token: str) -> str | None:
        for c, t in self.tokens.items():
            if t == token:
                return c
        return None

    def turn_color(self) -> str:
        return "white" if self.board.turn == chess.WHITE else "black"

    def live_clocks(self) -> dict:
        """Clocks as of right now, accounting for elapsed time on the side to move."""
        clocks = dict(self.clocks)
        if self.is_timed and self.status == "active":
            elapsed = time.time() - self.turn_started_at
            mover = self.turn_color()
            clocks[mover] = max(0.0, clocks[mover] - elapsed)
        return clocks

    def outcome_reason(self, outcome: chess.Outcome) -> str:
        term = outcome.termination
        if term == chess.Termination.CHECKMATE:
            return "checkmate"
        if term == chess.Termination.STALEMATE:
            return "stalemate"
        if term == chess.Termination.INSUFFICIENT_MATERIAL:
            return "insufficient material"
        if term == chess.Termination.SEVENTYFIVE_MOVES:
            return "75-move rule"
        if term == chess.Termination.FIVEFOLD_REPETITION:
            return "fivefold repetition"
        return "draw"

    def state_payload(self) -> dict:
        return {
            "type": "state",
            "fen": self.board.fen(),
            "turn": self.turn_color(),
            "clocks": self.live_clocks(),
            "is_timed": self.is_timed,
            "status": self.status,
            "result": self.result,
            "reason": self.reason,
            "last_move": (
                {"from": chess.square_name(self.board.peek().from_square),
                 "to": chess.square_name(self.board.peek().to_square)}
                if self.board.move_stack else None
            ),
            "move_count": self.board.fullmove_number,
        }


class GameStore:
    def __init__(self):
        self.games: dict[str, GameRoom] = {}

    def create(self, time_control_key: str, creator_color: str) -> tuple[GameRoom, str]:
        if time_control_key not in TIME_CONTROLS:
            raise ValueError("invalid time control")
        gid = new_id()
        while gid in self.games:
            gid = new_id()
        room = GameRoom(id=gid, time_control_key=time_control_key)
        if creator_color == "random":
            creator_color = secrets.choice(["white", "black"])
        token = new_token()
        room.tokens[creator_color] = token
        self.games[gid] = room
        return room, token

    def join(self, gid: str) -> tuple[GameRoom, str, str] | None:
        room = self.games.get(gid)
        if room is None:
            return None
        color = room.open_color()
        if color is None:
            return None
        token = new_token()
        room.tokens[color] = token
        if room.open_color() is None:
            room.status = "active"
            room.turn_started_at = time.time()
        return room, token, color

    def get(self, gid: str) -> GameRoom | None:
        return self.games.get(gid)


store = GameStore()
