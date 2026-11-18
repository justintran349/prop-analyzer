"""Phone alerts through ntfy (free, no account):

1. Install the "ntfy" app on your phone and subscribe to a topic name only you
   know, e.g. justin-props-8f3k2.
2. Put NTFY_TOPIC=justin-props-8f3k2 in .env.
Anyone who knows the topic name can read it, so make it hard to guess.
"""
from __future__ import annotations

import pandas as pd
import requests

from . import config


def enabled() -> bool:
    return bool(config.NTFY_TOPIC)


def send(title: str, message: str, priority: str = "default", tags: str = "basketball") -> bool:
    if not enabled():
        return False
    r = requests.post(f"https://ntfy.sh/{config.NTFY_TOPIC}", data=message.encode("utf-8"),
                      headers={"Title": title.encode("utf-8"), "Priority": priority, "Tags": tags},
                      timeout=15)
    return r.ok


def edge_alerts(board: pd.DataFrame, min_edge: float = config.ALERT_MIN_EDGE, top: int = 10) -> str | None:
    if board.empty:
        return None
    hot = board[board["edge"] >= min_edge].head(top)
    if hot.empty:
        return None
    lines = [f"{r.player} {r.pick} {r.line} {config.STATS[r.stat]} {r.game} "
             f"({r.prob_pick:.0%}, edge {r.edge:+.0%}, {r.book})" for r in hot.itertuples()]
    return "\n".join(lines)


def move_alerts(moves: pd.DataFrame) -> str | None:
    if moves is None or moves.empty:
        return None
    return "\n".join(f"{r.player} {config.STATS.get(r.stat, r.stat)}: {r.open} -> {r.now} ({r.move:+.1f})"
                     for r in moves.itertuples())
