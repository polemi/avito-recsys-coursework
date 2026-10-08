from __future__ import annotations

from datetime import datetime, timedelta


def leaf_key(vertical: str, category: str, param1: str) -> str:
    """Return the stable full taxonomy path used as the prediction class."""
    return f"{vertical}|{category}|{param1}"


def is_history_event(ts: datetime, snapshot: datetime, history_days: int = 90) -> bool:
    """History is left-closed and right-open: [T-history_days, T)."""
    return snapshot - timedelta(days=history_days) <= ts < snapshot


def is_target_event(ts: datetime, snapshot: datetime, target_days: int = 14) -> bool:
    """Target is left-closed and right-open: [T, T+target_days)."""
    return snapshot <= ts < snapshot + timedelta(days=target_days)


def new_target_leaves(
    history_contacts: list[tuple[datetime, str]],
    future_contacts: list[tuple[datetime, str]],
    snapshot: datetime,
    novelty_days: int = 90,
    target_days: int = 14,
) -> set[str]:
    masked = {
        leaf
        for ts, leaf in history_contacts
        if snapshot - timedelta(days=novelty_days) <= ts < snapshot
    }
    future = {
        leaf
        for ts, leaf in future_contacts
        if snapshot <= ts < snapshot + timedelta(days=target_days)
    }
    return future - masked

