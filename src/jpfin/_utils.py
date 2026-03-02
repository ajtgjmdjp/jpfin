"""Shared utilities."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any


def parse_date(d: Any) -> date | None:
    """Parse a date value from various formats.

    Handles: date, datetime, ISO format string. Returns None on failure.
    """
    if d is None:
        return None
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    if isinstance(d, str):
        try:
            return date.fromisoformat(d)
        except ValueError:
            return None
    return None
