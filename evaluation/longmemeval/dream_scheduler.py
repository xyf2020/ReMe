"""Dream scheduler for LongMemEval evaluation.

Handles the logic of when to call dream_cron based on session timestamps
crossing 23:00 boundaries.
"""

from datetime import datetime, timedelta
from typing import Optional


def find_dream_boundaries(sessions: list, session_timestamps: list[datetime]) -> list[datetime]:
    """Find all 23:00 boundaries between consecutive sessions.

    When sessions cross a 23:00 boundary, we need to call dream to consolidate
    the memory from the previous day before processing the next session.

    Args:
        sessions: List of session objects
        session_timestamps: List of session start timestamps (aligned with sessions)

    Returns:
        List of dream trigger timestamps (one for each 23:00 crossing)
    """
    if len(session_timestamps) < 2:
        return []

    dream_triggers = []

    for i in range(len(session_timestamps) - 1):
        current_time = session_timestamps[i]
        next_time = session_timestamps[i + 1]

        # Find all 23:00 boundaries between current and next session
        boundary = find_next_23h_boundary(current_time)
        while boundary and boundary < next_time:
            dream_triggers.append(boundary)
            # Look for the next 23:00 boundary
            boundary = find_next_23h_boundary(boundary + timedelta(minutes=1))

    return dream_triggers


def find_next_23h_boundary(from_time: datetime) -> Optional[datetime]:
    """Find the next 23:00 after the given time.

    Args:
        from_time: Starting time

    Returns:
        Next 23:00 datetime, or None if not found within 48 hours
    """
    # Start from the same day
    candidate = from_time.replace(hour=23, minute=0, second=0, microsecond=0)

    # If we're already past 23:00 today, move to tomorrow
    if candidate <= from_time:
        candidate += timedelta(days=1)

    return candidate


def get_dream_date_for_boundary(boundary: datetime) -> str:
    """Get the date to pass to dream for a given 23:00 boundary.

    Dream should process the current day's notes, so we use the boundary's date.

    Args:
        boundary: The 23:00 boundary timestamp

    Returns:
        Date string in YYYY-MM-DD format
    """
    return boundary.strftime("%Y-%m-%d")


def should_dream_between_sessions(
    prev_session_end: datetime,
    next_session_start: datetime,
) -> bool:
    """Check if there's a 23:00 boundary between two sessions.

    Args:
        prev_session_end: End time of previous session
        next_session_start: Start time of next session

    Returns:
        True if at least one 23:00 boundary exists between them
    """
    boundary = find_next_23h_boundary(prev_session_end)
    return boundary is not None and boundary < next_session_start


def count_23h_boundaries(start: datetime, end: datetime) -> int:
    """Count how many 23:00 boundaries exist between two times.

    Args:
        start: Start time
        end: End time

    Returns:
        Number of 23:00 boundaries
    """
    if start >= end:
        return 0

    count = 0
    boundary = find_next_23h_boundary(start)
    while boundary and boundary < end:
        count += 1
        boundary = find_next_23h_boundary(boundary + timedelta(minutes=1))

    return count
