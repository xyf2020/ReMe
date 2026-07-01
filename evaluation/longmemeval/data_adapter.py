"""Load and convert LongMemEval dataset for ReMe evaluation."""

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class Turn:
    """A single conversation turn."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime


@dataclass
class Session:
    """A conversation session with timestamped turns."""
    session_id: str
    turns: list[Turn] = field(default_factory=list)
    session_start: Optional[datetime] = None

    @property
    def date_str(self) -> str:
        """Return date in YYYY-MM-DD format."""
        if self.session_start:
            return self.session_start.strftime("%Y-%m-%d")
        if self.turns:
            return self.turns[0].timestamp.strftime("%Y-%m-%d")
        return ""

    @property
    def start_time(self) -> Optional[datetime]:
        """Return the session start time."""
        if self.session_start:
            return self.session_start
        if self.turns:
            return self.turns[0].timestamp
        return None


@dataclass
class Question:
    """A LongMemEval question with metadata."""
    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: Optional[datetime] = None
    answer_session_ids: list[str] = field(default_factory=list)


@dataclass
class EvalItem:
    """A single evaluation item from LongMemEval."""
    question: Question
    sessions: list[Session] = field(default_factory=list)

    def filter_sessions_before_question(self) -> list[Session]:
        """Return only sessions that occur before the question date."""
        if not self.question.question_date:
            return self.sessions
        return [s for s in self.sessions if s.start_time and s.start_time <= self.question.question_date]


def parse_longmemeval_date(date_str: str) -> Optional[datetime]:
    """Parse LongMemEval date format: 'YYYY/MM/DD (DOW) HH:MM' or 'YYYY/MM/DD'.

    Examples:
        '2023/05/20 (Sat) 02:21' -> datetime(2023, 5, 20, 2, 21)
        '2023/05/30' -> datetime(2023, 5, 30)
    """
    if not date_str or not date_str.strip():
        return None

    date_str = date_str.strip()

    # Try format with day of week and time: 'YYYY/MM/DD (DOW) HH:MM'
    match = re.match(r'(\d{4}/\d{2}/\d{2})\s*\(\w+\)\s*(\d{2}:\d{2})', date_str)
    if match:
        date_part, time_part = match.groups()
        return datetime.strptime(f"{date_part} {time_part}", "%Y/%m/%d %H:%M")

    # Try format with time but no day of week: 'YYYY/MM/DD HH:MM'
    match = re.match(r'(\d{4}/\d{2}/\d{2})\s+(\d{2}:\d{2})', date_str)
    if match:
        date_part, time_part = match.groups()
        return datetime.strptime(f"{date_part} {time_part}", "%Y/%m/%d %H:%M")

    # Try date only: 'YYYY/MM/DD'
    match = re.match(r'(\d{4}/\d{2}/\d{2})', date_str)
    if match:
        date_part = match.group(1)
        return datetime.strptime(date_part, "%Y/%m/%d")

    return None


def assign_turn_timestamps(session_start: datetime, num_turns: int, turn_interval_seconds: int = 30) -> list[datetime]:
    """Assign timestamps to turns within a session.

    Each turn is spaced `turn_interval_seconds` apart from the previous turn.

    Args:
        session_start: When the session begins
        num_turns: Number of turns in the session
        turn_interval_seconds: Seconds between consecutive turns (default 30)

    Returns:
        List of timestamps for each turn
    """
    return [session_start + timedelta(seconds=i * turn_interval_seconds) for i in range(num_turns)]


def load_longmemeval_data(
    data_path: str,
    limit: Optional[int] = None,
) -> list[EvalItem]:
    """Load LongMemEval dataset and convert to ReMe-compatible format.

    Args:
        data_path: Path to the LongMemEval JSON file (e.g., longmemeval_s_cleaned.json)
        limit: Maximum number of items to load (None = all)

    Returns:
        List of EvalItem objects
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if limit is not None:
        data = data[:limit]

    items = []
    for idx, item in enumerate(data):
        # Parse question
        question_date = parse_longmemeval_date(item.get('question_date', ''))
        question = Question(
            question_id=str(item.get('question_id', idx)),
            question_type=item.get('question_type', ''),
            question=item.get('question', ''),
            answer=item.get('answer', ''),
            question_date=question_date,
            answer_session_ids=item.get('answer_session_ids', []),
        )

        # Parse sessions
        sessions = []
        haystack_sessions = item.get('haystack_sessions', [])
        haystack_dates = item.get('haystack_dates', [])
        haystack_session_ids = item.get('haystack_session_ids', [])

        for sess_idx, sess_turns in enumerate(haystack_sessions):
            # Get session start time
            session_start = None
            if sess_idx < len(haystack_dates):
                session_start = parse_longmemeval_date(haystack_dates[sess_idx])

            if not session_start:
                # Fallback: use a default date if parsing fails
                session_start = datetime(2023, 5, 20, 12, 0, 0)

            # Get session ID
            session_id = haystack_session_ids[sess_idx] if sess_idx < len(haystack_session_ids) else f"session_{sess_idx}"

            # Assign timestamps to turns
            num_turns = len(sess_turns)
            turn_timestamps = assign_turn_timestamps(session_start, num_turns)

            # Create Turn objects
            turns = []
            for turn_idx, turn_data in enumerate(sess_turns):
                role = turn_data.get('role', 'user')
                content = turn_data.get('content', '')
                timestamp = turn_timestamps[turn_idx]
                turns.append(Turn(role=role, content=content, timestamp=timestamp))

            sessions.append(Session(
                session_id=session_id,
                turns=turns,
                session_start=session_start,
            ))

        eval_item = EvalItem(question=question, sessions=sessions)

        # Filter sessions to only those before question date
        eval_item.sessions = eval_item.filter_sessions_before_question()

        # Sort sessions by start time
        eval_item.sessions.sort(key=lambda s: s.start_time or datetime.min)

        items.append(eval_item)

    return items


def format_session_for_reme(session: Session) -> list[dict]:
    """Convert a Session to ReMe's message format.

    ReMe expects messages with: name, role, content, created_at

    Args:
        session: Session object with timestamped turns

    Returns:
        List of message dicts compatible with ReMe's auto_memory
    """
    messages = []
    for turn in session.turns:
        messages.append({
            "name": turn.role,
            "role": turn.role,
            "content": [{"type": "text", "text": turn.content}],
            "created_at": turn.timestamp.isoformat(),
        })
    return messages
