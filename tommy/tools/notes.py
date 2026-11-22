from datetime import datetime
from pathlib import Path
from ..registry import tool

NOTES = Path.home() / "Documents" / "notes" / "quick.md"


@tool(text="the note, exactly what to save")
def add_note(text: str) -> str:
    """Save a quick note."""
    NOTES.parent.mkdir(parents=True, exist_ok=True)
    with NOTES.open("a") as f:
        f.write(f"- {datetime.now():%Y-%m-%d %H:%M} {text.strip()}\n")
    return "Noted."


@tool(count="how many recent notes")
def read_notes(count: int = 3) -> str:
    """Read back the latest quick notes."""
    lines = NOTES.read_text().splitlines()[-max(1, min(count, 10)):] if NOTES.exists() else []
    return " | ".join(l.lstrip("- ") for l in lines) or "No notes yet."
