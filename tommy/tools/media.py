from typing import Literal
from urllib.parse import quote_plus
from ..registry import Reply, tool
from .. import apps, system

@tool(name="the app or website to open, exactly as said")
def open_app(name: str) -> str:
    """Open an installed app or a website by its name."""
    app, _ = apps.resolve(name)               # only ever selects an installed entry or a listed site, never a command
    if app is None:
        return Reply(f"There's no app called {name}, sir.", "no_app")
    system.launch(apps.command(app))
    return f"Opening {app.name}."


@tool(query="the song, artist or kind of music to look for, exactly as said")
def play_music(query: str) -> str:
    """Search YouTube for a song, artist or kind of music."""
    system.launch(["brave", f"https://www.youtube.com/results?search_query={quote_plus(query)}"])
    return f"Searching YouTube for {query}."


@tool(action="play, pause, next or previous")
def media_control(action: Literal["play", "pause", "next", "previous"]) -> str:
    """Play, pause or skip whatever music or video is playing."""
    verb = {"play": "play", "pause": "pause", "next": "next", "previous": "previous"}[action]
    if not system.ok(["playerctl", verb]):
        return Reply("Nothing is playing, sir.", "failed")
    return {"play": "Playing.", "pause": "Paused.", "next": "Next track.", "previous": "Previous track."}[action]
