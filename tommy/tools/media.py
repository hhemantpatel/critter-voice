from typing import Literal
from urllib.parse import quote_plus
from ..registry import tool
from .. import system

# The model can only choose a name from this table; it never supplies a URL or a command.
SITES = {
    "youtube": ["brave", "https://www.youtube.com"],
    "whatsapp": ["brave", "--app=https://web.whatsapp.com"],
    "gmail": ["brave", "https://mail.google.com"],
    "github": ["brave", "https://github.com"],
    "chatgpt": ["brave", "https://chatgpt.com"],
    "maps": ["brave", "https://www.google.com/maps"],
    "spotify": ["spotify"],
}


@tool(site="which one to open: youtube, whatsapp, gmail, github, chatgpt, maps or spotify")
def open_site(site: Literal["youtube", "whatsapp", "gmail", "github", "chatgpt", "maps", "spotify"]) -> str:
    """Open a website or app."""
    system.launch(SITES[site])
    return f"Opening {site}."


@tool(query="the song, artist or kind of music to look for, exactly as said")
def play_music(query: str) -> str:
    """Search YouTube for a song, artist or kind of music."""
    system.launch(["brave", f"https://www.youtube.com/results?search_query={quote_plus(query)}"])
    return f"Searching YouTube for {query}."


@tool(action="play, pause, next or previous")
def media_control(action: Literal["play", "pause", "next", "previous"]) -> str:
    """Play, pause or skip whatever music or video is playing."""
    verb = {"play": "play", "pause": "pause", "next": "next", "previous": "previous"}[action]
    if not system.run(["playerctl", "-l"]):
        return "Nothing is playing."
    system.run(["playerctl", verb])
    return {"play": "Playing.", "pause": "Paused.", "next": "Next track.", "previous": "Previous track."}[action]
