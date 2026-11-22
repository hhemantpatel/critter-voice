from ..registry import tool


@tool(question="a short question asking the user what they meant")
def clarify(question: str) -> str:
    """Ask the user to repeat or explain when the request is unclear or matches no other tool."""
    return question
