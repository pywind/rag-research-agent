"""Utility functions used in our graph."""

from typing import Any, Literal, Sequence

from langchain_core.messages import BaseMessage, merge_message_runs
from pydantic import BaseModel


def prepare_messages(
    messages: Sequence[BaseMessage], system_prompt: str
) -> list[BaseMessage]:
    """Merge message runs and add instructions before and after to stay on task."""
    sys = {
        "role": "system",
        "content": f"""{system_prompt}
<memory-system>Reflect on following interaction. Use the provided tools to \
 retain any necessary memories about the user. Use parallel tool calling to handle updates & insertions simultaneously.</memory-system>
""",
    }
    m = {
        "role": "user",
        "content": "## End of conversation\n\n"
        "<memory-system>Reflect on the interaction above."
        " What memories ought to be retained or updated?</memory-system>",
    }
    return list(merge_message_runs(messages=[sys] + list(messages) + [m]))


def create_memory_function(
    model: type[BaseModel],
    description: str = "",
    custom_instructions: str = "",
    kind: Literal["patch", "insert"] = "patch",
) -> dict[Any, Any | dict]:
    """Create a memory function."""
    return {
        "name": model.__name__,
        "description": description or model.__doc__ or "",
        "parameters": model.model_json_schema(),
        "system_prompt": custom_instructions,
        "update_mode": kind,
    }
