"""Shared utility functions used in the project."""

from typing import Optional

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel


def _format_doc(doc: Document) -> str:
    """Format a single document as XML.

    Args:
        doc (Document): The document to format.

    Returns:
        str: The formatted document as an XML string.
    """
    metadata = doc.metadata or {}
    meta = "".join(f" {k}={v!r}" for k, v in metadata.items())
    if meta:
        meta = f" {meta}"

    return f"<document{meta}>\n{doc.page_content}\n</document>"


def format_docs(docs: Optional[list[Document]]) -> str:
    """Format a list of documents as XML.

    This function takes a list of Document objects and formats them into a single XML string.

    Args:
        docs (Optional[list[Document]]): A list of Document objects to format, or None.

    Returns:
        str: A string containing the formatted documents in XML format.

    Examples:
        >>> docs = [Document(page_content="Hello"), Document(page_content="World")]
        >>> print(format_docs(docs))
        <documents>
        <document>
        Hello
        </document>
        <document>
        World
        </document>
        </documents>

        >>> print(format_docs(None))
        <documents></documents>
    """
    if not docs:
        return "<documents></documents>"
    formatted = "\n".join(_format_doc(doc) for doc in docs)
    return f"""<documents>
{formatted}
</documents>"""


def _format_memory_value(value: dict) -> str:
    """Format a memory value dictionary into a readable bullet-point string."""
    bullet_points = []
    for key, val in value.items():
        # Skip empty values (None, empty strings, empty lists, etc.)
        if val and not (isinstance(val, (str, list)) and len(val) == 0):
            formatted_val = (
                val
                if isinstance(val, str)
                else ", ".join(map(str, val))
                if isinstance(val, list)
                else str(val)
            )
            readable_key = key.replace("_", " ").title()
            bullet_points.append(f"  - {readable_key}: {formatted_val}")
    return "\n".join(bullet_points) if bullet_points else "No details available"


def format_memories(memories) -> str:
    """Format the user's memories."""
    if not memories:
        return ""
    formatted_memories = "\n".join(
        f"• Memory:\n{_format_memory_value(m.value)}\n  Last updated: {m.updated_at}"
        for m in memories
    )
    return f"""

## Memories

You have noted the following memorable events from previous interactions with the user.
<memories>
{formatted_memories}
</memories>
"""


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
    """
    if "/" in fully_specified_name:
        provider, model = fully_specified_name.split("/", maxsplit=1)
    else:
        provider = ""
        model = fully_specified_name
    return init_chat_model(model, model_provider=provider)
