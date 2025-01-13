"""Model for the configuration of the memory graph."""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class Router(BaseModel):
    """Classify user query."""

    logic: str
    type: Literal["more-info", "langchain", "general"]


class Plan(BaseModel):
    """Generate research plan."""

    steps: list[str]


class Response(BaseModel):
    """Response containing generated search queries."""

    queries: list[str]


class User(BaseModel):
    """Store all important information about a user."""

    preferred_name: Optional[str] = None
    last_updated: datetime
    current_age: Optional[str] = None
    skills: list[str] = Field(description="Various skills the user has.")
    interests: list[str] = Field(
        description="A list of the user's interests",
    )
    conversation_preferences: list[str] = Field(
        description="A list of the user's preferred conversation styles, pronouns, topics they want to avoid, etc.",
    )
    topics_discussed: list[str] = Field(
        description="Unique topics the user has discussed"
    )
    other_preferences: list[str] = Field(
        description="Other preferences the user has expressed that informs how you should interact with them."
    )
    relationships: list[str] = Field(
        description="Store information about friends, family members, coworkers, and other important relationships the user has here. Include relevant information about them."
    )

    # Add validators for all list fields to ensure unique values
    @field_validator(
        "skills",
        "interests",
        "conversation_preferences",
        "topics_discussed",
        "other_preferences",
        "relationships",
    )
    @classmethod
    def ensure_unique_items(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        """Ensure that the list contains unique items."""
        if v is None:
            return v
        return list(dict.fromkeys(v))  # Preserves order while removing duplicates


class Note(BaseModel):
    """Save notable memories the user has shared with you for later recall and store contextual notes and memories about user interactions."""

    context: str = Field(
        description="The situation or circumstance where this memory may be relevant. "
        "Include any caveats or conditions that contextualize the memory. "
        "For example, if a user shares a preference, note if it only applies "
        "in certain situations (e.g., 'only at work'). Add any other relevant "
        "'meta' details that help fully understand when and how to use this memory."
    )
    content: str = Field(
        description="The specific information, preference, or event being remembered."
    )
