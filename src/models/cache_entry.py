"""Model for conversation history cache entry."""

from pydantic import BaseModel, Field
from typing import List

class ReferencedDocument(BaseModel):
    """Represents a single document referenced in an AI response."""
    doc_title: str | None = None
    doc_url: str | None = None

class AdditionalKwargs(BaseModel):
    """A structured model for the 'additional_kwargs' dictionary."""
    referenced_documents: List[ReferencedDocument] = Field(default_factory=list)

class LLMResponse(BaseModel):
    """Represents the complete response, mimicking LangChain's AIMessage structure."""
    text: str
    additional_kwargs: AdditionalKwargs | None = None


class CacheEntry(BaseModel):
    """Model representing a cache entry.

    Attributes:
        query: The query string
        response: The structured AI response.
        provider: Provider identification
        model: Model identification
    """

    query: str
    response: LLMResponse
    provider: str
    model: str


class ConversationData(BaseModel):
    """Model representing conversation data returned by cache list operations.

    Attributes:
        conversation_id: The conversation ID
        topic_summary: The topic summary for the conversation (can be None)
        last_message_timestamp: The timestamp of the last message in the conversation
    """

    conversation_id: str
    topic_summary: str | None
    last_message_timestamp: float
