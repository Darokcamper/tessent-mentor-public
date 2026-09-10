"""
agents/attachment_context.py

Process-global holder for the current user-attached document text.

All domain agents funnel through either agents.expert_agent.ask_expert or
agents.base_agent.ask_expert. Instead of threading an extra_context argument
through 14 domain-agent wrappers, the UI layer sets the attachment here before
invoking the routed agent, and the ask_expert implementations pick it up.
"""
_extra_context = None


def set_attachment_context(text):
    """Store the attachment text for the next ask_expert call."""
    global _extra_context
    _extra_context = text


def get_attachment_context():
    """Return the currently stored attachment text (or None)."""
    return _extra_context


def clear_attachment_context():
    """Clear the stored attachment text after a call completes."""
    global _extra_context
    _extra_context = None