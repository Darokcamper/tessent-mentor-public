"""Pure formatting helpers for exporting the active chat conversation."""


def format_chat_transcript(messages) -> str:
    """Return user and assistant chat messages as a portable Markdown document."""
    sections = []
    labels = {"user": "User", "assistant": "Assistant"}

    for message in messages or []:
        if not isinstance(message, dict):
            continue
        content = str(message.get("content") or "").strip()
        label = labels.get(message.get("role"))
        if label and content:
            sections.append(f"## {label}\n\n{content}")

    body = "\n\n---\n\n".join(sections) if sections else "_No messages in this conversation._"
    return f"# Tessent Mentor AI — Conversation\n\n{body}\n"
