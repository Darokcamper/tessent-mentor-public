def get_chat_context(messages, n=6):

    history = []

    for msg in messages[-n:]:

        role = msg["role"]

        content = msg["content"]

        history.append(
            f"{role.upper()}: {content}"
        )

    return "\n".join(history)