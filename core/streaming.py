from core.llm import llm

def stream_response(messages):

    for chunk in llm.stream(messages):

        if hasattr(chunk, "content"):

            yield chunk.content