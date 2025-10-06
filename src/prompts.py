def user_prompt(question, context) -> str:
    """
    Generates a user prompt for information retrieval task

    Args:
        text: The text that needs to be corrected

    Returns:
        str: A formatted user prompt for testing.
    """
    return f"""Answer the following question using the provided context. 
If you can't find the answer, do not pretend you know it, but answer "I don't know".

### Question
{question}

### Context 
{context}
"""


def system_prompt() -> str:
    """
    Generates a user prompt for information retrieval task

    Args:
        text: The text that needs to be corrected

    Returns:
        str: A formatted user prompt for testing.
    """
    return f"""You are an culture assistant specialized in information about museuems, cultural foundations and events. You get as input questions and need to anwser with accuracy."""
