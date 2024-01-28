from typing import List
from openai import OpenAI, OpenAIError
import os
import json
import re

client = OpenAI(api_key=os.environ.get("client_API_KEY"))
assert client.api_key is not None, "client_API_KEY environment variable must be set"
# get gpt model env variable, or set default
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")

from tenacity import retry, wait_random_exponential, stop_after_attempt


def clean_str(message):
    # Preserving line breaks but removing extra whitespace from each line
    lines = message.split("\n")
    cleaned_lines = [line.strip() for line in lines]

    message = " ".join(cleaned_lines)  # Rejoining the lines with a space
    message = message.replace("\t", " ")  # Replacing tabs with a single space
    # Handling plus signs as before
    message = message.replace(" + ", " ")
    message = message.replace("+ ", " ")
    message = message.replace(" +", " ")
    message = re.sub(r"\s+", " ", message).strip()  # Normalizing whitespace
    return message


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Embed texts using client's ada model.

    Args:
        texts: The list of texts to embed.

    Returns:
        A list of embeddings, each of which is a list of floats.

    Raises:
        Exception: If the client API call fails.
    """
    if isinstance(texts, str):
        texts = [texts]
    # Call the client API to get the embeddings
    response = client.embeddings.create(input=texts, model=EMBEDDING_MODEL)

    # Extract the embedding data from the response
    data = response.data  # type: ignore

    # Return the embeddings as a list of lists of floats

    # return [np.random.rand(1536).tolist() for _ in range(len(texts))]
    return [result.embedding for result in data]


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
def get_chat_completion(messages, tools=None, tool_choice="auto", model="gpt-4"):
    """
    Generate a chat completion using client's chat completion API.

    Args:
        messages: The list of messages in the chat history.
        tools: The list of available tools (functions).
        tool_choice: The choice of tool to use ("auto" or specific tool name).
        model: The name of the model to use for the completion.

    Returns:
        A string containing the chat completion or the response from the model.

    Raises:
        Exception: If the client API call fails.
    """
    # Call the client chat completion API with the given messages and tools
    response = client.chat.completions.create(
        model=model, messages=messages, tools=tools, tool_choice=tool_choice
    )

    # Process the response and handle tool calls if any
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    # Check if the model wanted to call a function
    if tool_calls:
        # Extend the conversation with the assistant's reply
        messages.append(response_message)

        functions = []
        # Handle each tool call
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            functions.append(
                {"function_name": function_name, "function_args": function_args}
            )

        if len(functions) == 1:
            return functions[0]
        else:
            return functions

    else:
        # If no tool calls, return the initial completion
        return response


if __name__ == "__main__":
    # Test get_embeddings function
    texts = ["Hello world!", "How are you?"]
    embeddings = get_embeddings(texts)
    print(f"Embeddings: {embeddings}")

    # Test get_chat_completion function
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
    ]
    completion = get_chat_completion(messages)
    print(f"Chat completion: {completion}")
