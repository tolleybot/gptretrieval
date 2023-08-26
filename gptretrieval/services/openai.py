from typing import List
import openai
import os
import numpy as np

# set openai key from environment variable or assert
openai.api_key = os.environ.get("OPENAI_API_KEY")
assert openai.api_key is not None, "OPENAI_API_KEY environment variable must be set"

from tenacity import retry, wait_random_exponential, stop_after_attempt


def clean_str(message):
    message = message.replace("\n", "")
    message = message.replace("\t", "")
    message = message.replace("\r", "")
    message = message.replace(" + ", " ")  # Remove + operators with spaces around them
    message = message.replace("+ ", " ")  # Remove + operators with space after them
    message = message.replace(" +", " ")  # Remove + operators with space before them
    message = message.strip()  # Remove any leading or trailing whitespace
    return message


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Embed texts using OpenAI's ada model.

    Args:
        texts: The list of texts to embed.

    Returns:
        A list of embeddings, each of which is a list of floats.

    Raises:
        Exception: If the OpenAI API call fails.
    """
    if isinstance(texts, str):
        texts = [texts]
    # Call the OpenAI API to get the embeddings
    response = openai.Embedding.create(input=texts, model="text-embedding-ada-002")

    # Extract the embedding data from the response
    data = response["data"]  # type: ignore

    # Return the embeddings as a list of lists of floats

    # return [np.random.rand(1536).tolist() for _ in range(len(texts))]
    return [result["embedding"] for result in data]


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
def get_chat_completion(
    messages,
    functions=None,
    function_call="auto",
    model="gpt-3.5-turbo",  # use "gpt-4" for better results
):
    """
    Generate a chat completion using OpenAI's chat completion API.

    Args:
        messages: The list of messages in the chat history.
        model: The name of the model to use for the completion. Default is gpt-3.5-turbo, which is a fast, cheap and versatile model. Use gpt-4 for higher quality but slower results.

    Returns:
        A string containing the chat completion.

    Raises:
        Exception: If the OpenAI API call fails.
    """
    # call the OpenAI chat completion API with the given messages
    if functions:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            functions=functions,
            function_call=function_call,  # auto is default, but we'll be explicit
        )

        response_message = response["choices"][0]["message"]

        # Step 2: check if GPT wanted to call a function
        if response_message.get("function_call"):
            function_name = response_message["function_call"]["name"]
            message = response_message["function_call"]["arguments"]
            message = clean_str(message)
            function_args = json.loads(message)
            return {"function_name": function_name, "function_args": function_args}
        else:
            raise Exception(
                "GPT-4 tried to call a function, but no functions were provided"
            )

    else:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
        )

        choices = response["choices"]  # type: ignore
        completion = choices[0].message.content.strip()
        print(f"Completion: {completion}")
        return completion
