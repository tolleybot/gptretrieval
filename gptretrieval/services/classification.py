import os
from . import openai

# get gpt model env variable, or set default
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4")


labels_dict = {
    0: {
        "name": "Class or Struct Definition",
        "description": "Code that defines a class or struct. Excludes the methods within a class; only includes the class signature and member variables.",
    },
    1: {
        "name": "Function or Method Definition",
        "description": "Code that defines a function or method. Does not include usage examples of the function or method.",
    },
    2: {
        "name": "Code Usage or Example",
        "description": "Examples of how to use certain code, functions, or classes. Distinct from the actual definition of functions or classes.",
    },
    3: {
        "name": "Instructional Code Implementation",
        "description": "How to implement or use code.  Such as how do I create a function to do ABC, or how is a class used to do XYZ.",
    },
    4: {
        "name": "Database Implementation",
        "description": "Code that implements or calls a database.",
    },
    5: {
        "name": "Error Handling",
        "description": "Code segments dedicated to handling errors or exceptions.",
    },
    6: {
        "name": "UI Code",
        "description": "Code related to user interface design and interaction.",
    },
    7: {
        "name": "Configuration Code",
        "description": "Code used for configuring the system, application, or environment.",
    },
    8: {
        "name": "Documentation",
        "description": "Comments and documentation that explain the code. Does not include code itself.",
    },
    9: {
        "name": "REST API Implementation or Usage",
        "description": "Code that either implements a server or client, or calls a REST API.",
    },
    10: {
        "name": "Code Usage Search",
        "description": "Looking for location and or file where a specific function or class or variable is being used",
    },
}


def create_prompt_for_gpt(labels_dict):
    """
    Convert a dictionary of labels into a text block for GPT prompt.

    Parameters:
    labels_dict (dict): A dictionary containing label indices as keys and another dictionary with 'name' and 'description' as values.

    Returns:
    str: A text block suitable for use as a GPT prompt.
    """
    prompt = "The following are example labels but are not exclusive:\n\n"
    for _, label_info in labels_dict.items():
        prompt += f"Label - {label_info['name']}:\n"
        prompt += f"{label_info['description']}\n\n"
    return prompt


prompt_text = create_prompt_for_gpt(labels_dict)


def classify_question(question: str, model=GPT_MODEL, token_length=4096):
    """Call OpenAI to classify the given question."""
    question = question[:token_length]

    messages = [
        {
            "role": "system",
            "content": prompt_text
            + "\nYou can ask me to classify a question, and I will return a label for the question formatted as json. ",
        },
        {"role": "user", "content": f"Classify the following: Question - {question}"},
    ]

    # Define the function for the API call
    tools = [
        {
            "type": "function",
            "function": {
                "name": "classify_question",
                "description": "A function which takes in a question and classifies it",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question_label": {
                            "type": "string",
                            "description": "The label index assigned to the question",
                        }
                    },
                    "required": ["question_label"],
                },
            },
        }
    ]

    return openai.get_chat_completion(messages, tools=tools, model=model)


def classify_code(
    code: str, question: str, question_label: str, model=GPT_MODEL, token_length=4096
):
    """
    Call OpenAI to generate potential code labels based on the question and question label.
    """
    code = code[:token_length]

    messages = [
        {
            "role": "system",
            "content": prompt_text
            + "\nGiven a question and its classification, you can ask me to classify a code snippet. ",
        },
        {
            "role": "user",
            "content": f"The question is: '{question}'. It is classified as: '{question_label}'. Given this context, how would you classify the following code snippet: {code}?",
        },
    ]

    # Define the function for the API call
    tools = [
        {
            "type": "function",
            "function": {
                "name": "classify_code",
                "description": "A function which takes in a code label",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code_label": {
                            "type": "integer",
                            "description": "The label for the code",
                        }
                    },
                    "required": ["code_label"],
                },
            },
        }
    ]

    return openai.get_chat_completion(messages, tools=tools, model=model)
