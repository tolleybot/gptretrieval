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


def classify_question(question: str, token_length=4096):
    """Call OpenAI and summarize the function or class definition"""

    question = question[:token_length]

    system_message = {
        "role": "system",
        "content": f"{prompt_text}\nYou can ask me to classify a question, \
            and I will return a label for the question formatted as json. \
            formatted as {{'question': 'label']}}",
    }
    user_message = {
        "role": "user",
        "content": f"Classify the following: Question - {question}",
    }

    functions = [
        {
            "name": "classify_question",
            "description": "A function which takes in the label for question",
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
        }
    ]

    resp = openai.get_chat_completion(
        [system_message, user_message],
        functions,
        function_call={"name": "classify_question"},
        model=GPT_MODEL,
    )

    return resp


def classify_code(code: str, question: str, question_label: str, token_length=4096):
    """
    Call OpenAI to generate potential code labels based on the question and question label.

    Parameters:
    code (str): The code snippet.
    question (str): The question text.
    question_label (str): The label assigned to the question.

    Returns:
    Response from OpenAI API.
    """
    # Craft the system message with the labels dictionary and instruction
    code = code[:token_length]

    system_message = {
        "role": "system",
        "content": f"{prompt_text}\nGiven a question and its classification, you can ask me to classify a code snippet. \
            The classification of the code snippet is '1' if it should align with the context provided by the question and its classification else its 0. \
                Think of the code classification as the role the code plays in the context of answering the classified question. \
                    For example, if the question is asking for a class definition, but the code snippet is using a class without \
                        defining it, the code snippet should be classified as '0' or irrelevant.",
    }

    user_message = {
        "role": "user",
        "content": f"The question is: '{question}'. It is classified as: '{question_label}'. Given this context, how would you classify the following code snippet: {code}?",
    }

    # Define the function for the API call
    functions = [
        {
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
        }
    ]

    # Make the API call to GPT-4 with the crafted messages and function
    resp = openai.get_chat_completion(
        [system_message, user_message],
        functions,
        function_call={"name": "classify_code"},
        model=GPT_MODEL,
    )

    return resp
