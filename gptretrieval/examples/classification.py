import json
import sys

sys.path.append("/Users/dtolley/Documents/Projects/gptretrieval/gptretrieval")
from services import openai

GPT_TOKEN_LENGTH = 4096

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
        "name": "Algorithm Implementation",
        "description": "Code that implements a specific algorithm. Distinct from general function or method definitions, representing a complete or partial algorithm.",
    },
    4: {
        "name": "Data Structure Implementation",
        "description": "Code that implements data structures like arrays, linked lists, trees, etc. Does not include general class or struct definitions.",
    },
    5: {
        "name": "Library or Package Usage",
        "description": "Code that primarily demonstrates how to use external libraries or packages. Does not include import statements alone.",
    },
    6: {
        "name": "Error Handling",
        "description": "Code segments dedicated to handling errors or exceptions.",
    },
    7: {
        "name": "UI Code",
        "description": "Code related to user interface design and interaction.",
    },
    8: {
        "name": "Configuration Code",
        "description": "Code used for configuring the system, application, or environment.",
    },
    9: {
        "name": "Documentation",
        "description": "Comments and documentation that explain the code. Does not include code itself.",
    },
    10: {
        "name": "Import Statements",
        "description": "Lines of code that import libraries and packages. Only includes the import statements, not the usage of the libraries.",
    },
    11: {
        "name": "Initialization Code",
        "description": "Code that initializes variables, objects, or the environment. Does not include class or struct definitions.",
    },
}

test_cases = {
    0: [
        {
            "question": "How do you define a simple class in Python?",
            "code": "class MyClass:\n    def __init__(self, name):\n        self.name = name",
            "answer": (0, 0),  # Class or Struct Definition for both question and code
        }
    ],
    1: [
        {
            "question": "Can you provide a function that adds two numbers?",
            "code": "def add_numbers(a, b):\n    return a + b",
            "answer": (
                1,
                1,
            ),  # Function or Method Definition for both question and code
        }
    ],
    2: [
        {
            "question": "How do I use the add_numbers function?",
            "code": "result = add_numbers(3, 5)\nprint(result)",
            "answer": (2, 2),  # Code Usage or Example for both question and code
        }
    ],
    3: [
        {
            "question": "Can you show an implementation of the bubble sort algorithm?",
            "code": "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]",
            "answer": (3, 3),  # Algorithm Implementation for both question and code
        }
    ],
    4: [
        {
            "question": "How do you implement a stack data structure?",
            "code": "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, item):\n        self.items.append(item)\n    def pop(self):\n        return self.items.pop()",
            "answer": (
                4,
                4,
            ),  # Data Structure Implementation for both question and code
        }
    ],
    5: [
        {
            "question": "How do you use the pandas library to read a CSV file?",
            "code": "import pandas as pd\ndata = pd.read_csv('file.csv')",
            "answer": (5, 5),  # Library or Package Usage for both question and code
        }
    ],
    6: [
        {
            "question": "Can you show me how to handle a division by zero error?",
            "code": "try:\n    result = 10 / 0\nexcept ZeroDivisionError:\n    print('Cannot divide by zero!')",
            "answer": (6, 6),  # Error Handling for both question and code
        }
    ],
    7: [
        {
            "question": "How can I create a button in a Tkinter window?",
            "code": "from tkinter import Tk, Button\nroot = Tk()\nbutton = Button(root, text='Click Me')\nbutton.pack()\nroot.mainloop()",
            "answer": (7, 7),  # UI Code for both question and code
        }
    ],
    8: [
        {
            "question": "How do you set up a configuration file for logging in Python?",
            "code": "[loggers]\nkeys=root\n\n[logger_root]\nlevel=DEBUG\nhandlers=consoleHandler\n\n[handlers]\nkeys=consoleHandler\n\n[handler_consoleHandler]\nclass=StreamHandler\nlevel=DEBUG\nformatter=consoleFormatter\nargs=(sys.stdout,)",
            "answer": (8, 8),  # Configuration Code for both question and code
        }
    ],
    9: [
        {
            "question": "What is the purpose of documentation in code?",
            "code": "# This function adds two numbers\ndef add_numbers(a, b):\n    # Add the numbers and return the result\n    return a + b",
            "answer": (9, 9),  # Documentation for both question and code
        }
    ],
    10: [
        {
            "question": "How do you import libraries in Python?",
            "code": "import math\nfrom datetime import datetime",
            "answer": (10, 10),  # Import Statements for both question and code
        }
    ],
    11: [
        {
            "question": "Can you show an example of variable initialization?",
            "code": "x = 10\nname = 'John'",
            "answer": (11, 11),  # Initialization Code for both question and code
        }
    ],
}


def create_prompt_for_gpt(labels_dict):
    """
    Convert a dictionary of labels into a text block for GPT prompt.

    Parameters:
    labels_dict (dict): A dictionary containing label indices as keys and another dictionary with 'name' and 'description' as values.

    Returns:
    str: A text block suitable for use as a GPT prompt.
    """
    prompt = "The following are the labels and their descriptions:\n\n"
    for index, label_info in labels_dict.items():
        prompt += f"Label {index} - {label_info['name']}:\n"
        prompt += f"{label_info['description']}\n\n"
    return prompt


def classify_code_and_question(question: str, code: str, labels: str):
    """Call OpenAI and summarize the function or class definition"""

    prompt_text = create_prompt_for_gpt(labels_dict)
    code = code[:GPT_TOKEN_LENGTH]

    system_message = {
        "role": "system",
        "content": f"{prompt_text}\nYou can ask me to classify a question and a code snippet, \
            and I will return a label for the question and the code formatted as json. \
            formatted as {{'question': 'label', 'code': 'label'}}",
    }
    user_message = {
        "role": "user",
        "content": f"Classify the following: Question - {question}, Code - {code}",
    }

    functions = [
        {
            "name": "classify_code_and_question",
            "description": "Classifies the given question and code snippet into predefined labels.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question_label": {
                        "type": "string",
                        "description": "The label assigned to the question based on predefined classification criteria.",
                    },
                    "code_label": {
                        "type": "string",
                        "description": "The label assigned to the code snippet based on predefined classification criteria.",
                    },
                },
                "required": ["question_label", "code_label"],
            },
        }
    ]

    resp = openai.get_chat_completion(
        [system_message, user_message],
        functions,
        function_call={"name": "classify_code_and_question"},
        model="gpt-4",
    )

    return resp


# create a main entry point
def main():
    # print the summary for each test case
    for label, test_case in test_cases.items():
        for case in test_case:
            gpt_response = classify_code_and_question(
                case["question"], case["code"], labels_dict
            )
            # print question and code, and what the answer is supposed to be
            print(f"Question: {case['question']}")
            print(f"Code: {case['code']}")
            print(f"Answer: {case['answer']}")
            # print the GPT response
            print(f"GPT Response: {gpt_response}\n")


if __name__ == "__main__":
    main()
