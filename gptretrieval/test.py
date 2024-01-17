from services.classification import classify_question, classify_code


# add a main function
if __name__ == "__main__":
    # Test classify_question function
    question = "How do I create a function to do ABC?"
    question_label = classify_question(question)
    print(f"Question label: {question_label}")

    # Test classify_code function
    code = "def foo():\n    pass"
    code_label = classify_code(code, question, question_label)
    print(f"Code label: {code_label}")
