from transformers import RobertaModel, RobertaTokenizer
import torch
from typing import List
import os

model_dir = os.getenv("TRANSFORMERS_MODEL_DIR")

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

# Load the CodeBERT model and tokenizer
model_name = "microsoft/codebert-base"
if model_dir:
    tokenizer = RobertaTokenizer.from_pretrained(model_dir)
    model = RobertaModel.from_pretrained(model_dir)
else:
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name).to(device)


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Embed texts using either OpenAI's ada model or CodeBERT.

    Args:
        texts: The list of texts to embed.
        use_openai: If True, use OpenAI's ada model. Otherwise, use CodeBERT.

    Returns:
        A list of embeddings, each of which is a list of floats.
    """
    if isinstance(texts, str):
        texts = [texts]

    embeddings = []
    for text in texts:
        # Tokenize the text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        # Move the inputs to the device
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

        # Get the embeddings from the model
        with torch.no_grad():
            outputs = model(**inputs)

        # The embeddings are the average of the token embeddings
        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy().tolist()

        embeddings.append(embedding[0])

    return embeddings


# setup a main function so we can run this file to test the code
if __name__ == "__main__":
    # Test the function with a piece of Python code
    code = "def hello_world():\n    print('Hello, world!')"
    embedding = get_embeddings(code)
    print(embedding)
