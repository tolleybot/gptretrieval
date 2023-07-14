import sentence_transformers
import torch
from typing import List

# Load the CodeBERT model and tokenizer
model_name = "krlvi/sentence-msmarco-bert-base-dot-v5-nlpl-code_search_net"

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
model = sentence_transformers.SentenceTransformer(model_name, device=device)


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

    # Get the embeddings from the model
    embeddings = model.encode(texts, convert_to_tensor=True)
    embeddings = embeddings.cpu().numpy().tolist()

    return embeddings


# setup a main function so we can run this file to test the code
if __name__ == "__main__":
    # Test the function with a piece of Python code
    code = "def hello_world():\n    print('Hello, world!')"
    embedding = get_embeddings([code, code])
    print(embedding)