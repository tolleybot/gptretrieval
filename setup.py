from setuptools import setup, find_packages

setup(
    name="gptretrieval",  # Required
    version="1.0.0",  # Required
    url="https://github.com/tolleybot/gptretrieval.git",  # Optional
    author="Don Tolley",  # Optional
    author_email="tolleybot@gmai.com",  # Optional
    classifiers=[  # Optional
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    keywords="gptretrieval, AI, OpenAI",  # Optional
    packages=find_packages(exclude=["contrib", "docs", "tests"]),  # Required
    install_requires=[
        "PyPDF2",
        "textract",
        "sentence-transformers",
        "nltk",
        "uvicorn",
        "fastapi",
        "python-multipart",
        "pymilvus",
        "pyyaml",
        "beautifulsoup4",
        "markdown",
        "openai",
        "tenacity",
        "tree_sitter",
        "tree_sitter_languages",
    ],
    project_urls={  # Optional
        "Bug Reports": "https://github.com/tolleybot/gptretrieval.git/issues",
        "Source": "https://github.com/tolleybot/gptretrieval.git/",
    },
)
