import setuptools

setuptools.setup(
    name="hipporag",
    version="2.0.0-alpha.4",
    author="Bernal Jimenez Gutierrez",
    author_email="jimenezgutierrez.1@osu.edu",
    description="A powerful graph-based RAG framework that enables LLMs to identify and leverage connections within new knowledge for improved retrieval.",
    long_description="Vendored HippoRAG runtime subset used by Em2Mem Online Server.",
    long_description_content_type="text/plain",
    url="https://github.com/OSU-NLP-Group/HippoRAG",
    package_dir={"": "src"},
    packages=setuptools.find_packages("src"),
    python_requires=">=3.10",
    install_requires=[
        "torch",
        "transformers",
        "litellm",
        "gritlm",
        "networkx",
        "python_igraph",
        "tiktoken",
        "pydantic",
        "tenacity",
        "einops", # No version specified
        "tqdm", # No version specified
        "boto3", # No version specified
    ]
)
