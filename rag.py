class ChromaDBEmbeddingFunction:
    def __init__(self, langchain_embeddings):
        self.embeddings = langchain_embeddings

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]

        return self.embeddings.embed_documents(input)
