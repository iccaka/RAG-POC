class ChromaDBEmbeddingFunction:
    """
    A class that wraps LangChain embeddings to provide a callable interface for embedding documents.

    This class is designed to integrate with LangChain's embedding functions, allowing users to
    easily generate embeddings for one or more input documents. It can handle both single string
    inputs and lists of strings.

    Attributes:
        embeddings: An instance of a LangChain embeddings class used to generate document embeddings.

    Methods:
        __call__(input): Generates embeddings for the provided input, which can be a single string
                         or a list of strings.
    """

    def __init__(self, langchain_embeddings):
        """
        Initializes the ChromaDBEmbeddingFunction with the specified LangChain embeddings.

        Args:
            langchain_embeddings: An instance of a LangChain embeddings class that provides the
                                  functionality to embed documents.
        """

        self.embeddings = langchain_embeddings

    def __call__(self, input):
        """
        Generates embeddings for the provided input.

        This method allows the instance to be called as a function. It checks the type of the input
        and converts a single string input into a list before passing it to the embeddings instance
        for embedding generation.

        Args:
            input (str or list): A single document as a string or a list of documents to be embedded.

        Returns:
            list: A list of embeddings corresponding to the input documents.
        """

        if isinstance(input, str):
            input = [input]

        return self.embeddings.embed_documents(input)
