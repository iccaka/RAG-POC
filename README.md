# RAG-POC

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline designed for managing and querying concert-related documents. Utilizing **ChromaDB** for document storage and the **Ollama language model** for natural language processing, this system allows users to efficiently add, classify, and query concert information.

### Key Features:

- **Document Management**: Users can add concert-related documents to a ChromaDB collection, enabling organized storage and retrieval.
- **Intent Classification**: The system analyzes user input to determine whether the intent is to add a document or ask a question, streamlining user interactions.
- **Concert Relevance Detection**: It checks if the provided text is related to concerts, tours, or performers, ensuring that only relevant information is stored.
- **RAG Pipeline**: The system retrieves relevant documents based on user queries and generates context-aware responses using the Ollama language model.

## Getting started

### Run locally

1. Clone the **RAG-POC** repository in your desired directory:
    ```bash
   $> git clone https://github.com/iccaka/RAG-POC.git
   ```
2. Run these commands in order:
    ```bash
   # create a new virtual environment
   $> python3 -m venv venv

   # start the virtual environment
   $> source /venv/bin/activate
   
   # install project dependencies
   $> pip3 install -r requirements.txt
   ```

Tested on python version **3.10.12**

## Dependencies

Please refer to [requirements.txt](requirements.txt) for a list of the python module dependencies.

## Authors

* **Hristo Mitsev** - *Initial work* - [iccaka](https://github.com/iccaka)

See also the list of [contributors](https://github.com/iccaka/RAG-POC/graphs/contributors) who participated 
in this project.

## Built With

* [PyCharm](https://www.jetbrains.com/pycharm/) - *The IDE used*

## License

This project is licensed under the MIT License - *see the* 
[LICENSE](https://github.com/iccaka/RAG-POC/blob/master/LICENSE.md) *file for details.*
