import os

import chromadb
from rag import ChromaDBEmbeddingFunction
from langchain_ollama import OllamaEmbeddings, OllamaLLM

llm_model = 'qwen:1.8b'


def add_docs_to_collection(documents, ids):
    collection.add(
        documents=documents,
        ids=ids
    )


def query_chromadb(query_text, n_results=1):
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )

    return results['documents'], results['metadatas']


def query_ollama(prompt, llm_model=llm_model):
    llm = OllamaLLM(model=llm_model)

    return llm.invoke(prompt)


def rag_pipeline(query_text):
    retrieved_docs, metadata = query_chromadb(query_text)
    context = ' '.join(retrieved_docs[0]) if retrieved_docs else 'No relevant documents found.'

    augmented_prompt = f"Context: {context}\n\nQuestion: {query_text}\nAnswer:"
    print('==========Augmented Prompt==========')
    print(augmented_prompt)

    return query_ollama(augmented_prompt, llm_model)


if __name__ == '__main__':
    chroma_client = chromadb.PersistentClient(path=os.path.join(os.getcwd(), 'chroma_db'))

    embedding = ChromaDBEmbeddingFunction(
        OllamaEmbeddings(
            model=llm_model,
            base_url='http://localhost:11434'
        )
    )

    collection_name = 'collection_1'
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={'description': 'demo1'},
        embedding_function=embedding
    )

    documents = [
        'Lady Gaga has announced her Chromatica II Tour for 2025, starting in September. Key venues include Madison '
        'Square Garden (New York) on September 18, and the United Center (Chicago) on September 23. She will be joined '
        'by Rina Sawayama as a special guest for select North American dates.',
        'Beyoncé\'s Renaissance World Tour will cover 15 cities across Europe in Spring 2026. Logistics teams are '
        'coordinating with local crews for stage setup, especially in Paris and Berlin, where venue capacities exceed '
        '70,000. Custom light rigs and portable dance floors are being shipped from LA.',
        'The O2 Arena will host multiple events during Winter 2025. It has a seating capacity of 20,000 and recently '
        'upgraded its sound system. Coldplay and Dua Lipa are scheduled for back-to-back concerts in December.',
        'Coldplay\’s 2026 World Tour will feature rotating guest appearances. Notable names include Sigrid, Mitski, and '
        'Stromae, depending on the continent. The band will play five nights at Foro Sol in Mexico City in March.',
        'An addendum to the Eras Tour 2025 includes three new dates in Australia: Sydney (Feb 3), Brisbane (Feb 6), '
        'and Melbourne (Feb 10). These are in addition to the previously announced Asia-Pacific leg. Venues confirmed '
        'include Accor Stadium and Marvel Stadium.',
        'The Southeast Asia leg of The Weeknd\'s tour in early 2026 includes shows in Bangkok, Kuala Lumpur, and Manila.'
        ' Due to customs delays, the logistics team is advised to pre-clear lighting equipment two weeks in advance and'
        ' confirm stage dimensions with each venue manager.',
        'Metallica’s "WorldWired II" Fall Tour 2025 includes stops in Detroit, Seattle, and Phoenix. The tour emphasizes'
        ' minimal setup time — under 6 hours per venue. All dates use a rotating stage layout, and tuning sessions are '
        'scheduled two hours before doors open.',
        'Billie Eilish will headline major European festivals in July 2025, including Roskilde (Denmark), Mad Cool '
        '(Spain), and Open’er (Poland). Her team requested environmentally friendly transport options for gear, and '
        'sustainable merch partnerships are underway.'
    ]
    doc_ids = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5', 'doc6', 'doc7', 'doc8']
    add_docs_to_collection(documents, doc_ids)

    while True:
        try:
            user_input = input('(Use \'exit\' to exit.)\nEnter query: ')

            if user_input == 'exit':
                break
            else:
                print('==========Response from LLM==========\n', rag_pipeline(user_input))
        except ValueError:
            print('Invalid input, please try again.')
            continue
