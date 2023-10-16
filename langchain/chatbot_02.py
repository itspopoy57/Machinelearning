#complete but edit the api

import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
#we need to put the api on the .env

##
#pip install -r ./requirements.txt -q
#pip install pypdf -q
#pip install langchain --upgrade -q
#pip install docx2txt
#pip install wikipedia -q
# these are separated

#funcs


def load_document(file):
    import os
    name, extension = os.path.splitext(file)
    
    
    
    #for pdf files
    
    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
        
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    else:
        print('Document format is not supported')
        return None
        
        
    data = loader.load()
    return data

#wikipedia private public loader
def load_from_wiki(query, lang='en', load_max_docs=2):
    from langchain.document_loaders import WikipediaLoader
    loader = WikipediaLoader(query=query , lang=lang, load_max_docs=load_max_docs)
    data = loader.load()
    return data


###chunkin
def chunk_data(data, chunk_size=256):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    chunks = text_splitter.split_documents(data)
    return chunks


# embedding uploading to vector database ''
def insert_of_fetch_embeddings(index_name):
    import pinecone
    from langchain.vectorstores import Pinecone
    from langchain.embeddings.openai import OpenAIEmbeddings
    
    embeddings = OpenAIEmbeddings()
    pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment=os.environ.get('PINECONE_ENV'))
    
    if index_name in pinecone.list_indexes():
        print(f'Index {index_name} already exist. Loading embeddings', end='')
        vector_store = Pinecone.from_existing_index(index_name, embeddings)
        print('Ok')
    
    #creating
    else:
        print(f'Creating index {index_name} and embeddings ...', end='')
        pinecone.create_index(index_name, dimension=1536, metric='cosine')
        vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
        print('Ok')
         
    return vector_store

def delete_pinecone_index(index_name):
    #better to delete all indexes than just one
    import pinecone
    pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment=os.environ.get('PINECONE_ENV'))
    
    if index_name =='all':
        indexes = pinecone.list_indexes()
        print('Deleting all indexes ...')
        for index in indexes:
            pinecone.delete_index(index)
        print('Ok')
    else:
        print(f'deleting index {index_name} ...', end='')
        pinecone.delete_index(index_name)
        print('Ok')

def ask_and_get_answer(vector_store, q):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI
    
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    
    answer = chain.run(q)
    return answer
    
##adding memory

def ask_with_memory(vector_store, question, chat_history=[]):
    from langchain.chains import ConversationalRetrievalChain
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})

    crc = ConversationalRetrievalChain.from_llm(llm, retriever)
    result = crc({'question': question, 'chat_history': chat_history})

    # Append the question and its answer to chat history
    chat_history.append((question, result['answer']))

    return result['answer'], chat_history


data = load_document('files/ibm1.pdf')
chunks = chunk_data(data)
print(len(chunks))
#chunks 190 portions


#create an index and create chunks and upload the vector

index_name = 'ibm1'
vector_store = insert_of_fetch_embeddings(index_name)
#change pinecode vector database first!! must!!


import time

i = 1
chat_history = []  # Initialize an empty list for chat history

print('Write "Quit" or "Exit" to quit.')
while True:
    q = input(f'Question #{i}: ')
    i = i + 1
    if q.lower() in ['quit', 'exit']:
        print('Quitting... Bye bye!')
        time.sleep(2)
        break

    result, chat_history = ask_with_memory(vector_store, q, chat_history)
    print(f'\nAnswer: {result}')
    print(f'\n{"-" * 50}\n')






