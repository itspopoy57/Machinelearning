# embedding uploading to vector database ''
def insert_of_fetch_embeddings(index_name):
    import pinecone
    from langchain.vectorstores import Pinecone
    from langchain.embeddings.openai import OpenAIEmbeddings
    
    embeddings = OpenAIEmbeddings()
    pinecone.ini(api_key=os.environ.get('PINECONE_API_KEY'), environment=os.environ.get('PINECONE_ENV'))
    
    if index_name in pinecone.list_indexes():
        print(f'Index {index_name} already exist. Loading embeddings', end='')
        vector_store = Pinecone.from_existing_index(index_name, embeddings)
        print('Ok')
    
    #creating
    else:
        print(f'Creating index {index_name} and embeddings ...', end='')
        pinecone.create_index(index_name, dimension=1536, metric='cosine')
        vector_store = Pinecone.from_documents(chunks, embeddings, index_name)
        print('Ok')
        
    return vector_store

###chunkin
def chunk_data(data, chunk_size=256):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    chunks = text_splitter.split_documents(data)
    return chunks



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

        
