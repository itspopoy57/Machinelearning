'''
vector
'''

##vector databases
### efficient data processing LLMs
#databases for vector database into numeric representation
#free( pinecone, chroma, milvus, qdrant)

#sql vs vector database
        #vector database uses the pipeline ANN Approximate Nearest Neigbor

##Splitting and Embedding Text Using Lang Chain

import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

from langchain.text_splitter import RecursiveCharacterTextSplitter
with open('test/WWITEMS.txt', encoding="utf8") as f:    
    wwitems = f.read()
###
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len
)
###
chunks = text_splitter.create_documents([wwitems])
print(chunks[2])

###

#print(chunks[20].page_content)
#each chunks are
print(f'Now you have {len(chunks)}')

###
###Embedding Cost
def print_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    print(f'Total Tokens: {total_tokens}')
    print(f'Embedding Cost in USD: {total_tokens/1000 * 0.0004:.6f}')
    
print_embedding_cost(chunks)


###

from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

###
#converting to vector

vector = embedding.embed_query(chunks[0].page_content)
print(vector)

### i used jupyter notebook thats why
#!pip install pinecone-client
import os
import pinecone
from langchain.vectorstores import Pinecone

pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment=os.environ.get('PINECONE_ENV'))

#deleting all the indexes
indexes = pinecone.list_indexes()
for i in indexes:
    print('deleting all indexes')
    pinecone.delete_index(i)
    print('Done')


###
index_name = 'wwitems'
if index_name not in pinecone.list_indexes():
    print(f'Creating index {index_name} ...')
    pinecone.create_index(index_name, dimension=1536, metric='cosine')
    print('done!')

vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)

##asking questionss (similarity search)
###
query = "Display Screen"
result = vector_store.similarity_search(query)
print(result)

##
for r in result:
    print(r.page_content)
    print('-' * 50)

###

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':3})

chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)


###

#query = "can you explain to me this document IBM RPG code"
query = ""
answer = chain.run(query)
print(answer)


                            





