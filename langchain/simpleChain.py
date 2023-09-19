from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain

import os
from dotenv import load_dotenv, find_dotenv
load_dotenv("./.env", override=True)
apis = os.environ.get('OPENAI_API_KEY')
#i put the api keys to .env

llm = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature=0.5, openai_api_key=apis)
template = ''' You are an Experienced virologist.
write a few sentences about the following {virus} in {language}.'''

prompt = PromptTemplate(
    input_variables=['virus', 'language'],
    template = template)

chain = LLMChain(llm = llm, prompt=prompt)
output = chain.run({'virus':'amoeba', 'language': 'french'})
#is you need only the var 1 which is the virus and dont care about the langauge
#output = chain.run('amoeba')

  print(output)
