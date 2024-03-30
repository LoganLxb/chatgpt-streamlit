from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
import gradio
import os

os.environ["OPENAI_API_KEY"] = ''

def construct_index(directory_path):
    # set number of output tokens
    Settings.llm = OpenAI(model="gpt-3.5-turbo")
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
    Settings.num_output = 512
    Settings.context_window = 3900

    docs = SimpleDirectoryReader(directory_path).load_data()

    index = VectorStoreIndex.from_documents(docs, embed_model=Settings.embed_model,llm=Settings.llm)
    
    #Directory in which the indexes will be stored
    index.storage_context.persist(persist_dir="indexes")

    return index

def chatbot(input_text):
    
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="indexes")
    
    #load indexes from directory using storage_context 
    query_engne = load_index_from_storage(storage_context).as_query_engine()
    
    response = query_engne.query(input_text)
    
    #returning the response
    return response.response

#Creating the web UIusing gradio
iface = gradio.Interface(fn=chatbot,
                     inputs=gradio.Textbox(lines= 5 , label= "Enter your questions here" ), 
                     outputs="text",
                     title="Custom-trained AI Chatbot")

#Constructing indexes based on the documents in traininData folder
#This can be skipped if you have already trained your app and need to re-run it
index = construct_index("trainingData")

#launching the web UI using gradio
iface.launch(share=True)
