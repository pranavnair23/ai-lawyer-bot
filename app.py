from dotenv import load_dotenv
import streamlit as st
from langchain.chains import RetrievalQA
#from langchain.llms import HuggingFaceHub
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Qdrant
import qdrant_client
import os
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, user_template, bot_template
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
import streamlit as st


def get_vector_store():
    
    client = qdrant_client.QdrantClient(
        os.getenv("QDRANT_HOST"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    
    embeddings = embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

    vector_store = Qdrant(
        client=client, 
        collection_name="PenalCode", 
        embeddings=embeddings,
    )
    
    return vector_store

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    #llm = HuggingFaceHub( repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm =llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question':user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i%2 == 0:
            st.write(user_template.replace("{{MSG}}",message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}",message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    
    st.set_page_config(page_title="Legal Assistant", page_icon=":robot_face:")
    st.write(css, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center; color: red;font-family:Georgia'>AI Lawyer Bot 🤖</h1>", unsafe_allow_html=True)
    st.subheader("\"_Is that legal❓_\"")
    st.write("This bot is made to answer all your legal queries in the context of the Indian Penal Code.")
    with st.expander("**Disclamer**"):
        st.write("1. **This is not legal advice**.")
        st.write("2. While the model has the context of the IPC it has not been fine-tuned and hence may not be able to answer all your queries. ")
    st.divider()    
    st.caption("Try something like \"What is the punishment for criminal intimidation?\" or \"How is theft defined in the IPC?\"")

    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None 
    
    # create vector store
    vector_store = get_vector_store()

    st.session_state.conversation = get_conversation_chain(vector_store)
    
    user_question = st.text_input("Ask your questions here:")  
    if user_question:
        handle_userinput(user_question)
    
    
        
if __name__ == '__main__':
    main()
