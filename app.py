import os
import streamlit as st
from dotenv import load_dotenv
from src.utils import load_pdfs, text_split, download_embeddings, update_json
from src.prompt import *
from streamlit_extras.add_vertical_space import add_vertical_space

from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.callbacks.manager import get_openai_callback



# Sidebar contents
with st.sidebar:
    st.title('💬 LLM Chat App') 
    st.markdown('''
                ## About
                This app is an LLM-powered chat-bot and built using:
                - [LangChain](https://python.langchain.com/)
                - [OpenAI LLM model](https://platform.openai.com/docs/models) 
                - [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
                - [FAISS](https://faiss.ai/index.html)
                - [Streamlit](https://streamlit.io/)
                '''
                )
    add_vertical_space(6)
    st.write('Made with ❤️ by [Vijay Kumar Gupta](https://www.linkedin.com/in/vijayiitk/)')


def main():
    st.title("Chat with your own PDF 💬")
    st.subheader("Drag and drop your pdf and ask the questions!")

    load_dotenv()
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

    pdf = st.file_uploader("Upload your pdf", type='pdf')
    
    if pdf is not None:

        save_folder = 'data/'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_path = os.path.join(save_folder, pdf.name)
        with open(save_path, mode='wb') as w:
            w.write(pdf.getvalue())

        extracted_data = load_pdfs(save_folder)
        text_chunks = text_split(extracted_data)

        if os.path.isfile(save_path):
            os.remove(save_path)

        embeddings = download_embeddings()

        docsearch = FAISS.from_documents(text_chunks, embeddings)
        #st.write('Vectors created and stored')
        
        retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":5})
        llm = ChatOpenAI(
            model = "gpt-4o-mini",
            temperature=0.6
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
                ]
                )
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        file_name = 'responses.json'
        
        # Accept user' questions/query
        query = st.text_input("Ask a question about your PDF file:")
        if query:
            print(query)
            with get_openai_callback() as cb:
                response = rag_chain.invoke({"input": str(query)})
                print(cb)
            print("Response : ", response["answer"])
            st.write("Response : ", response["answer"])
            data = {"Input": response["input"], "Response": response["answer"]}
            update_json(data, file_name)



if __name__ == '__main__':
    main()