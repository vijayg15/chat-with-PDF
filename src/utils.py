import os, json
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings


def load_pdfs(path):
    loader = PyPDFDirectoryLoader(path,
                                  glob = '**/[!.]*.pdf',
                                  extract_images = False
                                  )
    documents = loader.load()
    return documents


def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


def download_embeddings():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        #dimensions=687
    )
    return embeddings


def update_json(data, file_name):
    if os.path.exists(file_name):
        with open(file_name, 'r') as file:
            loaded = json.load(file)
            loaded.append(data)
    else:
        loaded = [data]
            
    with open(file_name,"w") as f:
        json.dump(loaded, f, indent=2)
        f.close()   