import io
import streamlit as st
from PyPDF2 import PdfReader
import os
import glob
import pandas as pd
from pypdf import PdfReader
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer
import tqdm
import pinecone
import io

st.title('Upload PDF in Pinecone')

def process_pdf_files(pdf_file):
    # Read the contents of the uploaded file
    pdf_bytes = pdf_file.read()

    # Use BytesIO to create an in-memory binary stream
    with io.BytesIO(pdf_bytes) as pdf_stream:
        reader = PdfReader(pdf_stream)
        text = ""
        for page in reader.pages:
            text += page.extract_text()

    return text

# Create a file uploader and search box
uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True)


pdf_name = []
for f in uploaded_files:
    #st.write(f.name)
    pdf_name.append(f.name)
    

data_list = []

# Process uploaded files and filter by search query
if uploaded_files:
    for pdf_file in uploaded_files:
        text = process_pdf_files(pdf_file)
        #st.write(text)
        data_list.append({"knowledge": pdf_file.name, "data": text})
	#st.write(pdf_file)

# Create a pandas dataframe
    df = pd.DataFrame(data_list)

# Write the dataframe to a CSV file
    df.to_csv('data.csv', index=False)

    data = pd.read_csv('data.csv')



# set device to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
# load the retriever model from huggingface model hub
    retriever = SentenceTransformer("flax-sentence-embeddings/all_datasets_v3_mpnet-base", device=device)


    # Load the CSV data into a pandas dataframe
    df = pd.read_csv("data.csv")

    # Convert the data into a list of strings
    data = df['data'].tolist()

    # Use the loaded retriever to get the embeddings of the data
    embeddings = retriever.encode(data)

# Get the vector dimension
    vector_dim = embeddings.shape[1]

    st.write(f"The vector dimension of the data is: {vector_dim}")

up_pine = st.button('Upload to pinecone')

if up_pine:
    # connect to pinecone environment
    pinecone.init(
        api_key="b9447d59-d2e5-466c-a04d-e30d91ff6db2", #41b0a66d-ea54-4356-9ea6-98cf4208b85f
        environment="us-west4-gcp"  # find next to API key in console
    )
    index_name = "meq-quick-start-guide"
    if index_name not in pinecone.list_indexes():
        # if does not exist, create index
        pinecone.create_index(
            index_name,
            dimension=vector_dim,
            metric='cosine',
            metadata_config=None
        )
    # connect to index
    index = pinecone.Index(index_name)
    st.write('Index Created')
    # view index stats
    #index.describe_index_stats()

    #!pip install tqdm



    batch_size = 64
    
    df = df.dropna()

    for i in range(0, len(df), batch_size):
        # find end of batch
        i_end = min(i+batch_size, len(df))
        # extract batch
        batch = df.iloc[i:i_end]
        # generate embeddings for batch
        emb = retriever.encode(batch["data"].tolist()).tolist()
        # get metadata
        meta = batch.to_dict(orient="records")
        # create unique IDs
        ids = [f"{idx}" for idx in range(i, i_end)]
        # add all to upsert list
        to_upsert = list(zip(ids, emb, meta))
        # upsert/insert these records to pinecone
        _ = index.upsert(vectors=to_upsert)
        # find end of
    st.write('Upload Complete')
    st.write("All uploaded, Thanks")
    def query_pinecone(query, top_k):
        # generate embeddings for the query
        xq = retriever.encode([query]).tolist()
        # search pinecone index for context passage with the answer
        xc = index.query(xq, top_k=top_k, include_metadata=True)
        return xc



#data = pd.read_csv('data.csv')



# set device to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# load the retriever model from huggingface model hub
retriever = SentenceTransformer("flax-sentence-embeddings/all_datasets_v3_mpnet-base", device=device)


    # Load the CSV data into a pandas dataframe
df = pd.read_csv("data.csv")

    # Convert the data into a list of strings
data = df['data'].tolist()

    # Use the loaded retriever to get the embeddings of the data
embeddings = retriever.encode(data)

vector_dim = embeddings.shape[1]
index_name = "meq-quick-start-guide"
pinecone.init(
    api_key="b9447d59-d2e5-466c-a04d-e30d91ff6db2", #41b0a66d-ea54-4356-9ea6-98cf4208b85f
    environment="us-west4-gcp"  # find next to API key in console
    )
index = pinecone.Index(index_name)

# Get the vector dimension

def query_pinecone(query, top_k):
    # generate embeddings for the query
    xq = retriever.encode([query]).tolist()
    # search pinecone index for context passage with the answer
    xc = index.query(xq, top_k=top_k, include_metadata=True)
    return xc

def format_query(query, context):
        # extract passage_text from Pinecone search result and add the <P> tag
    context = [f"<P> {m['metadata']['passage_text']}" for m in context]
        # concatinate all context passages
    context = " ".join(context)
    # contcatinate the query and context passages
    query = f"question: {query} context: {context}"
    return query  
q = st.button("Generate Query")
if q:
    search_query = st.text_input("what knowledge you have? One time memory")
    st.write("Uploaded PDFS name")
    st.write(pdf_name)
   
        #st.write(result)

    
     