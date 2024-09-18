import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=4000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    You are a highly knowledgeable assistant. Your goal is to answer the user's question accurately based on the provided context. If the answer is found in the context, respond clearly. If not, say, "The answer isn't found in the provided data, but here is what I found based on the closest match."\n\n
    Context:\n {context}\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(questions):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    table = []

    chain = get_conversational_chain()

    for i in range(len(st.session_state.pdf_docs)):
        new_db = FAISS.load_local(f"faiss_index_{i}", embeddings)

        for question in questions:
            # Use similarity_search_with_score to get results with scores
            docs_with_scores = new_db.similarity_search_with_score(question, k=5)
            # Filter documents with a lower threshold score to include partial matches
            docs = [doc for doc, score in docs_with_scores if score > 0.3]

            # If no docs match the score threshold, still provide a response
            if not docs:
                docs = [docs_with_scores[0][0]]  # Take the most similar document

            response = chain(
                {"input_documents": docs, "question": question}
                , return_only_outputs=True)
            response = response["output_text"].replace("The answer isn't found in the provided data, but here is what I found based on the closest match", "closest match")
            print(f"{question}: {response}")
            table.append({"Details": question, f"Candidate {i+1}": response})

    df = pd.DataFrame(table)
    df = df.groupby('Details', as_index=False).agg(lambda x: ' '.join(x.dropna().astype(str)))

    st.table(df)



def main():
    st.set_page_config("Election helper")
    st.header("Compare Candidates")

    user_question = st.text_input("Ask what you want to compare")

    if user_question:
        if 'pdf_docs' in st.session_state and st.session_state.pdf_docs:
            with st.spinner("Processing..."):
                user_input([f"{user_question} of the candidate"])
        else:
            st.warning("Please upload PDF documents first.")

    questions = [
        "What is the full name of the candidate",
        "What is the age of or how old the candidate is",
        "What are the name of the spouse of the candidate",
        "How many children does the candidate have",
        "What are the name of the parents of the candidate",
        "What are the name of the siblings of the candidate",
        "What is the educational background of the candidate",
        "What are the previous roles or positions the candidate held",
        "What political actions or contributions has the candidate made previously",
        "What are the things he will do id elected or key initiatives or promises the candidate has made if elected"
    ]

    if st.button("Compare Everything"):
        if 'pdf_docs' in st.session_state and st.session_state.pdf_docs:
            with st.spinner("Processing..."):
                user_input(questions)
        else:
            st.warning("Please upload PDF documents first.")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your Data and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    st.session_state.pdf_docs = pdf_docs
                    for i, pdf in enumerate(pdf_docs):
                        raw_text = get_pdf_text([pdf])
                        text_chunks = get_text_chunks(raw_text)
                        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
                        vector_store.save_local(f"faiss_index_{i}")
                    st.success("Done")
            else:
                st.warning("Please upload PDF documents before processing.")

if __name__ == "__main__":
    main()