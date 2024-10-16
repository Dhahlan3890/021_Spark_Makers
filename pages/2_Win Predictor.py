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
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from scipy.stats import norm

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context say, "It looks like it is not in the data" then anwer from your knowledge, don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])

def check_positivity(comments):
    model = genai.GenerativeModel("gemini-1.5-flash")
    positivity = 0
    for comment in comments:
        response = model.generate_content(f"{comment}\n check if the comment is positive or nagative. if positive return 1, if negative return -1, if neutral return 0. only tell the output in the format of 1, -1, 0")
        print(response.text)
        if "1" in response.text and "-1" not in response.text and "0" not in response.text:
            positivity += 1
        elif "-1" in response.text and "0" not in response.text:
            positivity -= 1

    return positivity

def check_positivity_array(comments):
    model = genai.GenerativeModel("gemini-1.5-flash")
    positivity = []
    for comment in comments:
        response = model.generate_content(f"{comment}\n check if the comment is positive or nagative. if positive return 1, if negative return -1, if neutral return 0. only tell the output in the format of 1, -1, 0")
        print(response.text)
        if "1" in response.text and "-1" not in response.text and "0" not in response.text:
            positivity.append(1)
        elif "-1" in response.text and "0" not in response.text:
            positivity.append(-1)

    return positivity

def calculate_z_value(sentiment_list, expected_proportion=0.5):
    # Step 1: Calculate sample size and sample proportion
    n = len(sentiment_list)
    num_positive = sum(sentiment_list)
    sample_proportion = num_positive / n

    # Step 2: Calculate standard error
    standard_error = np.sqrt((expected_proportion * (1 - expected_proportion)) / n)

    # Step 3: Calculate Z-value
    z_value = (sample_proportion - expected_proportion) / standard_error

    return z_value

def calculate_probability_from_z(z_value):
    # Calculate probability using cumulative distribution function (CDF) of normal distribution
    probability = norm.cdf(z_value)
    return probability


def predict_vote_percentage(features, model):
    X = np.array([features])
    prediction = model.predict(X)
    return prediction[0]


from sklearn.metrics import mean_squared_error, r2_score

def main():
    st.set_page_config("Election helper")
    st.header("WIN PREDICTOR")

    model = RandomForestRegressor(n_estimators=100, random_state=42)

    with st.sidebar:
        st.title("Menu:")
        xlsx_data = st.file_uploader("Upload your Excel Data and Click on the Submit & Process Button", accept_multiple_files=False)
        if st.button("Submit & Process"):
            if xlsx_data is not None:
                with st.spinner("Processing..."):
                    df = pd.read_excel(xlsx_data)
                    X = df[['How many Main Parties that support', 'No.of Adult voters(%)', 'No.Youth voters(%)', 'New Voters(%)', 'Inflation Rate(%)', 'Unemployement Rate (%)', 'No.of Coverage Districts', 'Spending on Election(milions)', 'comment_positivity_percentage']].values
                    y = df['vote_percentage'].values
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    model.fit(X_train, y_train)
                    st.session_state['model'] = model
                    
                    # Calculate and display model accuracy
                    y_pred = model.predict(X_test)
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    st.write(f"Model Mean Squared Error: {mse:.4f}")
                    st.write(f"Model R-squared: {r2:.4f}")
                    
                    st.success("Model trained successfully!")
            else:
                st.error("Please upload an Excel file before processing.")

    st.subheader("Predict Vote Percentage")
    comments_data = st.file_uploader("Upload your Excel Data of the comments and Click on the Check Positivity Button", accept_multiple_files=False, key="comment_file_uploader_key")
    if st.button("Check Positivity"):
        if comments_data is not None:
            with st.spinner("Processing..."):
                df = pd.read_excel(comments_data)
                df = df.dropna()
                comments = df[['Comment']].values
                positivity = check_positivity(comments)
                positivity_array = check_positivity_array(comments)
                z_value = calculate_z_value(positivity_array)
                probability = calculate_probability_from_z(z_value)
                table = [{"Comment No": comments[i], "Positive/Negative": "positive" if positivity_array[i] == 1 else "negative" if positivity_array[i] == -1 else "neutral" } for i in range(len(comments))]

                df = pd.DataFrame(table)

                st.table(df)

                st.write(f"Positivity Score: {positivity}")
                st.write(f"Positivity percentage {(positivity/len(comments))*100}%")
                st.write(f"Z-Value: {z_value:.4f}")
                st.write(f"Positivity Score using Z-Value: {probability*100:.4f}")
    comment_positivity_percentage = st.slider("Positivity Percentage of Comments", 0, 100, 50)
    main_parties_support = st.number_input("How many major parties support to candidate", min_value=0, value=1)
    adult_voters = st.slider("Percentage of Adult Voters (%)", 0, 100, 50)
    youth_voters = st.slider("Percentage of Youth Voters(%)", 0, 100, 30)
    new_voters = st.slider("Percentage of New Voters for this election(%)", 0, 100, 20)
    inflation_rate = st.slider("Inflation Rate(%)", 0.0, 20.0, 5.0)
    unemployment_rate = st.slider("Unemployment Rate (%)", 0.0, 20.0, 5.0)
    coverage_districts = st.number_input("No. of  districts where the candidate organized election meetings or election promotion programs", min_value=1, value=10)
    election_spending = st.number_input("Election Expenditures (Rs.millions)", min_value=0.0, value=1.0)

    if st.button("Predict Vote Percentage"):
        if 'model' in st.session_state:
            features = [main_parties_support, adult_voters, youth_voters, new_voters, 
                        inflation_rate, unemployment_rate, coverage_districts, election_spending,comment_positivity_percentage]
            vote_percentage = predict_vote_percentage(features, st.session_state['model'])
            st.write(f"Predicted Vote Percentage: {vote_percentage:.2f}%")
        else:
            st.error("Please upload and process the Excel data before making predictions.")

if __name__ == "__main__":
    main()