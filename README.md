# Election Helper App

This project is an Election Helper App that uses Streamlit, Google's Generative AI, and LangChain to create an interactive platform for predicting election outcomes, comparing candidates, and answering questions based on PDF documents.

## Features

- **Upload and Process PDFs**: Extract detailed candidate information from uploaded PDFs
- **Predict Election Outcome**: Use advanced models to predict the winning candidate through the `predict.py` component
- **Compare Candidates**: Compare candidates' profiles, policies, and more with the `compare.py` component
- **Interactive Question Answering**: Ask questions and receive answers based on the content of PDFs using the `read_pdf.py` component
- **Google's Generative AI Integration**: Enhanced natural language processing for more accurate responses

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Set up your environment variables:
   - Create a `.env` file in the project root
   - Add your Google API key: `GOOGLE_API_KEY=your_api_key_here`

## Usage

1. Run the Streamlit app:

```bash
streamlit run app.py
```

2. Use the following components for their respective tasks:
   - **Election Outcome Prediction**: `predict.py`
   - **Candidate Comparison**: `compare.py`
   - **PDF-based Question Answering**: `read_pdf.py`

3. Navigate to the provided local URL in your web browser to use the application.

## Dependencies

- streamlit
- google-generativeai
- python-dotenv
- langchain
- PyPDF2
- chromadb
- faiss-cpu
- langchain_google_genai

## Project Structure

- `app.py`: Main Streamlit application
- `predict.py`: Election outcome prediction component
- `compare.py`: Candidate comparison component
- `read_pdf.py`: PDF processing, text extraction, and question answering component
- `requirements.txt`: List of project dependencies

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).
