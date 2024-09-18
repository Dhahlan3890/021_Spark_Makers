# PDF Question Answering System

This project is a PDF Question Answering System that uses Streamlit, Google's Generative AI, and LangChain to create an interactive application for querying information from PDF documents.

## Features

- Upload and process PDF documents
- Extract text content from PDFs
- Use Google's Generative AI for question answering
- Interactive Streamlit interface

## Installation

1. Clone this repository
2. Install the required dependencies:


pip install -r requirements.txt

3. Set up your environment variables:
   - Create a `.env` file in the project root
   - Add your Google API key: `GOOGLE_API_KEY=your_api_key_here`

## Usage

Run the Streamlit app:


streamlit run app.py

Navigate to the provided local URL in your web browser to use the application.

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
- `read_pdf.py`: PDF processing and text extraction
- `requirements.txt`: List of project dependencies

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).