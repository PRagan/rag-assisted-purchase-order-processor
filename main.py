"""
RAG Project: PDF Processing with Pinecone Vector Database
Author: Philip Ragan
Description: Downloads PDFs, extracts text with EasyOCR, stores in Pinecone as a vector, then answers questions using OpenAI
"""

import os
import requests
import tempfile
import uuid
from pathlib import Path
from typing import List, Dict, Any
import logging

# PDF and OCR imports
import fitz  # PyMuPDF
import easyocr
from PIL import Image
import io

# LangChain imports
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA

# Pinecone
import pinecone
from pinecone import Pinecone

# Environment variables
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipelineProcessor:
    """Main class for processing PDFs and implementing RAG pipeline"""
    
    def __init__(self):
        """Initialize the RAG processor with API keys and configurations"""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "rag-pdf-index")
        
        if not self.openai_api_key or not self.pinecone_api_key:
            raise ValueError("Please set OPENAI_API_KEY and PINECONE_API_KEY environment variables")
        
        # Initialize OCR reader
        self.ocr_reader = easyocr.Reader(['en'])
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=self.openai_api_key,
            model="text-embedding-3-large",
            dimensions=1024
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.vector_store = None
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            openai_api_key=self.openai_api_key,
            model_name="gpt-3.5-turbo",
            temperature=0
        )
        
    def setup_pinecone_index(self, dimension: int = 1536):
        """Setup or connect to Pinecone index"""
        try:
            # Check if index exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=dimension,
                    metric="cosine",
                    spec={
                        "serverless": {
                            "cloud": "aws",
                            "region": "us-east-1"
                        }
                    }
                )
            else:
                logger.info(f"Using existing Pinecone index: {self.index_name}")
                
        except Exception as e:
            logger.error(f"Error setting up Pinecone index: {e}")
            raise
    
    def download_pdf(self, url: str, filename: str = None) -> str:
        """Download PDF from URL and return local file path"""
        try:
            logger.info(f"Downloading PDF from: {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            if not filename:
                filename = f"pdf_{uuid.uuid4().hex[:8]}.pdf"
            
            # Create temp directory if it doesn't exist
            temp_dir = Path("temp_pdfs")
            temp_dir.mkdir(exist_ok=True)
            
            file_path = temp_dir / filename
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"PDF downloaded successfully: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error downloading PDF from {url}: {e}")
            raise
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using PyMuPDF and EasyOCR for images"""
        try:
            logger.info(f"Extracting text from: {pdf_path}")
            doc = fitz.open(pdf_path)
            extracted_text = ""
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                
                # First, try to extract text directly
                text = page.get_text()
                
                if text.strip():
                    extracted_text += f"\n--- Page {page_num + 1} ---\n{text}\n"
                else:
                    # If no text found, use OCR on page image
                    logger.info(f"No text found on page {page_num + 1}, using OCR...")
                    
                    # Convert page to image
                    pix = page.get_pixmap()
                    img_data = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_data))
                    
                    # Use EasyOCR to extract text
                    ocr_results = self.ocr_reader.readtext(img_data)
                    ocr_text = " ".join([result[1] for result in ocr_results])
                    
                    if ocr_text.strip():
                        extracted_text += f"\n--- Page {page_num + 1} (OCR) ---\n{ocr_text}\n"
            
            doc.close()
            logger.info(f"Text extraction completed. Total characters: {len(extracted_text)}")
            return extracted_text
            
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            raise
    
    def process_pdfs(self, pdf_urls: List[str]) -> List[Document]:
        """Download and process multiple PDFs"""
        all_documents = []
        
        for i, url in enumerate(pdf_urls):
            try:
                # Download PDF
                pdf_path = self.download_pdf(url, f"document_{i+1}.pdf")
                
                # Extract text
                text = self.extract_text_from_pdf(pdf_path)
                
                if text.strip():
                    # Create document with metadata
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": url,
                            "document_id": i + 1,
                            "file_path": pdf_path
                        }
                    )
                    all_documents.append(doc)
                
                # Clean up downloaded file (optional)
                # os.remove(pdf_path)
                
            except Exception as e:
                logger.error(f"Failed to process PDF {url}: {e}")
                continue
        
        return all_documents
    
    def create_vector_store(self, documents: List[Document]):
        """Split documents and create vector store in Pinecone"""
        try:
            logger.info("Splitting documents into chunks...")
            
            # Split documents
            split_docs = self.text_splitter.split_documents(documents)
            logger.info(f"Created {len(split_docs)} document chunks")
            
            # Setup Pinecone index with correct dimension for text-embedding-3-large
            self.setup_pinecone_index(dimension=1024)
            
            # Create vector store
            logger.info("Creating embeddings and storing in Pinecone...")
            self.vector_store = PineconeVectorStore.from_documents(
                documents=split_docs,
                embedding=self.embeddings,
                index_name=self.index_name
            )
            
            logger.info("Vector store created successfully!")
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise
    
    def setup_qa_chain(self):
        """Setup the question-answering chain"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Please create vector store first.")
        
        # Create retriever
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 4}  # Retrieve top 4 similar chunks
        )
        
        # Create custom prompt template
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

        Context: {context}

        Question: {question}
        
        Answer: """
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        qa_chain = None
        try:
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
        except Exception as e:
            logger.error(f"Error setting up QA chain: {e}")
            raise
        
        return qa_chain
    
    def answer_questions(self, questions: List[str]) -> Dict[str, Any]:
        """Answer a list of questions using the RAG pipeline"""
        qa_chain = self.setup_qa_chain()
        results = {}
        
        for i, question in enumerate(questions, 1):
            try:
                logger.info(f"Processing question {i}: {question}")
                
                result = qa_chain.invoke({"query": question})
                
                results[f"question_{i}"] = {
                    "question": question,
                    "answer": result["result"],
                    "source_documents": [
                        {
                            "content": doc.page_content[:200] + "...",
                            "metadata": doc.metadata
                        }
                        for doc in result["source_documents"]
                    ]
                }
                
            except Exception as e:
                logger.error(f"Error answering question '{question}': {e}")
                results[f"question_{i}"] = {
                    "question": question,
                    "answer": f"Error: {str(e)}",
                    "source_documents": []
                }
        
        return results


def main():
    """Main function to run the RAG pipeline"""

    # Get base URL from environment variable
    pdf_url = os.getenv("PDF_URL", "https://hook.example.com/")
    
    # Example PDF URLs (replace with your actual URLs)
    pdf_urls = [
        f"{pdf_url}",
    ]
    
    # Example questions
    questions = [
        "What is the date of the purchase order?",
        "Which company sent the purchase order?",
        "How many items are in the purchase order?",
        "What is the total value of the items in the purchase order?"
    ]
    
    try:
        # Initialize processor
        processor = RAGPipelineProcessor()
        
        # Process PDFs
        logger.info("Starting PDF processing...")
        documents = processor.process_pdfs(pdf_urls)
        
        if not documents:
            logger.error("No documents were successfully processed!")
            return
        
        logger.info(f"Successfully processed {len(documents)} documents")
        
        # Create vector store
        processor.create_vector_store(documents)
        
        # Answer questions
        logger.info("Starting question answering...")
        results = processor.answer_questions(questions)
        
        # Print results
        print("\n" + "="*80)
        print("QUESTION ANSWERING RESULTS")
        print("="*80)
        
        for key, result in results.items():
            print(f"\nQ: {result['question']}")
            print(f"A: {result['answer']}")
            print(f"Sources: {len(result['source_documents'])} document(s)")
            print("-" * 40)
        
        logger.info("RAG pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()