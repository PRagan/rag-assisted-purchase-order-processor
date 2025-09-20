# PDF Purchase Order Processor

A sophisticated PDF processing system that combines OCR, Retrieval Augmented Generation (RAG), and Large Language Models to analyze purchase orders and answer questions about their content.

## Core Features

### 🔍 **Advanced PDF Processing**
- **Multi-format Text Extraction**: Combines PyMuPDF for direct text extraction and EasyOCR for image-based text recognition
- **Intelligent OCR Fallback**: Automatically switches to OCR when direct text extraction fails
- **Batch Processing**: Handles multiple PDF documents simultaneously

### 🧠 **RAG-Powered Intelligence**
- **Vector Database Integration**: Uses Pinecone for efficient similarity search and document retrieval
- **Smart Text Chunking**: Implements recursive character text splitting with configurable overlap
- **Context-Aware Responses**: Leverages retrieved document chunks to provide accurate, source-backed answers

### 💬 **Natural Language Querying**
- **Purchase Order Analysis**: Extracts key information like dates, companies, items, and totals
- **Question Answering**: Responds to natural language queries about document content
- **Source Attribution**: Provides references to specific document sections for transparency

### 🔧 **Enterprise-Ready Architecture**
- **Configurable Settings**: Centralized configuration management via `config.py`
- **Environment Variables**: Secure API key and settings management
- **Comprehensive Logging**: Detailed logging for monitoring and debugging
- **Error Handling**: Robust error handling with graceful degradation

## Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key
- Pinecone API key

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd rag-assisted-purchase-order-processor
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   PINECONE_ENVIRONMENT=us-east-1-aws
   PINECONE_INDEX_NAME=rag-pdf-index
   PDF_URL=https://your-pdf-url.com/document.pdf
   ```

4. **Run the application**
   ```bash
   python main.py
   ```

## Usage

### Basic Usage

The system processes PDF purchase orders and answers predefined questions:

```python
from main import RAGPipelineProcessor

# Initialize the processor
processor = RAGPipelineProcessor()

# Process PDF documents
pdf_urls = ["https://example.com/purchase-order.pdf"]
documents = processor.process_pdfs(pdf_urls)

# Create vector store
processor.create_vector_store(documents)

# Ask questions
questions = [
    "What is the date of the purchase order?",
    "Which company sent the purchase order?",
    "How many items are in the purchase order?",
    "What is the total value of the items?"
]

results = processor.answer_questions(questions)
```

### Configuration Options

Customize the system behavior by modifying `config.py`:

```python
# Text processing settings
chunk_size: int = 1000          # Size of text chunks
chunk_overlap: int = 200        # Overlap between chunks

# Retrieval settings
retrieval_k: int = 4            # Number of documents to retrieve

# LLM settings
llm_model: str = "gpt-3.5-turbo"
llm_temperature: float = 0.0    # Response creativity (0 = deterministic)
```

### Supported Question Types

The system can answer various types of questions about purchase orders:

- **Date Information**: "When was this order placed?", "What's the delivery date?"
- **Company Details**: "Who is the supplier?", "What's the buyer's company?"
- **Item Analysis**: "List all items in the order", "What quantities were ordered?"
- **Financial Data**: "What's the total amount?", "Show me the unit prices"
- **General Queries**: "Summarize this purchase order", "Are there any special terms?"

### Advanced Features

#### Custom PDF Processing
```python
# Process local PDF files
processor = RAGPipelineProcessor()
text = processor.extract_text_from_pdf("path/to/local/file.pdf")
```

#### Vector Store Management
```python
# Setup custom Pinecone index
processor.setup_pinecone_index(dimension=1024)

# Create vector store with custom settings
processor.create_vector_store(documents)
```

#### Question-Answer Chain Customization
```python
# Setup QA chain with custom retrieval parameters
qa_chain = processor.setup_qa_chain()
result = qa_chain.invoke({"query": "Your custom question"})
```

### Output Format

The system returns structured results with source attribution:

```json
{
  "question_1": {
    "question": "What is the date of the purchase order?",
    "answer": "The purchase order is dated March 15, 2024.",
    "source_documents": [
      {
        "content": "Purchase Order #12345 dated March 15, 2024...",
        "metadata": {
          "source": "https://example.com/po.pdf",
          "document_id": 1,
          "file_path": "temp_pdfs/document_1.pdf"
        }
      }
    ]
  }
}
```

### Error Handling

The system includes comprehensive error handling:
- Invalid PDF URLs or corrupted files
- OCR processing failures
- API rate limiting and connectivity issues
- Vector database connection problems

Check the logs for detailed error information and troubleshooting guidance.

## License
This project was originally created to demonstrate my capacity to developed enterprise ready AI applications.