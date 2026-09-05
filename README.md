# Advanced RAG Learning

A hands-on learning repository for understanding and implementing Retrieval-Augmented Generation (RAG), from document ingestion and vector search to advanced retrieval and contextual compression.

## Overview

Large Language Models generate responses using knowledge stored in their parameters. However, parametric knowledge may be outdated, incomplete, or unavailable for private domain-specific information.

Retrieval-Augmented Generation solves this problem by retrieving relevant information from an external knowledge source and providing it to the LLM as context before generating the final response.

This repository contains my practical implementations and revision notes from learning Advanced RAG using Python, LangChain, and related libraries.

## RAG Data Flow

```text
Knowledge Sources
       ↓
Document Loaders
       ↓
Document Objects
       ↓
Text Splitting
       ↓
Embeddings
       ↓
Vector Store
       ↓
Retriever
       ↓
Relevant Context
       ↓
Prompt + Context + LLM
       ↓
Generated Answer
```

## Topics Covered

### RAG Fundamentals

- What is RAG?
- Why is RAG required?
- Parametric knowledge vs external knowledge
- Limitations of LLM context windows
- Hallucination in LLM applications
- How RAG reduces hallucination
- RAG vs fine-tuning
- Components of a RAG system
- RAG data flow
- Common RAG failure points

### Document Loaders

- Why document loaders are required
- LangChain `Document` objects
- `load()` vs `lazy_load()`
- Loading structured and unstructured data
- Preserving source metadata
- Cleaning extracted content

Implemented knowledge sources:

- JSON files
- Text files
- Web pages
- Recursively crawled websites
- PDF documents

Loader and parser examples:

- `JSONLoader`
- `WebBaseLoader`
- `RecursiveUrlLoader`
- `TextLoader`
- `PyPDFLoader`
- `PDFPlumberLoader`
- `PDFMinerLoader`
- Direct `pypdf` implementation
- Direct `pdfplumber` implementation
- Direct `pdfminer.six` implementation

### Retrieval Strategies

- Similarity search
- Similarity score threshold retrieval
- Maximum Marginal Relevance — MMR
- BM25 retrieval
- Hybrid search
- Parent Document Retriever

### Contextual Compression

- `ContextualCompressionRetriever`
- `LLMChainExtractor`
- `EmbeddingsFilter`
- `DocumentCompressorPipeline`
- Embedding-based document filtering
- LLM-based relevant-content extraction

## Document Loader Flow

```text
JSON / TXT / PDF / Web
          ↓
     Document Loader
          ↓
LangChain Document Objects
          ↓
 page_content + metadata
          ↓
      Text Splitter
          ↓
         Chunks
```

Document loaders only extract and normalize source content. Chunking, embeddings, vector storage, and retrieval are separate stages.

## PDF Loader Comparison

| Library | Best suited for | Limitation |
|---|---|---|
| `pypdf` | Regular text-based PDFs and quick prototypes | Complex layouts and tables may be inaccurate |
| `pdfplumber` | Tables and detailed page inspection | Detailed processing may be slower |
| `pdfminer.six` | Lower-level text and layout analysis | Requires more custom page-level processing |

Scanned PDFs may not contain a text layer. Such documents require OCR before their contents can be used in a RAG pipeline.

## Retrieval Strategy Comparison

| Retriever | Purpose |
|---|---|
| Similarity Search | Retrieves documents semantically similar to the query |
| Threshold Retriever | Returns documents above a minimum similarity score |
| MMR | Balances relevance and diversity |
| BM25 | Performs keyword-based sparse retrieval |
| Hybrid Search | Combines semantic and keyword retrieval |
| Parent Document Retriever | Retrieves small chunks but returns their larger parent documents |

## Contextual Compression Flow

```text
User Query
    ↓
Base Retriever
    ↓
Candidate Documents
    ↓
Document Compressor
    ↓
Filtered or Extracted Context
    ↓
LLM
    ↓
Final Answer
```

Contextual compression improves the quality of retrieved context by removing irrelevant documents or extracting only relevant portions before sending them to the LLM.

## Project Structure

```text
advanced-rag-learning/
│
├── 01-rag-fundamentals/
├── 02-document-loaders/
│   ├── json-loader/
│   ├── web-loader/
│   ├── recursive-url-loader/
│   ├── text-loader/
│   └── pdf-loaders/
│
├── 03-text-splitting/
├── 04-embeddings/
├── 05-vector-stores/
├── 06-advanced-retrievers/
├── 07-contextual-compression/
├── knowledge-source/
│
├── .env.example
├── .gitignore
├── pyproject.toml
├── uv.lock
└── README.md
```

The directory structure will be expanded as new Advanced RAG concepts are implemented.

## Tech Stack

- Python
- LangChain
- LangChain Core
- LangChain Text Splitters
- OpenAI
- `pypdf`
- `pdfplumber`
- `pdfminer.six`
- Beautiful Soup
- jq
- Jupyter Notebook
- uv

## Installation

Clone the repository:

```bash
git clone git@github-jayee28:jayee28/advanced-rag-learning.git
cd advanced-rag-learning
```

Install dependencies using uv:

```bash
uv sync
```

If the project does not yet contain a `pyproject.toml`:

```bash
uv init
uv add langchain langchain-core langchain-text-splitters
uv add pypdf pdfplumber pdfminer-six
uv add beautifulsoup4 jq python-dotenv
```

Add provider-specific packages when required:

```bash
uv add langchain-openai
```

## Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
```

Load it in Python:

```python
from dotenv import load_dotenv

load_dotenv()
```

Never commit the `.env` file.

Add it to `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
.ipynb_checkpoints/
```

## Running the Project

Activate the virtual environment on Windows:

```bash
.venv\Scripts\activate
```

Run a Python file:

```bash
uv run python main.py
```

Start Jupyter:

```bash
uv run jupyter notebook
```

## Important Note About LangChain Community

Some examples in this repository may use:

```python
from langchain_community.document_loaders import ...
```

`langchain-community` is being sunset and is no longer actively maintained. These examples are retained for understanding existing LangChain APIs and course material.

Newer implementations should prefer maintained standalone integration packages or direct parser libraries when appropriate.

## Learning Goals

Through this repository, I am learning how to:

- Build complete RAG ingestion pipelines
- Load multiple knowledge-source formats
- Preserve useful document metadata
- Select an appropriate PDF parser
- Process large sources using lazy loading
- Create meaningful chunks
- Generate and store embeddings
- Compare sparse and dense retrieval
- Improve relevance using advanced retrievers
- Reduce irrelevant context using contextual compression
- Design production-oriented RAG systems

## Progress

- [x] RAG fundamentals
- [x] RAG components and data flow
- [x] Common RAG problems
- [x] JSON document loading
- [x] Web-based document loading
- [x] Recursive URL loading
- [x] Text file loading
- [x] PDF loading
- [x] Lazy loading
- [x] Similarity search
- [x] Threshold-based retrieval
- [x] MMR retrieval
- [x] BM25 retrieval
- [x] Hybrid search
- [x] Contextual compression
- [ ] Text-splitting strategies
- [ ] Parent Document Retriever
- [ ] Query transformation
- [ ] Reranking
- [ ] RAG evaluation
- [ ] Complete end-to-end Advanced RAG application

## Author

**Jayeeta Barman**

Full-Stack AI Developer focused on building applications with React, Node.js, Python, LangChain, LangGraph, LLMs, and RAG.