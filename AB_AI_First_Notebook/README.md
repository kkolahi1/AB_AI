---
title: AB Testing RAG Agent
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
sdk_version: 3.14
app_port: 8501
pinned: false
---

# AB Testing RAG Agent

This application is a Streamlit-based frontend for an AB Testing QA system that uses a carefully designed retrieval-augmented generation (RAG) approach with a LangGraph architecture.

## 5 Minute or Less Loom Walkthrough:

https://www.loom.com/share/04d9ff3e756f4d3dbe1aa5a3cbcb39d9 

## Document Addressing the Deliverables for this Certification Challenge:

https://docs.google.com/document/d/1zti0W70-rcuc9_23vsF_AfurYzU79V9HtQP_JqeQ_eI/edit?usp=sharing 

## All Relevent code:

Can be found in the AB_Testing_RAG_Agent.ipynb notebook here:

https://github.com/kkolahi1/AIE6/blob/main/11_Certification_Challenge/AB_Testing_RAG_Agent.ipynb 


## Features

- QA system specialized in AB Testing topics
- Intelligent query routing with LangGraph
- Source citations for all answers
- Streamlit interface for easy interaction

## Setup for Development

### Prerequisites

- Python 3.9+
- OpenAI API key
- Huggingface account and token (for deployment)

### Environment Setup

1. Clone this repository
2. Create a `.env` file in the root directory with the following content:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   HF_TOKEN=your_huggingface_token_here
   ```

### Process the PDFs

Before running the app, you need to process the PDF files to create the vectorstore:

```bash
python process_data.py
```

This will:
1. Load PDFs from `notebook_version/data/`
2. Process, chunk, and embed the documents
3. Create a Qdrant vectorstore in `data/processed_data/`

### Running the App Locally

Once the data is processed, you can run the Streamlit app:

```bash
streamlit run app/app.py
```

## Deployment to Huggingface Spaces

### Prerequisites for Deployment

1. Huggingface account
2. Docker installed locally

### Steps to Deploy

1. Process the PDFs locally: `python process_data.py`
2. Build the Docker image: `docker build -t ab-testing-qa .`
3. Create a new Huggingface Space (Docker-based)
4. Add your Huggingface token and OpenAI API key as secrets in the space
5. Push the Docker image to Huggingface

### Huggingface Spaces Configuration

The application is configured to use the following secrets:
- `OPENAI_API_KEY`: Your OpenAI API key
- `HF_TOKEN`: Your Huggingface token

## System Architecture

The AB Testing QA system uses a sophisticated LangGraph architecture:

1. **Initial RAG Node**: Retrieves documents and attempts to answer the query
2. **Helpfulness Judge**: Determines if:
   - The query is related to AB Testing
   - The initial response is helpful enough
3. **Agent Node**: If needed, uses specialized tools to improve the answer:
   - Standard retrieval tool
   - Query-rephrasing retrieval tool
   - ArXiv search tool

## Data Processing

The system processes PDFs using a specific approach:
1. Merges PDF pages while maintaining page metadata
2. Uses RecursiveCharacterTextSplitter with specific parameters
3. Embeds using OpenAI's text-embedding-3-small model
4. Stores in a Qdrant vectorstore 