import os
import pickle
import streamlit as st
import json
from pathlib import Path
from typing import Annotated, List, TypedDict, Dict, Any, Literal, Optional, NotRequired
import operator
import numpy as np
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain.schema.output_parser import StrOutputParser
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from pydantic import BaseModel, Field
import asyncio
import requests
from tavily import TavilyClient, AsyncTavilyClient
from langchain_community.retrievers import ArxivRetriever
from enum import Enum
from dataclasses import dataclass, fields
from langchain_core.runnables import RunnableConfig
from langchain.chat_models import init_chat_model
from langgraph.constants import Send
from langgraph.types import interrupt, Command
from IPython.display import Markdown, display
import uuid



# Debug function to print directory information at startup
def debug_startup_info():
    """Print debug information at startup to help identify file locations"""
    print("=" * 50)
    print("DEBUG STARTUP INFO")
    print("=" * 50)
    # Print current working directory
    print(f"Current working directory: {os.getcwd()}")
    # Check for the data directory
    print("\nChecking for data directory:")
    if os.path.exists("data"):
        print("Found 'data' directory in current directory")
        print(f"Contents: {os.listdir('data')}")
        if os.path.exists("data/processed_data"):
            print(f"Contents of data/processed_data: {os.listdir('data/processed_data')}")
    # Check common paths that might exist in Hugging Face Spaces
    common_paths = [
        "/data",
        "/repository",
        "/app",
        "/app/data",
        "/repository/data",
        "/app/repository",
        "AB_AI_RAG_Agent/data"
    ]
    print("\nChecking common paths:")
    for path in common_paths:
        if os.path.exists(path):
            print(f"Found path: {path}")
            print(f"Contents: {os.listdir(path)}")
            # Check for processed_data subdirectory
            processed_path = os.path.join(path, "processed_data")
            if os.path.exists(processed_path):
                print(f"Found processed_data at: {processed_path}")
                print(f"Contents: {os.listdir(processed_path)}")
    print("=" * 50)


# Run debug info at startup
debug_startup_info()

# Enable debugging for file paths
import os
DEBUG_FILE_PATHS = True

def debug_paths():
    if DEBUG_FILE_PATHS:
        print("Current working directory:", os.getcwd())
        print("Files in /data:", os.listdir("/data") if os.path.exists("/data") else "Not found")
        print("Files in /data/processed_data:", os.listdir("/data/processed_data") if os.path.exists("/data/processed_data") else "Not found")
        for path in ["/repository", "/app", "/app/data"]:
            if os.path.exists(path):
                print(f"Files in {path}:", os.listdir(path))

# Load environment variables
load_dotenv()


# Check for required API keys
required_keys = ["COHERE_API_KEY", "ANTHROPIC_API_KEY", "TAVILY_API_KEY"]
missing_keys = [key for key in required_keys if not os.environ.get(key)]
if missing_keys:
    st.error(f"Missing required API keys: {', '.join(missing_keys)}. Please set them as environment variables.")
    st.stop()


# Custom vector store implementation
class CustomVectorStore(VectorStore):
    def __init__(self, embedded_docs, embedding_model):
        self.embedded_docs = embedded_docs
        self.embedding_model = embedding_model

    def similarity_search_with_score(self, query, k=5):
        # Get the query embedding
        query_embedding = self.embedding_model.embed_query(query)
        # Calculate similarity scores
        results = []
        for doc in self.embedded_docs:
            # Calculate cosine similarity (1 - cosine distance)
            similarity = 1 - cosine(query_embedding, doc["embedding"])
            results.append((doc, similarity))
        # Sort by similarity score (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        # Convert to Document objects and return top k
        documents_with_scores = []
        for doc, score in results[:k]:
            document = Document(
                page_content=doc["text"],
                metadata=doc["metadata"]
            )
            documents_with_scores.append((document, score))
        return documents_with_scores

    def similarity_search(self, query, k=5):
        docs_with_scores = self.similarity_search_with_score(query, k)
        return [doc for doc, _ in docs_with_scores]


    @classmethod
    def from_texts(cls, texts, embedding, metadatas=None, **kwargs):
        """Implement required abstract method from VectorStore base class."""
        # Create embeddings for the texts
        embeddings = embedding.embed_documents(texts)
        # Create embedded docs format
        embedded_docs = []
        for i, (text, embedding_vector) in enumerate(zip(texts, embeddings)):
            metadata = metadatas[i] if metadatas else {}
            embedded_docs.append({
                "text": text,
                "embedding": embedding_vector,
                "metadata": metadata
            })
        # Return an instance with the embedded docs
        return cls(embedded_docs, embedding)


def find_processed_data():
    """Find the processed_data directory path"""
    possible_paths = [
        "data/processed_data",
        "../data/processed_data",
        "/data/processed_data",
        "/app/data/processed_data",
        "./data/processed_data",
        "/repository/data/processed_data",
        "AB_AI_RAG_Agent/data/processed_data"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            required_files = ["chunks.pkl", "bm25_retriever.pkl", "embedding_info.json", "embedded_docs.pkl"]
            if all(os.path.exists(os.path.join(path, f)) for f in required_files):
                print(f"Found processed_data at: {path}")
                return path
    raise FileNotFoundError("Could not find processed_data directory with required files")



@st.cache_resource
def initialize_vectorstore():
    """Initialize the hybrid retriever system with Cohere reranking"""
    try:
        # Find processed data directory
        processed_data_path = find_processed_data()
        
        # Load documents
        with open(os.path.join(processed_data_path, "chunks.pkl"), "rb") as f:
            documents = pickle.load(f)
        
        # Load BM25 retriever
        with open(os.path.join(processed_data_path, "bm25_retriever.pkl"), "rb") as f:
            bm25_retriever = pickle.load(f)
        bm25_retriever.k = 5 
        
        # Load embedding model info
        with open(os.path.join(processed_data_path, "embedding_info.json"), "r") as f:
            embedding_info = json.load(f)
        
        # Load pre-computed embedded docs
        with open(os.path.join(processed_data_path, "embedded_docs.pkl"), "rb") as f:
            embedded_docs = pickle.load(f)
        
        # Initialize embedding model
        embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_info["model_name"]
        )
        
        # Create custom vectorstore using pre-computed embeddings
        vectorstore = CustomVectorStore(embedded_docs, embedding_model)
        qdrant_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # Create hybrid retriever
        hybrid_retriever = EnsembleRetriever(
            retrievers=[qdrant_retriever, bm25_retriever],
            weights=[0.5, 0.5],
        )
        
        # Create Cohere reranker
        cohere_rerank = CohereRerank(
            model="rerank-english-v3.0",
            top_n=5,
        )
        
        reranker = ContextualCompressionRetriever(
            base_compressor=cohere_rerank,
            base_retriever=hybrid_retriever
        )
        
        print("Successfully initialized retriever system")
        return reranker, documents
    except Exception as e:
        st.error(f"Error initializing retrievers: {str(e)}")
        st.stop()


# Define prompts

# Prompt to generate search queries to help with planning the report
report_planner_query_writer_instructions="""You are performing research for a report. 

<Report topic>
{topic}
</Report topic>

<Report organization>
{report_organization}
</Report organization>

<Task>
Your goal is to generate {number_of_queries} web search queries that will help gather information for planning the report sections. 

The queries should:

1. Be related to the Report topic
2. Help satisfy the requirements specified in the report organization

Make the queries specific enough to find high-quality, relevant sources while covering the breadth needed for the report structure.
</Task>
"""

# Prompt to generate the report plan
report_planner_instructions="""I want a plan for a report that is concise and focused.

<Report topic>
The topic of the report is:
{topic}
</Report topic>

<Report organization>
The report should follow this organization: 
{report_organization}
</Report organization>

<Context>
Here is context to use to plan the sections of the report: 
{context}
</Context>

<Task>
Generate a list of sections for the report. Your plan should be tight and focused with NO overlapping sections or unnecessary filler. 

For example, a good report structure might look like:
1/ intro
2/ overview of topic A
3/ overview of topic B
4/ comparison between A and B
5/ conclusion

Each section should have the fields:

- Name - Name for this section of the report.
- Description - Brief overview of the main topics covered in this section.
- Research - Whether to perform web research for this section of the report.
- Content - The content of the section, which you will leave blank for now.

Integration guidelines:
- Include examples and implementation details within main topic sections, not as separate sections
- Ensure each section has a distinct purpose with no content overlap
- Combine related concepts rather than separating them

Before submitting, review your structure to ensure it has no redundant sections and follows a logical flow.
</Task>

"""

# Query writer instructions
query_writer_instructions="""You are an expert technical writer crafting targeted web search queries that will gather comprehensive information for writing a technical report section.

<Report topic>
{topic}
</Report topic>

<Section topic>
{section_topic}
</Section topic>

<Task>
Your goal is to generate {number_of_queries} search queries that will help gather comprehensive information above the section topic. 

The queries should:

1. Be related to the topic 
2. Examine different aspects of the topic

Make the queries specific enough to find high-quality, relevant sources.
</Task>
"""

# Section writer instructions
section_writer_instructions = """You are an expert technical writer crafting one section of a technical report.

<Report topic>
{topic}
</Report topic>

<Section name>
{section_name}
</Section name>

<Section topic>
{section_topic}
</Section topic>

<Existing section content (if populated)>
{section_content}
</Existing section content>

<Source material>
{context}
</Source material>


<Guidelines for writing>
1. If the existing section content is not populated, write a new section from scratch.
2. If the existing section content is populated, write a new section that synthesizes the existing section content with the Source material. If there is a discrepancy between the existing section content and the Source material, use the existing section content as the primary source. The purpose of the Source material is to provide additional information and context to help fill the gaps in the existing section content.
</Guidelines for writing>

<Length and style>
- Strict 150-200 word limit
- No marketing language
- Technical focus
- Write in simple, clear language
- Start with your most important insight in **bold**
- Use short paragraphs (2-3 sentences max)
- Use ## for section title (Markdown format)
- Only use ONE structural element IF it helps clarify your point:
  * Either a focused table comparing 2-3 key items (using Markdown table syntax)
  * Or a short list (3-5 items) using proper Markdown list syntax:
    - Use `*` or `-` for unordered lists
    - Use `1.` for ordered lists
    - Ensure proper indentation and spacing
</Length and style>

<Quality checks>
- Exactly 150-200 words (excluding title and sources)
- Careful use of only ONE structural element (table or list) and only if it helps clarify your point
- One specific example / case study
- Starts with bold insight
- No preamble prior to creating the section content
- If there is a discrepancy between the existing section content and the Source material, use the existing section content as the primary source. The purpose of the Source material is to provide additional information and context to help fill the gaps in the existing section content.
</Quality checks>
"""

# Instructions for section grading
section_grader_instructions = """Review a report section relative to the specified topic:

<Report topic>
{topic}
</Report topic>

<section topic>
{section_topic}
</section topic>

<section content>
{section}
</section content>

<search type>
{current_iteration}
</search type>

<task>
Evaluate whether the section content adequately addresses the section topic.

If the section content does not adequately address the section topic, generate {number_of_follow_up_queries} follow-up search queries to gather missing information. Note that if search type is 1, your follow-up search queries will be used to search Arxiv for academic papers. If search type is 2 or more, your follow-up search queries will be used to search Tavily for general web search.
</task>

<format>
    grade: Literal["pass","fail"] = Field(
        description="Evaluation result indicating whether the response meets requirements ('pass') or needs revision ('fail')."
    )
    follow_up_queries: List[SearchQuery] = Field(
        description="List of follow-up search queries.",
    )
</format>
"""

final_section_writer_instructions="""You are an expert technical writer crafting a section that synthesizes information from the rest of the report.

<Report topic>
{topic}
</Report topic>

<Section name>
{section_name}
</Section name>

<Section topic> 
{section_topic}
</Section topic>

<Available report content>
{context}
</Available report content>

<Task>
1. Section-Specific Approach:

For Introduction:
- Use # for report title (Markdown format)
- 50-100 word limit
- Write in simple and clear language
- Focus on the core motivation for the report in 1-2 paragraphs
- Use a clear narrative arc to introduce the report
- Include NO structural elements (no lists or tables)
- No sources section needed

For Conclusion/Summary:
- Use ## for section title (Markdown format)
- 100-150 word limit
- For comparative reports:
    * Must include a focused comparison table using Markdown table syntax
    * Table should distill insights from the report
    * Keep table entries clear and concise
- For non-comparative reports: 
    * Only use ONE structural element IF it helps distill the points made in the report:
    * Either a focused table comparing items present in the report (using Markdown table syntax)
    * Or a short list using proper Markdown list syntax:
      - Use `*` or `-` for unordered lists
      - Use `1.` for ordered lists
      - Ensure proper indentation and spacing
- End with specific next steps or implications
- No sources section needed

3. Writing Approach:
- Use concrete details over general statements
- Make every word count
- Focus on your single most important point
</Task>

<Quality Checks>
- For introduction: 50-100 word limit, # for report title, no structural elements, no sources section
- For conclusion: 100-150 word limit, ## for section title, only ONE structural element at most, no sources section
- Markdown format
- Do not include word count or any preamble in your response
</Quality Checks>"""


initial_AB_topic_check_instructions="""You are checking if a given topic is related to A/B testing (even vaguely e.g. statistics, A/B testing, experimentation, etc.).

<Topic>
{topic}
</Topic>

<Task>
Check if the topic is related to A/B testing (even vaguely, e.g. statistics, A/B testing, experimentation, etc.).

If the topic is related to A/B testing (even vaguely), return 'true'.
If the topic is not related to A/B testing, return 'false'.
</Task>
"""

class Section(BaseModel):
    name: str = Field(
        description="Name for this section of the report.",
    )
    description: str = Field(
        description="Brief overview of the main topics and concepts to be covered in this section.",
    )
    research: bool = Field(
        description="Whether to perform web research for this section of the report."
    )
    content: str = Field(
        description="The content of the section."
    )   
    sources: str = Field(
        default="", 
        description="All sources used for this section"
    )

class Sections(BaseModel):
    sections: List[Section] = Field(
        description="Sections of the report.",
    )

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query for web search.")

class Queries(BaseModel):
    queries: List[SearchQuery] = Field(
        description="List of search queries.",
    )

class Feedback(BaseModel):
    grade: Literal["pass","fail"] = Field(
        description="Evaluation result indicating whether the response meets requirements ('pass') or needs revision ('fail')."
    )
    follow_up_queries: List[SearchQuery] = Field(
        description="List of follow-up search queries.",
    )

class ReportStateInput(TypedDict):
    topic: str # Report topic
    
class ReportStateOutput(TypedDict):
    final_report: str # Final report

class ReportState(TypedDict):
    topic: str # Report topic    
    sections: list[Section] # List of report sections 
    completed_sections: Annotated[list, operator.add] # Send() API key
    report_sections_from_research: str # String of any completed sections from research to write final sections
    final_report: str # Final report
    ab_testing_check: NotRequired[bool]  # Whether the topic is related to A/B testing

class SectionState(TypedDict):
    topic: str # Report topic
    section: Section # Report section  
    search_iterations: int # Number of search iterations done
    search_queries: list[SearchQuery] # List of search queries
    source_str: str # String of formatted source content from current iteration web search (for writer)
    source_str_all: str  # All accumulated sources (for user display)
    report_sections_from_research: str # String of any completed sections from research to write final sections
    completed_sections: list[Section] # Final key we duplicate in outer state for Send() API

class SectionOutputState(TypedDict):
    completed_sections: list[Section] # Final key we duplicate in outer state for Send() API


# Initialize the AB Testing report system
@st.cache_resource
def initialize_report_system(_reranker):
    """Initialize the AB Testing report system"""
    # Create a retriever reranker
    reranker = _reranker

    # Utilities and helpers

    tavily_client = TavilyClient()
    tavily_async_client = AsyncTavilyClient()

    def get_config_value(value):
        """
        Helper function to handle both string and enum cases of configuration values
        """
        return value if isinstance(value, str) else value.value

    # Helper function to get search parameters based on the search API and config
    def get_search_params(search_api: str, search_api_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Filters the search_api_config dictionary to include only parameters accepted by the specified search API.

        Args:
            search_api (str): The search API identifier (e.g., "tavily").
            search_api_config (Optional[Dict[str, Any]]): The configuration dictionary for the search API.

        Returns:
            Dict[str, Any]: A dictionary of parameters to pass to the search function.
        """
        # Define accepted parameters for each search API
        SEARCH_API_PARAMS = {
            "rag": [],  # RAG currently accepts no additional parameters
            "arxiv": ["load_max_docs", "get_full_documents", "load_all_available_meta"],
            "tavily": []  # Tavily currently accepts no additional parameters

        }

        # Get the list of accepted parameters for the given search API
        accepted_params = SEARCH_API_PARAMS.get(search_api, [])

        # If no config provided, return an empty dict
        if not search_api_config:
            return {}

        # Filter the config to only include accepted parameters
        return {k: v for k, v in search_api_config.items() if k in accepted_params}

    def get_next_search_type(search_iterations):
        if search_iterations == 0:
            return "RAG search (internal A/B testing knowledge base)"
        elif search_iterations == 1:  
            return "ArXiv web search (search academic papers on arXiv)"
        else:
            return "tavily web search (general web sources)"

    def deduplicate_and_format_sources(search_response, max_tokens_per_source, include_raw_content=True, search_iterations=None, return_has_sources=False):
        """
        Takes a list of search responses and formats them into a readable string.
        Limits the raw_content to approximately max_tokens_per_source.
 
        Args:
            search_responses: List of search response dicts, each containing:
                - query: str
                - results: List of dicts with fields:
                    - title: str
                    - url: str
                    - content: str
                    - raw_content: str|None
                    - score: float
            max_tokens_per_source: int
            include_raw_content: bool
            search_iterations: int, optional
                If 0, deduplicate by title (for RAG results) and show only title
                Otherwise, deduplicate by URL (for web/arxiv results) and show title + URL
            return_has_sources: bool, optional
                If True, returns (formatted_string, has_sources_bool)
                If False, returns just formatted_string 
            
        Returns:
            str OR tuple: 
                - If return_has_sources=False: formatted string
                - If return_has_sources=True: (formatted_string, has_sources_bool)
        """
        # Collect all results
        sources_list = []
        for response in search_response:
            sources_list.extend(response['results'])

        if not sources_list:
            empty_result = ""
            return (empty_result, False) if return_has_sources else empty_result
    
        # Deduplicate by title if search_iterations == 0 (RAG), otherwise by URL
        if search_iterations == 0:
            unique_sources = {source['title']: source for source in sources_list}
        else:
            unique_sources = {source['url']: source for source in sources_list}

        # Check if we have unique sources after deduplication
        has_unique_sources = bool(unique_sources)
    
        if not unique_sources:
            empty_result = ""
            return (empty_result, False) if return_has_sources else empty_result

        # Format output
        formatted_text = ""
        for i, source in enumerate(unique_sources.values(), 1):
            formatted_text += f"#### {source['title']}\n\n"
        
            # Only show URL if not RAG results (search_iterations != 0)
            if search_iterations != 0:
                formatted_text += f"#### URL: {source['url']}\n\n"
        
            if include_raw_content:
                # Using rough estimate of 4 characters per token
                char_limit = max_tokens_per_source * 4
                # Handle None raw_content
                raw_content = source.get('raw_content', '')
                if raw_content is None:
                    raw_content = ''
                    print(f"Warning: No raw_content found for source {source['url']}")
                if len(raw_content) > char_limit:
                    raw_content = raw_content[:char_limit] + "... [truncated]"
                    formatted_text += f"#### Full source content limited to {max_tokens_per_source} tokens \n\n"
                
        final_result = formatted_text.strip()
        return (final_result, has_unique_sources) if return_has_sources else final_result


    def format_sections(sections: list[Section]) -> str:
        """ Format a list of sections into a string """
        formatted_str = ""
        for idx, section in enumerate(sections, 1):
            formatted_str += f"""
    {'='*60} # divider line of 60 equal signs
    Section {idx}: {section.name}
    {'='*60} # divider line of 60 equal signs
    Description:
    {section.description}
    Requires Research: 
    {section.research}

    Content:
    {section.content if section.content else '[Not yet written]'}

    """
        return formatted_str

    async def tavily_search_async(search_queries):
        """
        Performs concurrent web searches using the Tavily API.

        Args:
            search_queries (List[SearchQuery]): List of search queries to process

        Returns:
                List[dict]: List of search responses from Tavily API, one per query. Each response has format:
                    {
                        'query': str, # The original search query
                        'follow_up_questions': None,      
                        'answer': None,
                        'images': list,
                        'results': [                     # List of search results
                            {
                                'title': str,            # Title of the webpage
                                'url': str,              # URL of the result
                                'content': str,          # Summary/snippet of content
                                'score': float,          # Relevance score
                                'raw_content': str|None  # Full page content if available
                            },
                            ...
                        ]
                    }
        """
        
        search_tasks = []
        for query in search_queries:
                search_tasks.append(
                    tavily_async_client.search(
                        query,
                        max_results=5,
                        include_raw_content=True,
                        topic="general"
                    )
                )

        # Execute all searches concurrently
        search_docs = await asyncio.gather(*search_tasks)

        return search_docs

    async def arxiv_search_async(search_queries, load_max_docs=5, get_full_documents=False, load_all_available_meta=True):
        """
        Performs concurrent searches on arXiv using the ArxivRetriever.

        Args:
            search_queries (List[str]): List of search queries or article IDs
            load_max_docs (int, optional): Maximum number of documents to return per query. Default is 5.
            get_full_documents (bool, optional): Whether to fetch full text of documents. Default is True.
            load_all_available_meta (bool, optional): Whether to load all available metadata. Default is True.

        Returns:
            List[dict]: List of search responses from arXiv, one per query. Each response has format:
                {
                    'query': str,                    # The original search query
                    'follow_up_questions': None,      
                    'answer': None,
                    'images': [],
                    'results': [                     # List of search results
                        {
                            'title': str,            # Title of the paper
                            'url': str,              # URL (Entry ID) of the paper
                            'content': str,          # Formatted summary with metadata
                            'score': float,          # Relevance score (approximated)
                            'raw_content': str|None  # Full paper content if available
                        },
                        ...
                    ]
                }
        """

        # Debug: Log the start of ArXiv search
        print(f"[DEBUG] Starting ArXiv search with {len(search_queries)} queries: {[str(q) for q in search_queries]}")
        
        async def process_single_query(query):
            print(f"[DEBUG] Processing ArXiv query: {query}")
            try:
                # Debug: Log retriever creation
                print(f"[DEBUG] Creating ArxivRetriever with params: load_max_docs={load_max_docs}, get_full_documents={get_full_documents}, load_all_available_meta={load_all_available_meta}")
                
                # Create retriever for each query
                retriever = ArxivRetriever(
                    load_max_docs=load_max_docs,
                    get_full_documents=get_full_documents,
                    load_all_available_meta=load_all_available_meta
                )
                
                print(f"[DEBUG] ArxivRetriever created successfully")
                
                # Run the synchronous retriever in a thread pool
                loop = asyncio.get_event_loop()
                print(f"[DEBUG] About to invoke retriever for query: {query}")
                docs = await loop.run_in_executor(None, lambda: retriever.invoke(query))
                
                print(f"[DEBUG] ArXiv query '{query}' returned {len(docs)} documents")
                
                # Debug: Log document details
                if docs:
                    print(f"[DEBUG] First document metadata keys: {list(docs[0].metadata.keys())}")
                    print(f"[DEBUG] First document has page_content: {bool(docs[0].page_content)}")
                else:
                    print(f"[DEBUG] no documents returned for query: {query}")
                
                results = []
                # Assign decreasing scores based on the order
                base_score = 1.0
                score_decrement = 1.0 / (len(docs) + 1) if docs else 0
                
                for i, doc in enumerate(docs):
                    # Normalize metadata keys to lowercase with underscores
                    normalized_metadata = {k.lower().replace(' ', '_'): v for k, v in doc.metadata.items()}

                    print(f"[DEBUG] Processing doc {i+1}: {normalized_metadata.get('title', 'No title')}")

                    # Extract metadata using consistent lowercase keys
                    url = normalized_metadata.get('entry_id', '')
                    title = normalized_metadata.get('title', '')
                    authors = normalized_metadata.get('authors', '')
                    published = normalized_metadata.get('published')
                    
                    # Handle summary with fallback to page_content
                    summary = normalized_metadata.get('summary', '')
                    if not summary and doc.page_content:
                        summary = doc.page_content.strip()
                    
                    # Build content with guaranteed fields
                    content_parts = []
                    if summary:
                        content_parts.append(f"Summary: {summary}")
                    if authors:
                        content_parts.append(f"Authors: {authors}")

                    
                    # Add publication information
                    if published:
                        published_str = published.isoformat() if hasattr(published, 'isoformat') else str(published)
                        content_parts.append(f"Published: {published_str}")
                    
                    # Add additional metadata if available
                    primary_category = normalized_metadata.get('primary_category', '')
                    if primary_category:
                        content_parts.append(f"Primary Category: {primary_category}")
                    
                    categories = normalized_metadata.get('categories', [])
                    if categories:
                        if isinstance(categories, list):
                            content_parts.append(f"Categories: {', '.join(categories)}")
                        else:
                            content_parts.append(f"Categories: {categories}")
                    
                    comment = normalized_metadata.get('comment', '')
                    if comment:
                        content_parts.append(f"Comment: {comment}")
                    
                    journal_ref = normalized_metadata.get('journal_ref', '')
                    if journal_ref:
                        content_parts.append(f"Journal Reference: {journal_ref}")
                    
                    doi = normalized_metadata.get('doi', '')
                    if doi:
                        content_parts.append(f"DOI: {doi}")
                    
                    # Get PDF link if available in the links
                    links = normalized_metadata.get('links', [])
                    if links:
                        for link in links:
                            if 'pdf' in str(link).lower():
                                content_parts.append(f"PDF: {link}")
                                break
                    
                    # Join all content parts with newlines 
                    content = "\n".join(content_parts)
                    
                    result = {
                        'title': title,
                        'url': url,
                        'content': content,
                        'score': base_score - (i * score_decrement),
                        'raw_content': doc.page_content if get_full_documents else None
                    }
                    results.append(result)
                    
                print(f"[DEBUG] Query '{query}' processed successfully, returning {len(results)} results")
                
                return {
                    'query': query,
                    'follow_up_questions': None,
                    'answer': None,
                    'images': [],
                    'results': results
                }
            except Exception as e:
                # Handle exceptions gracefully
                print(f"[DEBUG ERROR] Error processing arXiv query '{query}': {str(e)}")
                print(f"[DEBUG ERROR] Exception type: {type(e).__name__}")
                import traceback
                print(f"[DEBUG ERROR] Full traceback: {traceback.format_exc()}")
                return {
                    'query': query,
                    'follow_up_questions': None,
                    'answer': None,
                    'images': [],
                    'results': [],
                    'error': str(e)
                }
        
        # Process queries sequentially with delay to respect arXiv rate limit (1 request per 3 seconds)
        search_docs = []
        for i, query in enumerate(search_queries):
            try:
                # Add delay between requests (3 seconds per ArXiv's rate limit)
                if i > 0:  # Don't delay the first request
                    print(f"[DEBUG] Adding 4-second delay before processing query {i+1}")
                    await asyncio.sleep(4.0)
                
                result = await process_single_query(query)
                search_docs.append(result)
                print(f"[DEBUG] Completed processing query {i+1}/{len(search_queries)}")
            except Exception as e:
                # Handle exceptions gracefully
                print(f"[DEBUG ERROR] Error processing arXiv query '{query}': {str(e)}")
                search_docs.append({
                    'query': query,
                    'follow_up_questions': None,
                    'answer': None,
                    'images': [],
                    'results': [],
                    'error': str(e)
                })
                
                # Add additional delay if we hit a rate limit error
                if "429" in str(e) or "Too Many Requests" in str(e):
                    print("[DEBUG] ArXiv rate limit exceeded. Adding additional delay...")
                    await asyncio.sleep(7.0)  # Add a longer delay if we hit a rate limit
        
        print(f"[DEBUG] ArXiv search completed. Total results across all queries: {sum(len(doc.get('results', [])) for doc in search_docs)}")
        return search_docs

    async def rag_search_async(search_queries):
        """
        Performs concurrent RAG searches of our thorough A/B testing collection using the reranker.

        Args:
            search_queries (List[SearchQuery]): List of search queries to process

        Returns:
            List[dict]: List of search responses from RAG, one per query. Each response has format:
                {
                    'query': str, # The original search query
                    'follow_up_questions': None,      
                    'answer': None,
                    'images': list,
                    'results': [                     # List of search results
                        {
                            'title': str,            # Title in format "Kohavi: {title}, Section: {section}"
                            'url': str,              # None for RAG results
                            'content': str,          # None for RAG results
                            'score': float,          # None for RAG results
                            'raw_content': str|None  # Chunk's page_content
                        },
                        ...
                    ]
                }
        """
        
        async def single_rag_search(query):
            # Retrieve documents. It's a best practice to return contexts in ascending order
            docs_descending = reranker.get_relevant_documents(query)
            docs = docs_descending[::-1]
            
            # Format each document as a result
            results = []
            for doc in docs:
                source_path = doc.metadata.get("source", "")
                filename = source_path.split("/")[-1] if "/" in source_path else source_path

                # Remove .pdf extension if present
                if filename.endswith('.pdf'):
                    filename = filename[:-4]

                section = doc.metadata.get("section_title", "unknown")
                
                title = f"Kohavi: {filename}, Section: {section}"
                
                results.append({
                    'title': title,
                    'url': None,
                    'content': None,
                    'score': None,
                    'raw_content': doc.page_content
                })
            
            return {
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': results
            }
        
        # Create tasks for concurrent execution
        search_tasks = [single_rag_search(query) for query in search_queries]
        
        # Execute all searches concurrently
        search_responses = await asyncio.gather(*search_tasks)
        
        return search_responses
    

    DEFAULT_REPORT_STRUCTURE = """Use this structure to create a report on the user-provided topic:

    1. Introduction (no research needed - REQUIRED)
    - Brief overview of the topic area
    - Set research=false for this section

    2. Main Body Sections:
    - Each section should focus on a sub-topic of the user-provided topic
    - These sections require research

    3. Conclusion (no research needed - REQUIRED) 
    - Aim for 1 structural element (either a list of table) that distills the main body sections 
    - Provide a concise summary of the report
    - Set research=false for this section

    IMPORTANT: Always include at least one Introduction section and one Conclusion section with research=false."""

    # Enum classes in Python create sets of named constants with unique values
    class SearchAPI(Enum):
        TAVILY = "tavily"
        ARXIV = "arxiv"
        RAG = "rag"

    class PlannerProvider(Enum):
        ANTHROPIC = "anthropic"
        OPENAI = "openai"

    class WriterProvider(Enum):
        ANTHROPIC = "anthropic"
        OPENAI = "openai"

    # Dataclasses automatically generate boilerplate code for classes that primarily store data
    # Dataclasses automatically create __init__, __repr__, __eq__ methods
    @dataclass(kw_only=True)
    class Configuration:
        """The configurable fields for the chatbot."""
        report_structure: str = DEFAULT_REPORT_STRUCTURE # Defaults to the default report structure

        ### SET THESE NUMBERS HIGHER FOR A LARGER / MORE DETAILED REPORT - YOU MAY RUN INTO RATE LIMITING ISSUES
        number_of_queries: int = 1 # Number of search queries to generate per iteration
        max_search_depth: int = 3 # Maximum number of reflection + search iterations

        ### UNCOMMENT BELOW IF RUN INTO RATE LIMIT ISSUES
        # planner_provider: PlannerProvider = PlannerProvider.OPENAI  # Defaults to OpenAI as provider
        # planner_model: str = "o3-mini" # Defaults to o3-mini, add "-thinking" to enable thinking mode
        # writer_provider: WriterProvider = WriterProvider.OPENAI # Defaults to OpenAI as provider
        #writer_model: str = "o3-mini" # Defaults to o3-mini

        ### COMMENT BELOW IF RUN INTO RATE LIMIT ISSUES
        planner_provider: PlannerProvider = PlannerProvider.ANTHROPIC  # Defaults to Anthropic as provider
        planner_model: str = "claude-opus-4-20250514" # Defaults to claude-opus-4-20250514
        writer_provider: WriterProvider = WriterProvider.ANTHROPIC # Defaults to Anthropic as provider
        writer_model: str = "claude-sonnet-4-20250514" # Defaults to claude-sonnet-4-20250514

        
        search_api: SearchAPI = SearchAPI.TAVILY # Default to TAVILY
        search_api_config: Optional[Dict[str, Any]] = None 

        @classmethod
        def from_runnable_config(
            cls, config: Optional[RunnableConfig] = None
        ) -> "Configuration":
            """Create a Configuration instance from a RunnableConfig."""
            configurable = (
                config["configurable"] if config and "configurable" in config else {}
            )
            values: dict[str, Any] = {
                f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
                for f in fields(cls)
                if f.init
            }
            return cls(**{k: v for k, v in values.items() if v})

    # Nodes
    async def generate_report_plan(state: ReportState, config: RunnableConfig):
        """ Generate the report plan """

        # Inputs
        topic = state["topic"]

        # Get configuration
        configurable = Configuration.from_runnable_config(config)
        report_structure = configurable.report_structure
        number_of_queries = configurable.number_of_queries
        # We want to use tavily as the search API for generating the report plan
        search_api = "tavily"
        

        # Convert JSON object to string if necessary
        if isinstance(report_structure, dict):
            report_structure = str(report_structure)

        # Set writer model (model used for query writing and section writing)
        writer_provider = get_config_value(configurable.writer_provider)
        writer_model_name = get_config_value(configurable.writer_model)
        writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider) 

        # Forces the model to generate valid JSON matching the Queries schema, which 
        # makes it easier to process the results systemically
        structured_llm = writer_model.with_structured_output(Queries)

        # Format system instructions
        system_instructions_query = report_planner_query_writer_instructions.format(topic=topic, report_organization=report_structure, number_of_queries=number_of_queries)

        # Generate queries  
        results = structured_llm.invoke([SystemMessage(content=system_instructions_query),
                                        HumanMessage(content="Generate search queries that will help with planning the sections of the report.")])

        # Web search
        query_list = [query.search_query for query in results.queries]

        search_api_config = configurable.search_api_config or {}
        params_to_pass = get_search_params(search_api, search_api_config)

        # Search the web with parameters
        if search_api == "tavily":
            search_results = await tavily_search_async(query_list, **params_to_pass)
            source_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1500, include_raw_content=False)
        elif search_api == "arxiv":
            search_results = await arxiv_search_async(query_list, **params_to_pass)
            source_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1500, include_raw_content=False)
        else:
            raise ValueError(f"Unsupported search API: {search_api}")

        # Format system instructions
        system_instructions_sections = report_planner_instructions.format(topic=topic, report_organization=report_structure, context=source_str)

        # Set the planner
        planner_provider = get_config_value(configurable.planner_provider)
        planner_model = get_config_value(configurable.planner_model)

        # Report planner instructions
        planner_message = """Generate the sections of the report. Your response must include a 'sections' field containing a list of sections. 
                            Each section must have: name, description, plan, research, and content fields."""

        # Run the planner

        planner_llm = init_chat_model(
        model=planner_model,  
        model_provider=planner_provider,
        max_tokens=32_000,
        thinking={"type": "enabled", "budget_tokens": 24_000}  
        )

        # Forces the model to generate valid JSON matching the Sections schema, which 
        # makes it easier to process the results systemically
        structured_llm = planner_llm.with_structured_output(Sections)
        report_sections = structured_llm.invoke([SystemMessage(content=system_instructions_sections),
                                                    HumanMessage(content=planner_message)])

        # Get sections
        sections = report_sections.sections

        return Command(goto=[Send("build_section_with_web_research", {"topic": topic, "section": s, "search_iterations": 0}) for s in sections if s.research], update={"sections": sections})

    def generate_queries(state: SectionState, config: RunnableConfig):
        """ Generate search queries for a report section to query our A/B testing RAG collection """

        # Get state 
        topic = state["topic"]
        section = state["section"]

        # Get configuration
        configurable = Configuration.from_runnable_config(config)
        number_of_queries = configurable.number_of_queries

        # Generate queries 
        writer_provider = get_config_value(configurable.writer_provider)
        writer_model_name = get_config_value(configurable.writer_model)
        writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider) 
        structured_llm = writer_model.with_structured_output(Queries)

        # Format system instructions
        system_instructions = query_writer_instructions.format(topic=topic, 
                                                            section_topic=section.description, 
                                                            number_of_queries=number_of_queries)

        # Generate queries  
        queries = structured_llm.invoke([SystemMessage(content=system_instructions),
                                        HumanMessage(content="Generate search queries on the provided topic.")])

        return {"search_queries": queries.queries}

    async def search_rag_and_web(state: SectionState, config: RunnableConfig):
        """ Search A/B testing RAG collection and web with dual source tracking """

        # Get state 
        search_queries = state["search_queries"]
        search_iterations = state["search_iterations"]
        existing_source_str_all = state.get("source_str_all", "")  # All previous sources

        # Get configuration and choose search API based on iteration
        configurable = Configuration.from_runnable_config(config)
        
        if search_iterations == 0:
            search_api = "rag"
        elif search_iterations == 1:
            search_api = "arxiv"
        else:
            search_api = "tavily"

        # Execute search 
        query_list = [query.search_query for query in search_queries]
        search_api_config = configurable.search_api_config or {}
        params_to_pass = get_search_params(search_api, search_api_config)

        if search_api == "rag":
            search_results = await rag_search_async(query_list)
        elif search_api == "arxiv":
            search_results = await arxiv_search_async(query_list, **params_to_pass)
        elif search_api == "tavily":
            search_results = await tavily_search_async(query_list)
        else:
            raise ValueError(f"Unsupported search API: {search_api}")

        # Format current iteration sources and check if there are any
        # Use return_has_sources=True to get both the formatted string and the boolean
        current_source_str, has_sources = deduplicate_and_format_sources(
            search_results, 
            max_tokens_per_source=1500, 
            include_raw_content=True, 
            search_iterations=search_iterations,
            return_has_sources=True
        )

        # Only add iteration header and sources if there are actually sources to display
        if has_sources:
            iteration_header = f"{'='*80}\n### SEARCH ITERATION {search_iterations + 1} - {search_api.upper()} RESULTS\n{'='*80}\n\n"
            
            # Accumulate all sources for user display
            if existing_source_str_all:
                accumulated_source_str = existing_source_str_all + "\n\n" + iteration_header + current_source_str
            else:
                accumulated_source_str = iteration_header + current_source_str
        else:
            # No sources found, don't add header, keep existing sources
            accumulated_source_str = existing_source_str_all
            current_source_str = ""  # No sources for writer

        return {
            "source_str": current_source_str,  # Only current iteration for writer
            "source_str_all": accumulated_source_str,  # All sources for user display
            "search_iterations": search_iterations + 1
        }

    def write_section(state: SectionState, config: RunnableConfig) -> Command[Literal[END, "search_rag_and_web"]]:
        """ Write a section of the report """

        # Get state 
        topic = state["topic"]
        section = state["section"]
        source_str = state["source_str"]
        search_iterations = state["search_iterations"]  

        # Get configuration
        configurable = Configuration.from_runnable_config(config)

        # Get configuration
        configurable = Configuration.from_runnable_config(config)

        # Format system instructions
        system_instructions = section_writer_instructions.format(topic=topic, 
                                                                section_name=section.name, 
                                                                section_topic=section.description, 
                                                                context=source_str, 
                                                                section_content=section.content)
        
        # Generate section  
        writer_provider = get_config_value(configurable.writer_provider)
        writer_model_name = get_config_value(configurable.writer_model)
        writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider) 
        section_content = writer_model.invoke([SystemMessage(content=system_instructions),
                                            HumanMessage(content="Generate a report section based on the existing section content (if any) and the provided sources.")])
        
        # Write content to the section object  
        section.content = section_content.content

        # Grade prompt 
        section_grader_message = """Grade the report and consider follow-up questions for missing information.
                                If the grade is 'pass', return empty strings for all follow-up queries.
                                If the grade is 'fail', provide specific search queries to gather missing information."""
        
        section_grader_instructions_formatted = section_grader_instructions.format(topic=topic, 
                                                                                section_topic=section.description,
                                                                                section=section.content, 
                                                                                number_of_follow_up_queries=configurable.number_of_queries,
                                                                                current_iteration=search_iterations)
        
        # Use planner model for reflection
        planner_provider = get_config_value(configurable.planner_provider)
        planner_model = get_config_value(configurable.planner_model)

        reflection_llm = init_chat_model(
        model=planner_model,  
        model_provider=planner_provider,
        max_tokens=32_000,
        thinking={"type": "enabled", "budget_tokens": 24_000}  
        )

        reflection_model = reflection_llm.with_structured_output(Feedback)
        feedback = reflection_model.invoke([SystemMessage(content=section_grader_instructions_formatted),
                                                HumanMessage(content=section_grader_message)])
        
        # If the section is passing or max depth reached
        if feedback.grade == "pass" or state["search_iterations"] >= configurable.max_search_depth:
            # Store sources in the section object 
            section.sources = state.get("source_str_all", "") 

            return Command(
                update={
                    "completed_sections": [section]
                },
                goto=END
            )
        else:
            return Command(
                update={"search_queries": feedback.follow_up_queries, "section": section},
                goto="search_rag_and_web"
            )

    def write_final_sections(state: SectionState, config: RunnableConfig):
        """ Write final sections of the report, which do not require RAG or web search and use the completed sections as context """

        # Get configuration
        configurable = Configuration.from_runnable_config(config)

        # Get state 
        topic = state["topic"]
        section = state["section"]
        completed_report_sections = state["report_sections_from_research"]
        
        # Format system instructions
        system_instructions = final_section_writer_instructions.format(topic=topic, section_name=section.name, section_topic=section.description, context=completed_report_sections)

        # Generate section  
        writer_provider = get_config_value(configurable.writer_provider)
        writer_model_name = get_config_value(configurable.writer_model)
        writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider) 
        section_content = writer_model.invoke([SystemMessage(content=system_instructions),
                                            HumanMessage(content="Generate a report section based on the provided sources.")])
        
        # Write content to section 
        section.content = section_content.content

        # Write the updated section to completed sections
        return {"completed_sections": [section]}

    def gather_completed_sections(state: ReportState):
        """ Gather completed sections from research and format them as context for writing the final sections """    

        # Get original section order and completed sections
        original_sections = state["sections"]
        completed_sections = state["completed_sections"]
        
        # Create mapping of completed sections by name
        completed_by_name = {s.name: s for s in completed_sections}
        
        # Sort completed sections by original report order
        ordered_completed_sections = []
        for original_section in original_sections:
            if original_section.name in completed_by_name:
                ordered_completed_sections.append(completed_by_name[original_section.name])
        
        # Create sections without sources in correct order
        sections_without_sources = []
        for section in ordered_completed_sections:
            temp_section = Section(
                name=section.name,
                description=section.description,
                research=section.research,
                content=section.content,
                sources=""
            )
            sections_without_sources.append(temp_section)

        # Format in original report order
        completed_report_sections = format_sections(sections_without_sources)

        return {"report_sections_from_research": completed_report_sections}

    def initiate_final_section_writing(state: ReportState):
        """ Write any final sections using the Send API to parallelize the process """    

        # Kick off section writing in parallel via Send() API for any sections that do not require research
        return Command(goto=[Send("write_final_sections", {"topic": state["topic"], "section": s, "report_sections_from_research": state["report_sections_from_research"]}) for s in state["sections"] if not s.research ])


    def compile_final_report(state: ReportState):
        """ Compile the final report with section-grouped sources only for research sections """    

        # Get sections and sources
        sections = state["sections"]
        completed_sections = {s.name: s.content for s in state["completed_sections"]}

        # Update sections with completed content while maintaining original order
        for section in sections:
            section.content = completed_sections[section.name]

        # Compile main report
        main_report = "\n\n".join([s.content for s in sections])
        
        # Add sources section with organization by research sections only
        research_sections_with_sources = [s for s in state["completed_sections"] if s.research and s.sources]
        
        if research_sections_with_sources:
            sources_section = "\n\n## Sources Used\n\n"
            
            # Iterate through sections in original order and add sources if they exist
            for section in sections:
                if section.research:
                    # Find the completed section with sources
                    completed_section = next((s for s in state["completed_sections"] if s.name == section.name), None)
                    if completed_section and completed_section.sources:
                        sources_section += f"### Sources for Section: {section.name}\n\n"
                        sources_section += completed_section.sources + "\n\n"
            
            final_report_with_sources = main_report + sources_section
        else:
            final_report_with_sources = main_report

        return {"final_report": final_report_with_sources}

    def initial_AB_topic_check(state: ReportState, config):
        """ Checks if the topic is related to A/B testing """   

        # Get the topic
        topic = state["topic"]

        # Get configuration
        configurable = Configuration.from_runnable_config(config)

        # Format system instructions
        system_instructions = initial_AB_topic_check_instructions.format(topic=topic) 

        # initial check human message
        initial_AB_topic_check_message = """Check if the topic is related to A/B testing (even vaguely e.g. statistics, A/B testing, experimentation, etc.). If the topic is related to A/B testing (even vaguely), return 'true'. If the topic is not related to A/B testing, return 'false'. """

        # Use planner model for reflection
        planner_provider = get_config_value(configurable.planner_provider)
        planner_model = get_config_value(configurable.planner_model)

        reflection_model = init_chat_model(
        model=planner_model,  
        model_provider=planner_provider,
        max_tokens=32_000,
        thinking={"type": "enabled", "budget_tokens": 24_000}  
        )

        feedback = reflection_model.invoke([SystemMessage(content=system_instructions),
                                                HumanMessage(content=initial_AB_topic_check_message)])
        
        # Extract the response and determine if it's explicitly NOT A/B testing related
        response_content = str(feedback.content).lower().strip()
        is_explicitly_not_ab_testing = "false" in response_content
        
        # Update state with the result
        updated_state = state.copy()
        updated_state["ab_testing_check"] = not is_explicitly_not_ab_testing  # True unless explicitly false
        
        # Only if explicitly NOT A/B testing related, set the final message

        if is_explicitly_not_ab_testing:
            return {
                "ab_testing_check": False,
                "final_report": "I'm trained to only generate reports related to A/B testing. Thus, unfortunately, I can't make this report."
            }
        else:
            return {
                "ab_testing_check": True
            }

    def route_after_ab_check(state: ReportState):
        """Route to either generate_report_plan or end based on A/B testing check"""
        # Only end if we explicitly determined it's NOT A/B testing related
        if state.get("ab_testing_check", True):  # Default to True (continue) if check is missing
            return "generate_report_plan"
        else:
            return END

    section_builder = StateGraph(SectionState, output=SectionOutputState)
    section_builder.add_node("generate_queries", generate_queries)
    section_builder.add_node("search_rag_and_web", search_rag_and_web)
    section_builder.add_node("write_section", write_section)

    # Add edges
    section_builder.add_edge(START, "generate_queries")
    section_builder.add_edge("generate_queries", "search_rag_and_web")
    section_builder.add_edge("search_rag_and_web", "write_section")

    # Outer graph -- 

    # Add nodes
    builder = StateGraph(ReportState, input=ReportStateInput, output=ReportStateOutput, config_schema=Configuration)
    builder.add_node("initial_AB_topic_check", initial_AB_topic_check)
    builder.add_node("generate_report_plan", generate_report_plan)
    builder.add_node("build_section_with_web_research", section_builder.compile())
    builder.add_node("gather_completed_sections", gather_completed_sections)
    builder.add_node("write_final_sections", write_final_sections)
    builder.add_node("compile_final_report", compile_final_report)
    builder.add_node("initiate_final_section_writing", initiate_final_section_writing)


    # Add edges
    builder.add_edge(START, "initial_AB_topic_check")  # Start with AB check
    builder.add_conditional_edges("initial_AB_topic_check", route_after_ab_check, ["generate_report_plan", END])  # Conditional routing
    builder.add_edge("build_section_with_web_research", "gather_completed_sections")
    builder.add_edge("gather_completed_sections", "initiate_final_section_writing")
    builder.add_edge("write_final_sections", "compile_final_report")
    builder.add_edge("compile_final_report", END)

    return builder.compile()

def start_new_report(topic, report_placeholder):
    """Start a new report generation process"""
    with st.spinner("Generating comprehensive report...This may take about 3-7 minutes."):
        
        # Create input state
        input_state = {"topic": topic}
        
        # Run graph to completion
        try:
            config = {}

            # Use asyncio.run to handle async function
            result = asyncio.run(run_graph_to_completion(input_state, config))
            
            if result.get("ab_testing_check") == False:
                # Not AB testing related
                response = result.get("final_report", "This topic is not related to A/B testing.")
                report_placeholder.markdown(response)
                return response
            else:
                # AB testing related - show final report
                final_report = result.get("final_report", "")
                if final_report:
                    final_content = f"## 📄 Final Report\n\n{final_report}"
                    report_placeholder.markdown(final_content)
                    return final_content
                else:
                    error_msg = "No report was generated."
                    report_placeholder.error(error_msg)
                    return None
                    
        except Exception as e:
            error_msg = f"Error generating report: {str(e)}"
            report_placeholder.error(error_msg)
            return None

async def run_graph_to_completion(input_state, config):
    """Run the graph to completion"""
    result = await report_system.ainvoke(input_state, config)
    return result

# Streamlit interface
st.markdown(
    "<h1>📊 A/B<sub><span style='color:green;'>AI</span></sub></h1>",
    unsafe_allow_html=True
)
st.markdown("""
A/B<sub><span style='color:green;'>AI</span></sub> is a specialized agent that generates comprehensive reports on your provided A/B testing topics using a thorough collection of Ron Kohavi's work, including his book, papers, and LinkedIn posts. For each section of the report, if A/B<sub><span style='color:green;'>AI</span></sub> can't answer your questions using this collection, it will then search Arxiv. If that's not enough, it will finally search the web. It provides ALL sources, section by section. It has been trained to only write based on the sources it retrieves. Let's begin!
""", unsafe_allow_html=True)

# Initialize the system
try:
    # Show loading indicator
    loading_placeholder = st.empty()
    with loading_placeholder.container():
        import time
        for dots in [".", "..", "..."]:
            loading_placeholder.text(f"Loading{dots}")
            time.sleep(0.2)
    
    # Initialize components (but hide the details)
    vectorstore, chunks = initialize_vectorstore()
    report_system = initialize_report_system(vectorstore)
    
    # Clear loading indicator
    loading_placeholder.empty()
except Exception as e:
    st.error(f"Error initializing the system: {str(e)}")
    st.stop()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display chat history
for i, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    else:
        with st.chat_message("assistant"):
            st.write(message["content"])
            

# Chat input
query = st.chat_input("Please give me a topic on anything regarding A/B Testing...")

if query:
    # Display user message
    st.chat_message("user").write(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # Create assistant container immediately
    with st.chat_message("assistant"):
        report_placeholder = st.empty()
    
    # Start new report generation with placeholder
    final_content = start_new_report(query, report_placeholder)
    
    # Add to session state only after completion
    if final_content:
        st.session_state.messages.append({
            "role": "assistant", 
            "content": final_content
        })
    




