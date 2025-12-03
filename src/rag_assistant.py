"""
LangChain RAG Assistant with MCP Tool Integration
Combines document retrieval with external tool calling via MCP
"""

import os
import json
import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import re
from dotenv import load_dotenv
import httpx


# LangChain imports for v1.1.0
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# CONFIGURATION
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found in environment variables. "
        "Please set it in .env file or environment."
    )

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8001")
FAQ_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "faqs.txt")

# DOCUMENT LOADING & VECTOR STORE INITIALIZATION
class RAGAssistant:
    """
    Main RAG Assistant class that handles document retrieval and MCP tool integration
    """
    
    def __init__(self):
        """Initialize the RAG assistant with embeddings and vector store"""
        self.embedding_model = None
        self.vector_store = None
        self.llm = None
        self.retriever = None
        self.agent_executor = None
        self.chat_history: List[BaseMessage] = []
        
        logger.info("Initializing RAG Assistant...")
        self._initialize_embeddings()
        self._load_and_index_documents()
        self._initialize_llm()
        self._setup_agent()
        logger.info("RAG Assistant initialized successfully!")
    
    def _initialize_embeddings(self):
        """Initialize HuggingFace embeddings for document encoding"""
        logger.info("Loading embeddings model...")
        try:
            # Using a lightweight model suitable for CPU inference
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            logger.info("Embeddings model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading embeddings: {str(e)}")
            raise
    
    def _load_and_index_documents(self):
        """Load FAQ documents and create FAISS vector store"""
        logger.info(f"Loading documents from {FAQ_DATA_PATH}...")
        
        try:
            # Check if file exists
            if not os.path.exists(FAQ_DATA_PATH):
                logger.warning(f"FAQ file not found at {FAQ_DATA_PATH}, using sample data")
                faq_content = self._get_sample_faqs()
            else:
                with open(FAQ_DATA_PATH, "r", encoding="utf-8") as f:
                    faq_content = f.read()
            
            # Create Document objects
            documents = [Document(page_content=faq_content, metadata={"source": "faqs"})]
            
            # Split documents into chunks for better retrieval
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            split_docs = text_splitter.split_documents(documents)
            logger.info(f"Created {len(split_docs)} document chunks")
            
            # Create FAISS vector store from documents
            self.vector_store = FAISS.from_documents(
                split_docs,
                self.embedding_model
            )
            
            # Create retriever from vector store
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}  # Retrieve top 3 most similar chunks
            )
            
            logger.info("Vector store and retriever initialized")
            
        except Exception as e:
            logger.error(f"Error loading/indexing documents: {str(e)}")
            raise
    
    def _initialize_llm(self):
        """Initialize Groq LLM for inference"""
        logger.info("Initializing Groq LLM...")
        try:
            self.llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0.7,
                max_tokens=1024,
                api_key=GROQ_API_KEY,
                timeout=30
            )
            logger.info("Groq LLM initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Groq LLM: {str(e)}")
            raise
    
    def _setup_agent(self):
        """Setup ReAct agent with RAG and MCP tools"""
        logger.info("Setting up agent with tools...")
        
        # Create tools using Tool class
        tools = [
            Tool(
                name="rag_retriever",
                func=self._rag_retriever_tool,
                description="""Search product FAQs for relevant information about TechFlow Pro Suite.
                Use for questions about: installation, pricing, features, troubleshooting, billing, etc.
                Input: A question about the product.
                Example: 'What are the pricing plans?'"""
            ),
            Tool(
                name="mcp_exchange_rate",
                func=self._mcp_exchange_rate_tool,
                description="""Get current exchange rate between two currencies.
                Input: Two 3-letter currency codes separated by a space or comma.
                Example 1: 'USD EUR' -> converts USD to EUR
                Example 2: 'USD, EUR' -> converts USD to EUR
                Example 3: 'GBP JPY' -> converts GBP to JPY"""
            ),
            Tool(
                name="mcp_currency_info",
                func=self._mcp_currency_info_tool,
                description="""Get information about a specific currency.
                Input: A 3-letter currency code.
                Example: 'EUR' -> gets information about Euro
                Example: 'JPY' -> gets information about Japanese Yen"""
            )
        ]
        
        # Create system prompt
        system_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a helpful Product Support Assistant for TechFlow Pro Suite. 
Your role is to:
1. Answer product-related questions by retrieving relevant documentation
2. Help users with pricing, features, troubleshooting, and billing inquiries
3. Provide accurate information from the knowledge base

CRITICAL INSTRUCTIONS FOR TOOL USE:
- For exchange rates: Extract ONLY 3-letter currency codes (USD, EUR, GBP, INR, JPY, etc.)
- ALWAYS pass clean 3-letter codes to tools, NOT phrases like "USD to EUR"
- For exchange rate tool: Pass "USD EUR" NOT "USD to EUR"
- For currency info: Pass "EUR" NOT "Euro currency"

Be clear and concise in your responses.
Always mention what sources you used (FAQs, MCP tools, etc.)"""
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create agent
        agent = create_tool_calling_agent(
            llm=self.llm,
            tools=tools,
            prompt=system_prompt
        )
        
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            max_iterations=5,
            early_stopping_method="force",
            handle_parsing_errors=True
        )
        
        logger.info("Agent setup complete")
    
    def _rag_retriever_tool(self, query: str) -> str:
        """
        RAG Retriever Tool: Search product FAQs for relevant information
        """
        try:
            logger.info(f"[RAG RETRIEVER] Searching for: {query}")
            
            # Retrieve relevant documents
            docs = self.retriever.invoke(query)
            
            if not docs:
                result = "No relevant FAQ information found for this query."
            else:
                # Format retrieved documents
                result = "Found relevant information from FAQs:\n\n"
                for i, doc in enumerate(docs, 1):
                    content = getattr(doc, "page_content", str(doc))
                    result += f"[Source {i}] {content[:500]}...\n\n"
            
            logger.info(f"[RAG RETRIEVER] Retrieved {len(docs)} documents")
            return result
            
        except Exception as e:
            logger.error(f"[RAG RETRIEVER] Error: {str(e)}")
            return f"Error retrieving documents: {str(e)}"
    
    def _mcp_exchange_rate_tool(self, input_str: str) -> str:
        """
        MCP Tool: Get current exchange rate between two currencies
        
        Input format: "USD EUR" or "USD, EUR" (two 3-letter currency codes)
        """
        try:
            # Clean and parse the input
            input_str = input_str.strip().upper()
            
            # Try different separators
            if "," in input_str:
                parts = [p.strip() for p in input_str.split(",")]
            elif " " in input_str:
                parts = [p.strip() for p in input_str.split()]
            else:
                parts = [input_str]
            
            # Extract first two parts as currencies
            if len(parts) >= 2:
                from_currency = parts[0][:3] if parts[0] else "USD"
                to_currency = parts[1][:3] if parts[1] else "EUR"
            elif len(parts) == 1:
                from_currency = parts[0][:3] if parts[0] else "USD"
                to_currency = "EUR"  # Default
            else:
                from_currency = "USD"
                to_currency = "EUR"
            
            logger.info(f"[MCP TOOL] Fetching exchange rate: {from_currency} → {to_currency}")
            
            # Make HTTP request to MCP server
            url = f"{MCP_SERVER_URL}/mcp/call_tool"
            
            payload = {
                "tool": "get_current_exchange_rate",
                "parameters": {
                    "from_currency": from_currency,
                    "to_currency": to_currency
                }
            }
            
            # Try calling the MCP server
            try:
                response = httpx.post(url, json=payload, timeout=5.0)
                result = response.json()
                
                # Format result
                if result.get("status") == "success":
                    output = (
                        f"Exchange Rate (from MCP Server):\n"
                        f"From: {result.get('from_currency')}\n"
                        f"To: {result.get('to_currency')}\n"
                        f"Rate: {result.get('rate')}\n"
                        f"Timestamp: {result.get('timestamp')}"
                    )
                else:
                    output = f"Error from MCP: {result.get('error', 'Unknown error')}"
                
                logger.info(f"[MCP TOOL] Success: {output[:100]}")
                return output
                
            except Exception as e:
                # If MCP server not available, return mock data with note
                logger.warning(f"[MCP TOOL] Could not reach server at {url}, using demo data")
                return (
                    f"Exchange Rate (Demo Data - MCP Server Offline):\n"
                    f"From: {from_currency}\n"
                    f"To: {to_currency}\n"
                    f"Rate: 0.92 (sample data)\n"
                    f"Note: Real MCP server is offline. In production, ensure MCP server is running."
                )
            
        except Exception as e:
            logger.error(f"[MCP TOOL] Error: {str(e)}")
            return f"Error fetching exchange rate: {str(e)}"
    
    def _mcp_currency_info_tool(self, currency_code: str) -> str:
        """
        MCP Tool: Get information about a specific currency
        """
        try:
            # Clean input - take only first 3 characters
            currency_code = currency_code.strip().upper()[:3]
            
            logger.info(f"[MCP TOOL] Fetching currency info: {currency_code}")
            
            url = f"{MCP_SERVER_URL}/mcp/call_tool"
            
            payload = {
                "tool": "get_currency_info",
                "parameters": {
                    "currency_code": currency_code
                }
            }
            
            try:
                response = httpx.post(url, json=payload, timeout=5.0)
                result = response.json()
                
                if result.get("status") == "success":
                    output = (
                        f"Currency Information (from MCP Server):\n"
                        f"Code: {result.get('currency_code')}\n"
                        f"Name: {result.get('name')}\n"
                        f"Symbol: {result.get('symbol')}\n"
                        f"Country: {result.get('country')}\n"
                        f"Decimal Places: {result.get('decimal_places')}"
                    )
                else:
                    output = f"Error: {result.get('error', 'Unknown error')}"
                
                logger.info(f"[MCP TOOL] Success: {output[:100]}")
                return output
                
            except Exception as e:
                logger.warning(f"[MCP TOOL] Could not reach server, returning fallback info")
                return f"Currency information for {currency_code} could not be retrieved. MCP server may be offline."
            
        except Exception as e:
            logger.error(f"[MCP TOOL] Error: {str(e)}")
            return f"Error fetching currency info: {str(e)}"
    
    def process_query(self, user_query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Process user query and return answer with metadata
        """
        try:
            logger.info(f"Processing query: {user_query}")
            
            # Add user message to history
            self.chat_history.append(HumanMessage(content=user_query))
            
            # Invoke agent
            response = self.agent_executor.invoke({
                "input": user_query,
                "chat_history": self.chat_history,
                "agent_scratchpad": []
            })
            
            answer = response.get("output", "No answer generated")
            
            # Add assistant response to history
            self.chat_history.append(AIMessage(content=answer))
            
            # Prepare metadata
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "query": user_query,
                "tools_used": self._extract_tools_used(response),
                "token_usage": self._estimate_tokens(user_query + answer)
            }
            
            logger.info(f"Query processed successfully")
            return answer, metadata
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"Error: {str(e)}", {"error": str(e)}
    
    def _extract_tools_used(self, response: Dict[str, Any]) -> List[str]:
        """Extract which tools were called in the agent response"""
        tools_used = []
        
        # Check if response contains tool information
        if "intermediate_steps" in response:
            for step in response["intermediate_steps"]:
                if hasattr(step[0], "tool"):
                    tools_used.append(step[0].tool)
        
        return tools_used
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough estimate of tokens"""
        return len(text) // 4
    
    def _get_sample_faqs(self) -> str:
        """Return sample FAQs if file not found"""
        return """PRODUCT SUPPORT FAQs - TechFlow Pro Suite v2.5

1. INSTALLATION & SETUP
Q: How do I install TechFlow Pro Suite?
A: Download from our website, extract the ZIP file, and run install.exe.

2. PRICING & PLANS
Q: What pricing plans are available?
A: We offer three plans: Starter ($9.99/month), Professional ($29.99/month), and Enterprise (custom pricing).

3. FEATURES & FUNCTIONALITY
Q: What are the key features of TechFlow Pro?
A: Real-time synchronization, multi-device support, API integration, advanced reporting dashboard, 24/7 support.

4. TROUBLESHOOTING
Q: Why is my application running slowly?
A: Try updating to the latest version, clearing cache, and restarting the application.

5. BILLING & SUBSCRIPTIONS
Q: How can I cancel my subscription?
A: You can cancel anytime from your account settings under Subscriptions.

6. SECURITY & COMPLIANCE
Q: How is my data protected?
A: We use 256-bit AES encryption and comply with GDPR, CCPA, and SOC 2 Type II standards."""

# GLOBAL INSTANCE
_rag_assistant_instance = None

def get_rag_assistant() -> RAGAssistant:
    """Get or create the global RAG assistant instance"""
    global _rag_assistant_instance
    
    if _rag_assistant_instance is None:
        _rag_assistant_instance = RAGAssistant()
    
    return _rag_assistant_instance

if __name__ == "__main__":
    # Test the RAG assistant
    assistant = get_rag_assistant()
    
    # Test queries
    test_queries = [
        "What are the pricing plans for TechFlow Pro?",
        "Convert 100 USD to EUR",
        "Tell me about Euro currency"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        answer, metadata = assistant.process_query(query)
        print(f"\nAnswer:\n{answer}")
        print(f"\nMetadata: {metadata}")