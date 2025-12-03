"""
Streamlit UI for Product Support Assistant
Displays RAG and MCP tool integration results
"""

import os
import sys
import asyncio
import logging
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components



from rag_assistant import get_rag_assistant

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# STREAMLIT PAGE CONFIG

st.set_page_config(
    page_title="Product Support Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS STYLING

st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 2rem;
    }
    
    /* Stashing key styling */
    .stSelectbox, .stTextInput, .stTextArea {
        margin-bottom: 1rem;
    }
    
    /* Response box styling */
    .response-box {
        background-color: #f0f2f6;
        border-left: 5px solid #1f77e5;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .tool-used {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        font-size: 0.9rem;
    }
    
    .rag-badge {
        display: inline-block;
        background-color: #007bff;
        color: white;
        padding: 0.3rem 0.6rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin-right: 0.5rem;
    }
    
    .mcp-badge {
        display: inline-block;
        background-color: #ff6b6b;
        color: white;
        padding: 0.3rem 0.6rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin-right: 0.5rem;
    }
    
    .loading-message {
        color: #666;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)


# SESSION STATE MANAGEMENT

if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_assistant" not in st.session_state:
    st.session_state.rag_assistant = None

if "tool_calls" not in st.session_state:
    st.session_state.tool_calls = []


# INITIALIZATION
@st.cache_resource
def load_rag_assistant():
    """Load RAG assistant (cached to avoid reloading)"""
    logger.info("Loading RAG Assistant...")
    try:
        assistant = get_rag_assistant()
        logger.info("RAG Assistant loaded successfully")
        return assistant
    except Exception as e:
        logger.error(f"Error loading RAG Assistant: {str(e)}")
        st.error(f"Failed to load RAG Assistant: {str(e)}")
        return None

# SIDEBAR CONFIGURATION
with st.sidebar:
    st.title(" Configuration")
    
    st.markdown("---")
    
    st.subheader("About This Assistant")
    st.info("""
     **Product Support Assistant** combines:
    
    - **RAG (Retrieval-Augmented Generation)**: Searches product FAQs
    - **MCP Tools**: Accesses external services (exchange rates, etc.)
    - **Groq LLM**: Fast inference for accurate responses
    - **Streamlit UI**: Simple, interactive interface
    """)
    
    st.markdown("---")
    
    st.subheader("System Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.rag_assistant:
            st.success("✅ RAG Assistant")
        else:
            st.error("❌ RAG Assistant")
    
    with col2:
        try:
            import httpx
            response = httpx.get("http://localhost:8001/health", timeout=2.0)
            if response.status_code == 200:
                st.success("✅ MCP Server")
            else:
                st.warning("⚠️ MCP Server")
        except:
            st.warning("⚠️ MCP Server")
    
    st.markdown("---")
    
    st.subheader("Options")
    
    clear_history = st.button("🗑️ Clear Chat History", use_container_width=True)
    if clear_history:
        st.session_state.messages = []
        st.session_state.tool_calls = []
        st.success("Chat history cleared!")
        st.rerun()
    
    show_debug = st.checkbox("🐛 Show Debug Info", value=False)
    
    st.markdown("---")
    st.caption("Made using LangChain, Groq, and Streamlit")

# MAIN PAGE

st.title("🤖 Product Support Assistant")
st.markdown("*Powered by RAG, MCP, and Groq LLM*")

st.markdown("""
Welcome to your intelligent product support assistant! This system combines:
- **Document Retrieval (RAG)**: Searches your product knowledge base
- **External Tools (MCP)**: Calls external services like currency converters
- **Large Language Model (Groq)**: Provides intelligent responses

Ask anything about our products, and I'll help!
""")

st.markdown("---")


# CHAT INTERFACE

# Load RAG assistant
if st.session_state.rag_assistant is None:
    st.session_state.rag_assistant = load_rag_assistant()

# Display chat history
st.subheader("💬 Conversation")

chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Display tool usage if applicable
            if "tools_used" in message and message["tools_used"]:
                tool_info = message["tools_used"]
                
                with st.expander("📊 Tool Usage Details"):
                    if "rag_results" in tool_info:
                        st.markdown(f"<span class='rag-badge'>RAG</span>", unsafe_allow_html=True)
                        st.write("**Retrieved FAQ Information:**")
                        st.write(tool_info["rag_results"][:300] + "...")
                    
                    if "mcp_calls" in tool_info:
                        st.markdown(f"<span class='mcp-badge'>MCP</span>", unsafe_allow_html=True)
                        st.write("**External Tool Calls:**")
                        for call in tool_info["mcp_calls"]:
                            st.write(f"- {call}")


# INPUT & QUERY PROCESSING
st.markdown("---")

# User input
user_input = st.chat_input(
    "Ask me anything about our products...",
    key="user_input"
)

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🔄 Processing your query..."):
            try:
                # Get response from RAG assistant
                if st.session_state.rag_assistant:
                    response, metadata = st.session_state.rag_assistant.process_query(user_input)
                    
                    # Display response
                    st.write(response)
                    
                    # Create tool usage metadata
                    tools_used = {
                        "rag_results": response[:500],  # Store first 500 chars of response
                        "mcp_calls": metadata.get("tools_used", []),
                        "timestamp": metadata.get("timestamp", "")
                    }
                    
                    # Add assistant message to history with metadata
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "tools_used": tools_used,
                        "metadata": metadata
                    })
                    
                    # Display metadata in expander
                    with st.expander("📊 Response Metadata"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Tools Used", len(metadata.get("tools_used", [])))
                        
                        with col2:
                            st.metric("Tokens (~)", metadata.get("token_usage", 0))
                        
                        with col3:
                            st.metric("Response Time", "~1s")
                        
                        # Show detailed metadata
                        if show_debug:
                            st.json(metadata)
                else:
                    st.error("RAG Assistant not initialized. Please refresh the page.")
                    
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
                logger.error(f"Error in chat: {str(e)}")

# EXAMPLE QUERIES

st.markdown("---")

st.subheader("📚 Example Queries")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **RAG Queries** (Knowledge Base)
    - What are the pricing plans?
    - How do I install TechFlow Pro?
    - What features are included?
    """)

with col2:
    st.markdown("""
    **MCP Tool Queries** (External Tools)
    - What is the USD to EUR exchange rate?
    - Tell me about the Japanese Yen currency
    - Convert 50 USD to INR
    """)

st.markdown("---")

st.caption("💡 Tip: Use the sidebar to check system status and clear chat history")
