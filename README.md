# Product Support Assistant with RAG & MCP Integration

A document retrieval and tool-calling assistant for product support, combining Retrieval-Augmented Generation (RAG) with external tool integration via MCP.

## Screenshots
![Screenshot](Screenshot%202025-12-03%20113837.png)
![ScreenRecording](https://drive.google.com/file/d/1isI8-JJaJhOP6olVCTyZ8yprL0bRS3i_/view?usp=sharing)


## Features

- **RAG-based document retrieval** for product FAQs and support documentation
- **External tool integration** via MCP server for currency conversion and information
- **Streamlit UI** for interactive chat interface
- **Modular architecture** for easy extension and customization

## Getting Started

### Prerequisites

- Python 3.11+
- Groq API key (for LLM inference)
- MCP server running locally

### Installation

Clone the repository
git clone https://github.com/ImaduddeenKhan/RAG-MCP-product-support-assistant.git
cd RAG-MCP-product-support-assistant

Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

Install dependencies
pip install -r requirements.txt

Copy environment file
cp .env.example .env


### Configuration

Edit `.env` file with your Groq API key:
GROQ_API_KEY=your_api_key_here
MCP_SERVER_URL=http://localhost:8001


### Running the Application

Start MCP server
python src/mcp_server.py

Start Streamlit UI
streamlit run src/app.py


## Usage

1. Open Streamlit UI at http://localhost:8501
2. Ask questions about product features, pricing, or troubleshooting
3. Request currency conversion rates
4. Get information about specific currencies

## Architecture

- **RAG Assistant**: Combines document retrieval with tool calling
- **MCP Integration**: External tool server for currency information
- **Streamlit UI**: Interactive chat interface



## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
