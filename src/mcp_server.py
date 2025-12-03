"""
MCP Server Implementation using FastMCP
Provides exchange rate tools for the Product Support Assistant
Runs on: http://localhost:8001
"""

import os
import asyncio
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastmcp import FastMCP, Context
import httpx
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP(
    name="Product Support MCP Server",
    stateless_http=True
)

# Mock exchange rates data
MOCK_EXCHANGE_RATES = {
    ("USD", "EUR"): 0.92,
    ("USD", "GBP"): 0.79,
    ("USD", "INR"): 83.45,
    ("USD", "JPY"): 149.80,
    ("EUR", "USD"): 1.09,
    ("EUR", "GBP"): 0.86,
    ("EUR", "INR"): 90.80,
    ("GBP", "USD"): 1.27,
    ("GBP", "EUR"): 1.16,
    ("INR", "USD"): 0.012,
}

# Mock currency information
CURRENCY_INFO = {
    "USD": {
        "name": "United States Dollar",
        "symbol": "$",
        "country": "United States",
        "decimal_places": 2
    },
    "EUR": {
        "name": "Euro",
        "symbol": "€",
        "country": "European Union",
        "decimal_places": 2
    },
    "GBP": {
        "name": "British Pound Sterling",
        "symbol": "£",
        "country": "United Kingdom",
        "decimal_places": 2
    },
    "INR": {
        "name": "Indian Rupee",
        "symbol": "₹",
        "country": "India",
        "decimal_places": 2
    },
    "JPY": {
        "name": "Japanese Yen",
        "symbol": "¥",
        "country": "Japan",
        "decimal_places": 0
    }
}

# Regular async functions (not decorated with @mcp.tool)
async def get_current_exchange_rate(
    from_currency: str = "USD",
    to_currency: str = "EUR"
) -> dict:
    """
    Fetch current exchange rate between two currencies.
    """
    try:
        rate_key = (from_currency.upper(), to_currency.upper())
        
        if rate_key in MOCK_EXCHANGE_RATES:
            rate = MOCK_EXCHANGE_RATES[rate_key]
            logger.info(f"Exchange rate {from_currency} → {to_currency}: {rate}")
            
            return {
                "from_currency": from_currency.upper(),
                "to_currency": to_currency.upper(),
                "rate": rate,
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            }
        else:
            return {
                "from_currency": from_currency.upper(),
                "to_currency": to_currency.upper(),
                "rate": None,
                "status": "error",
                "error": f"Exchange rate pair {rate_key} not available"
            }
            
    except Exception as e:
        logger.error(f"Error fetching exchange rate: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }

async def get_currency_info(currency_code: str) -> dict:
    """
    Get information about a specific currency.
    """
    try:
        code = currency_code.upper()
        if code in CURRENCY_INFO:
            logger.info(f"Retrieved info for currency: {code}")
            return {
                "status": "success",
                "currency_code": code,
                **CURRENCY_INFO[code]
            }
        else:
            return {
                "status": "error",
                "error": f"Currency code {code} not found"
            }
            
    except Exception as e:
        logger.error(f"Error getting currency info: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }

# Also register them as MCP tools if you want MCP compatibility
@mcp.tool
async def get_current_exchange_rate_mcp(
    from_currency: str = "USD",
    to_currency: str = "EUR"
) -> dict:
    """MCP wrapper for exchange rate tool"""
    return await get_current_exchange_rate(from_currency, to_currency)

@mcp.tool
async def get_currency_info_mcp(currency_code: str) -> dict:
    """MCP wrapper for currency info tool"""
    return await get_currency_info(currency_code)

# Create FastAPI app
app = FastAPI(title="Product Support MCP Server", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "Product Support MCP Server",
        "version": "1.0.0",
        "endpoints": {
            "GET /health": "Health check",
            "GET /tools": "List available tools",
            "POST /mcp/call_tool": "Call a tool"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "Product Support MCP Server",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/tools")
async def list_tools():
    """List all available tools"""
    return {
        "tools": [
            {
                "name": "get_current_exchange_rate",
                "description": "Fetch current exchange rate between two currencies",
                "parameters": {
                    "from_currency": {
                        "type": "string",
                        "description": "Source currency code (e.g., USD)",
                        "default": "USD"
                    },
                    "to_currency": {
                        "type": "string", 
                        "description": "Target currency code (e.g., EUR)",
                        "default": "EUR"
                    }
                },
                "example_call": "POST /mcp/call_tool with {'tool': 'get_current_exchange_rate', 'parameters': {'from_currency': 'USD', 'to_currency': 'EUR'}}"
            },
            {
                "name": "get_currency_info",
                "description": "Get information about a specific currency",
                "parameters": {
                    "currency_code": {
                        "type": "string",
                        "description": "Currency code (e.g., USD, EUR, INR)"
                    }
                },
                "example_call": "POST /mcp/call_tool with {'tool': 'get_currency_info', 'parameters': {'currency_code': 'EUR'}}"
            }
        ]
    }

@app.post("/mcp/call_tool")
async def call_tool(payload: dict):
    """Call a tool by name with parameters."""
    try:
        tool_name = payload.get("tool")
        params = payload.get("parameters", {})
        
        if not tool_name:
            raise HTTPException(status_code=400, detail="Tool name is required")
        
        logger.info(f"Calling tool: {tool_name} with params: {params}")
        
        if tool_name == "get_current_exchange_rate":
            from_currency = params.get("from_currency", "USD")
            to_currency = params.get("to_currency", "EUR")
            result = await get_current_exchange_rate(from_currency, to_currency)
            return result
            
        elif tool_name == "get_currency_info":
            currency_code = params.get("currency_code")
            if not currency_code:
                raise HTTPException(status_code=400, detail="currency_code parameter is required")
            result = await get_currency_info(currency_code)
            return result
            
        else:
            raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
            
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error calling tool: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/exchange-rate/{from_currency}/{to_currency}")
async def exchange_rate_endpoint(from_currency: str, to_currency: str):
    """Direct endpoint for exchange rate"""
    result = await get_current_exchange_rate(from_currency, to_currency)
    return result

@app.get("/currency-info/{currency_code}")
async def currency_info_endpoint(currency_code: str):
    """Direct endpoint for currency info"""
    result = await get_currency_info(currency_code)
    return result

# Main Entry Points
def run_stdio_server():
    """Run MCP server using STDIO transport (for direct integration)"""
    logger.info("Starting MCP Server (STDIO transport)...")
    mcp.run(transport="stdio")

def run_http_server():
    """Run MCP server using HTTP transport"""
    import uvicorn
    
    logger.info("Starting MCP Server (HTTP transport on http://localhost:8001)...")
    logger.info("Available endpoints:")
    logger.info("  • GET  /health - Health check")
    logger.info("  • GET  /tools - List available tools")
    logger.info("  • POST /mcp/call_tool - Call a tool")
    logger.info("  • GET  /exchange-rate/{from}/{to} - Direct exchange rate")
    logger.info("  • GET  /currency-info/{code} - Direct currency info")
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8001,
        log_level="info"
    )

if __name__ == "__main__":
    # Check environment variable to determine transport
    transport = os.getenv("MCP_TRANSPORT", "http").lower()
    
    if transport == "stdio":
        run_stdio_server()
    else:
        run_http_server()