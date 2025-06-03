import os
import asyncio
import sys
from typing import Optional
from dotenv import load_dotenv
from vapi import AsyncVapi
from vapi.core.api_error import ApiError
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from loguru import logger
from pyngrok import ngrok

# Load environment variables
load_dotenv()

# Configure Loguru
logger.remove()  # Remove default handler
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/agent.log",
    rotation="500 MB",
    retention="10 days",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG"
)

# Initialize FastAPI app
app = FastAPI(title="Vapi Voice Agent")

# Initialize async VAPI client with proper configuration
vapi = AsyncVapi(
    token=os.getenv("VAPI_API_KEY"),
    timeout=60.0,  # 60 second timeout
)

class WebhookEvent(BaseModel):
    """Model for incoming webhook events"""
    type: str
    data: dict

class CallStartedData(BaseModel):
    """Model for call.started event data"""
    call_id: str
    assistant_id: Optional[str] = None

async def process_call_logs(call_id: str):
    """Process call logs asynchronously"""
    try:
        logs = vapi.logs.get(call_id=call_id)
        async for log in logs:
            logger.debug(f"Call log for {call_id}: {log}")
    except Exception as e:
        logger.error(f"Error processing logs for call {call_id}: {str(e)}")

@app.post("/webhook")
async def handle_webhook(event: WebhookEvent):
    """Handle incoming webhook events from VAPI"""
    try:
        if event.type == 'call.started':
            # Validate call data
            call_data = CallStartedData.model_validate(event.data)
            call_id = call_data.call_id
            assistant_id = call_data.assistant_id or os.getenv("VAPI_ASSISTANT_ID")
            
            if not assistant_id:
                logger.error("No assistant ID provided in webhook or environment")
                raise HTTPException(
                    status_code=400,
                    detail="No assistant ID configured"
                )
            
            logger.info(f"Received incoming call with ID: {call_id} for assistant: {assistant_id}")
            
            try:
                # Get assistant details using the correct SDK method
                assistant = await vapi.assistants.get(id=assistant_id)
                logger.info(f"Using assistant: {assistant.name}")
                
                # Start the conversation with the assistant
                await vapi.calls.start_conversation(
                    call_id=call_id,
                    assistant_id=assistant_id,
                    request_options={
                        "max_retries": 2,
                        "timeout_in_seconds": 30
                    }
                )
                logger.info(f"Started conversation for call {call_id}")
                
                # Process logs asynchronously without blocking the response
                asyncio.create_task(process_call_logs(call_id))
                
            except ApiError as e:
                logger.error(f"VAPI API error for call {call_id}: {e.status_code} - {e.body}")
                raise HTTPException(
                    status_code=500,
                    detail={
                        "message": f"VAPI API error: {e.status_code}",
                        "details": e.body
                    }
                )
        
        return {"status": "success"}
    
    except Exception as e:
        logger.exception(f"Error processing webhook: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    logger.debug("Health check requested")
    return {"status": "healthy"}

async def main():
    """Main function to run the FastAPI server"""
    import uvicorn
    
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # Start ngrok tunnel
    port = int(os.getenv('PORT', 3500))
    ngrok_tunnel = ngrok.connect(port)
    public_url = ngrok_tunnel.public_url
    logger.info(f"Ngrok tunnel established at: {public_url}")
    logger.info(f"Webhook URL for Vapi dashboard: {public_url}/webhook")
    
    # Verify assistant ID is configured
    assistant_id = os.getenv("VAPI_ASSISTANT_ID")
    if not assistant_id:
        logger.warning("VAPI_ASSISTANT_ID not set in environment variables")
    else:
        try:
            # Get assistant details using the correct SDK method
            assistant = await vapi.assistants.get(id=assistant_id)
            logger.info(f"Connected to assistant: {assistant.name}")
        except Exception as e:
            logger.error(f"Error connecting to assistant: {str(e)}")
    
    logger.info("Starting Vapi Voice Agent server")
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())
