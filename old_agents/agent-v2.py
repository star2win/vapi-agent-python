import os
import asyncio
import sys
import json
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from vapi import AsyncVapi
from vapi.core.api_error import ApiError
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from loguru import logger

# Load environment variables
load_dotenv()

# Configure Loguru
logger.remove()  # Remove default handler

# Terminal logger - only non-webhook messages
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    filter=lambda record: "webhook" not in record["message"].lower()
)

# Main log file - all logs
logger.add(
    "logs/agent.log",
    rotation="500 MB",
    retention="10 days",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",
    enqueue=True  # Enable async logging
)

# Webhook-specific log file
logger.add(
    "logs/webhooks.log",
    rotation="500 MB",
    retention="10 days",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    level="INFO",
    filter=lambda record: "webhook" in record["message"].lower(),
    enqueue=True  # Enable async logging
)

# Initialize FastAPI app
app = FastAPI(title="Vapi Voice Agent")

# Initialize async VAPI client with proper configuration
vapi = AsyncVapi(
    token=os.getenv("VAPI_API_KEY"),
    timeout=60.0,  # 60 second timeout
)

class Message(BaseModel):
    """Model for message content"""
    role: str
    message: Optional[str] = None
    content: Optional[str] = None
    time: Optional[float] = None
    secondsFromStart: Optional[float] = None
    endTime: Optional[float] = None
    duration: Optional[float] = None
    source: Optional[str] = None

class Artifact(BaseModel):
    """Model for message artifacts"""
    messages: List[Message]
    messagesOpenAIFormatted: List[Message]

class Monitor(BaseModel):
    """Model for call monitoring URLs"""
    listenUrl: str
    controlUrl: str

class Transport(BaseModel):
    """Model for call transport details"""
    provider: str
    callUrl: Optional[str] = None
    assistantVideoEnabled: Optional[bool] = None
    callSid: Optional[str] = None
    accountSid: Optional[str] = None
    callToken: Optional[str] = None

class Customer(BaseModel):
    """Model for customer information"""
    number: Optional[str] = None

class Voice(BaseModel):
    """Model for voice configuration"""
    model: str
    voiceId: str
    provider: str
    stability: float
    similarityBoost: float

class Model(BaseModel):
    """Model for AI model configuration"""
    model: str
    messages: List[Message]
    provider: str

class Transcriber(BaseModel):
    """Model for transcription configuration"""
    model: str
    language: str
    provider: str

class Server(BaseModel):
    """Model for server configuration"""
    url: str
    timeoutSeconds: int

class Assistant(BaseModel):
    """Model for assistant configuration"""
    name: str
    transcriber: Transcriber
    model: Model
    voice: Voice
    firstMessage: str
    voicemailMessage: str
    endCallMessage: str
    server: Server
    clientMessages: Optional[List[str]] = None
    id: Optional[str] = None
    orgId: Optional[str] = None
    createdAt: Optional[str] = None
    updatedAt: Optional[str] = None

class PhoneNumber(BaseModel):
    """Model for phone number details"""
    id: Optional[str] = None
    orgId: Optional[str] = None
    assistantId: Optional[str] = None
    number: Optional[str] = None
    createdAt: Optional[str] = None
    updatedAt: Optional[str] = None
    twilioAccountSid: Optional[str] = None
    name: Optional[str] = None
    provider: Optional[str] = None
    status: Optional[str] = None
    smsEnabled: Optional[bool] = None

class Call(BaseModel):
    """Model for call details"""
    id: str
    orgId: str
    createdAt: str
    updatedAt: str
    type: str
    monitor: Monitor
    transport: Transport
    webCallUrl: Optional[str] = None
    status: str
    assistant: Assistant
    assistantOverrides: Optional[dict] = None
    phoneCallProvider: Optional[str] = None
    phoneCallProviderId: Optional[str] = None
    phoneCallTransport: Optional[str] = None
    assistantId: Optional[str] = None
    phoneNumberId: Optional[str] = None
    customer: Optional[Customer] = None

class WebhookMessage(BaseModel):
    """Model for webhook message"""
    timestamp: int
    type: str
    status: Optional[str] = None
    role: Optional[str] = None
    turn: Optional[int] = None
    conversation: Optional[List[Message]] = None
    messages: Optional[List[Message]] = None
    messagesOpenAIFormatted: Optional[List[Message]] = None
    artifact: Optional[Artifact] = None
    call: Call
    phoneNumber: Optional[PhoneNumber] = None
    customer: Optional[Customer] = None
    assistant: Assistant

class WebhookEvent(BaseModel):
    """Model for incoming webhook events"""
    message: WebhookMessage

async def log_request(request: Request):
    """Log incoming request details"""
    try:
        body = await request.body()
        headers = dict(request.headers)
        query_params = dict(request.query_params)
        
        # Log request details
        logger.debug("=== Incoming Request ===")
        logger.debug(f"Method: {request.method}")
        logger.debug(f"URL: {request.url}")
        logger.debug(f"Headers: {json.dumps(headers, indent=2)}")
        logger.debug(f"Query Params: {json.dumps(query_params, indent=2)}")
        
        # Try to parse and log body if it's JSON
        try:
            body_json = json.loads(body)
            logger.debug(f"Body: {json.dumps(body_json, indent=2)}")
        except json.JSONDecodeError:
            logger.debug(f"Body (raw): {body}")
        
        logger.debug("=====================")
    except Exception as e:
        logger.error(f"Error logging request: {str(e)}")

async def process_call_logs(call_id: str):
    """Process call logs asynchronously"""
    try:
        # Get logs and await the response
        logs = await vapi.logs.get(call_id=call_id)
        async for log in logs:
            logger.debug(f"Call log for {call_id}: {log}")
    except Exception as e:
        logger.error(f"Error processing logs for call {call_id}: {str(e)}")

@app.post("/webhook")
async def handle_webhook(request: Request):
    """Handle incoming webhook events from VAPI"""
    # Log the incoming request
    await log_request(request)
    
    try:
        # Parse the request body
        body = await request.json()
        logger.info(f"Received webhook event: {json.dumps(body, indent=2)}")
        
        # Validate the event
        event = WebhookEvent.model_validate(body)
        message = event.message
        
        # Handle different event types
        if message.type == "speech-update":
            logger.info(f"Speech update for call {message.call.id}: {message.status}")
            if message.status == "started":
                logger.info(f"Assistant started speaking in call {message.call.id}")
            elif message.status == "stopped":
                logger.info(f"Assistant finished speaking in call {message.call.id}")
                
        elif message.type == "status-update":
            logger.info(f"Status update for call {message.call.id}: {message.status}")
            if message.status == "in-progress":
                logger.info(f"Call {message.call.id} is in progress")
            elif message.status == "completed":
                logger.info(f"Call {message.call.id} has completed")
            elif message.status == "failed":
                logger.error(f"Call {message.call.id} has failed")
                
        # Process logs asynchronously without blocking the response
        asyncio.create_task(process_call_logs(message.call.id))
        
        # Log success to terminal without webhook details
        logger.info("Webhook processed successfully")
        
    except Exception as e:
        logger.exception(f"Error processing webhook: {str(e)}")
    
    # Always return 200 OK response
    return JSONResponse(
        status_code=200,
        content={"status": "success", "message": "Webhook received"}
    )

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
        port=int(os.getenv('PORT', 3500)),
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())
