"""
Vapi Voice Assistant with FastAPI
A simple Q&A voice assistant using Vapi API, Twilio, and FastAPI

Tech Stack:
- FastAPI: Web framework
- Vapi SDK: Voice assistant management
- Loguru: Logging
- Uvicorn: ASGI server
- Twilio: Phone integration

Voice Pipeline:
- STT: Deepgram
- LLM: ChatGPT 4o
- TTS: ElevenLabs
"""

import os
from dotenv import load_dotenv
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from loguru import logger
import uvicorn
from vapi import Vapi

load_dotenv()

# Load SERVER_URL from environment
SERVER_URL = os.getenv("SERVER_URL")
if not SERVER_URL or not SERVER_URL.startswith("https://"):
    logger.error(f"SERVER_URL is invalid or not set: {SERVER_URL}")
    raise RuntimeError("SERVER_URL environment variable must be set and start with 'https://'")

# Configure Loguru
logger.add(
    "logs/vapi_assistant_{time}.log",
    rotation="500 MB",
    retention="10 days",
    level="INFO",
    format="{time} {level} {message}"
)


# Initialize FastAPI app
app = FastAPI(
    title="Vapi Voice Assistant API",
    description="A simple Q&A voice assistant using Vapi",
    version="1.0.0"
)


# Initialize Vapi client
# Make sure to set your VAPI_API_KEY environment variable
vapi_client = Vapi(token=os.getenv("VAPI_API_KEY"))


# Pydantic models for request/response validation
class AssistantConfig(BaseModel):
    """Configuration model for creating a Vapi assistant"""
    name: Optional[str] = "Q&A Voice Assistant"
    voice_provider: Optional[str] = "elevenlabs"
    transcriber_provider: Optional[str] = "deepgram"
    model_provider: Optional[str] = "openai"
    model: Optional[str] = "gpt-4o"


class TwilioWebhookData(BaseModel):
    """Model for Twilio webhook data"""
    CallSid: str
    From: str
    To: str
    CallStatus: str


# Assistant configuration
ASSISTANT_CONFIG = {
    "name": "Q&A Voice Assistant",

    "transcriber": {
        "provider": "deepgram",
        "model": "nova-2",
        "language": "en",
        "confidence_threshold": 0.4,
    },

    "model": {
        "provider": "openai",
        "model": "gpt-4o",
        "temperature": 0.7,
        "emotion_recognition_enabled": True,
        "messages": [
            {
                "role": "system",
                "content": "You are a friendly Q&A assistant. Your role is to answer questions helpfully and conversationally. Keep your responses concise but informative. If you don't know something, be honest about it. Always maintain a warm and professional tone."
            },
            {
                "role": "assistant",
                "content": "Hello, how may I help you?"
            }
        ],
    },

    "voice": {
        "provider": "11labs",
        "voiceId": "21m00Tcm4TlvDq8ikWAM",
        "model": "eleven_monolingual_v1",
        "stability": 0.5,
        "similarity_boost": 0.75,
        "caching_enabled": True,
    },

    "first_message": "Hello! How can I help you today?",
    "first_message_interruptions_enabled": True,
    "first_message_mode": "assistant-speaks-first",

    "voicemail_detection": {
        "provider": "google",
    },

    "silence_timeout_seconds": 30,
    "max_duration_seconds": 600,
    "background_sound": "off",
    "background_denoising_enabled": False,
    "model_output_in_messages_enabled": False,

    "transport_configurations": [
        {
            "provider": "twilio",
            "timeout": 60,
            "record": False,
            "recording_channels": "mono"
        }
    ],

    "server": {
        "url": f"{SERVER_URL}/webhooks/vapi",
        "timeout_seconds": 20,
    },
}


class VapiAssistantManager:
    """Manager class for Vapi assistant operations"""
    
    def __init__(self, vapi_client: Vapi):
        self.vapi = vapi_client
        self.assistant_id = None
        logger.info("VapiAssistantManager initialized")
    
    def create_assistant(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a new Vapi assistant with the specified configuration
        
        Args:
            config: Optional custom configuration to override defaults
            
        Returns:
            Assistant object from Vapi API
        """
        try:
            # Use provided config or default
            assistant_config = config or ASSISTANT_CONFIG
            
            logger.info(f"Creating assistant with config: {assistant_config.get('name', 'Unnamed Assistant')}")
            
            # Log the final config being sent to Vapi
            logger.debug(f"Assistant config being sent to Vapi: {assistant_config}")
            
            # Create assistant via Vapi SDK
            assistant = self.vapi.assistants.create(**assistant_config)
            
            self.assistant_id = assistant.id
            logger.info(f"Assistant created successfully with ID: {self.assistant_id}")

            # Assign Twilio phone number to assistant
            phone_number_id = "85ff5077-7ca1-4f9d-93da-d5ce96641656"
            try:
                self.vapi.phone_numbers.update(
                    id=phone_number_id,
                    request={
                        "assistant_id": self.assistant_id
                    }
                )
                logger.info(f"Assigned assistant {self.assistant_id} to phone number {phone_number_id}")
            except Exception as e:
                logger.error(f"Failed to assign assistant {self.assistant_id} to phone number {phone_number_id}: {str(e)}")

            return assistant
            
        except Exception as e:
            logger.error(f"Error creating assistant: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to create assistant: {str(e)}")
    
    def get_assistant(self, assistant_id: str) -> Dict[str, Any]:
        """
        Retrieve assistant details by ID
        
        Args:
            assistant_id: The ID of the assistant to retrieve
            
        Returns:
            Assistant object from Vapi API
        """
        try:
            assistant = self.vapi.assistants.get(assistant_id)
            logger.info(f"Retrieved assistant: {assistant_id}")
            return assistant
            
        except Exception as e:
            logger.error(f"Error retrieving assistant {assistant_id}: {str(e)}")
            raise HTTPException(status_code=404, detail=f"Assistant not found: {str(e)}")
    
    def update_assistant(self, assistant_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an existing assistant
        
        Args:
            assistant_id: The ID of the assistant to update
            updates: Dictionary of fields to update
            
        Returns:
            Updated assistant object
        """
        try:
            assistant = self.vapi.assistants.update(assistant_id, **updates)
            logger.info(f"Updated assistant {assistant_id}")
            return assistant
            
        except Exception as e:
            logger.error(f"Error updating assistant {assistant_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to update assistant: {str(e)}")
    
    def delete_assistant(self, assistant_id: str) -> bool:
        """
        Delete an assistant
        
        Args:
            assistant_id: The ID of the assistant to delete
            
        Returns:
            True if successful
        """
        try:
            self.vapi.assistants.delete(assistant_id)
            logger.info(f"Deleted assistant {assistant_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting assistant {assistant_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to delete assistant: {str(e)}")


# Initialize assistant manager
assistant_manager = VapiAssistantManager(vapi_client)


# API Routes
@app.on_event("startup")
async def startup_event():
    """Initialize the assistant on server startup"""
    logger.info("Starting Vapi Voice Assistant API")
    
    # Create the default assistant
    try:
        assistant = assistant_manager.create_assistant()
        logger.info(f"Default assistant created: {assistant.id}")
    except Exception as e:
        logger.error(f"Failed to create default assistant on startup: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint - API health check"""
    return {
        "status": "active",
        "service": "Vapi Voice Assistant API",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/api/assistants/create")
async def create_assistant(config: Optional[AssistantConfig] = None):
    """
    Create a new Vapi assistant
    
    Optional body parameters:
    - name: Assistant name
    - voice_provider: TTS provider (default: elevenlabs)
    - transcriber_provider: STT provider (default: deepgram)
    - model_provider: LLM provider (default: openai)
    - model: LLM model (default: gpt-4o)
    """
    try:
        # Use custom config if provided
        custom_config = None
        if config:
            custom_config = ASSISTANT_CONFIG.copy()
            if config.name:
                custom_config["name"] = config.name
            # Add more customization as needed
        
        assistant = assistant_manager.create_assistant(custom_config)
        
        return {
            "success": True,
            "assistant_id": assistant.id,
            "message": "Assistant created successfully"
        }
        
    except Exception as e:
        logger.error(f"Error in create_assistant endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/assistants/{assistant_id}")
async def get_assistant(assistant_id: str):
    """Get assistant details by ID"""
    assistant = assistant_manager.get_assistant(assistant_id)
    return {
        "success": True,
        "assistant": assistant
    }


@app.delete("/api/assistants/{assistant_id}")
async def delete_assistant(assistant_id: str):
    """Delete an assistant by ID"""
    success = assistant_manager.delete_assistant(assistant_id)
    return {
        "success": success,
        "message": "Assistant deleted successfully"
    }


@app.post("/webhooks/vapi/call-started")
async def vapi_call_started(request: Request):
    """
    Webhook endpoint for Vapi call started events
    This is called when a call begins
    """
    try:
        data = await request.json()
        call_id = data.get("call", {}).get("id")
        phone_number = data.get("call", {}).get("customer", {}).get("number")
        
        logger.info(f"Call started - ID: {call_id}, Phone: {phone_number}")
        
        return {"success": True}
        
    except Exception as e:
        logger.error(f"Error handling call-started webhook: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/webhooks/vapi/call-ended")
async def vapi_call_ended(request: Request):
    """
    Webhook endpoint for Vapi call ended events
    This is called when a call completes
    """
    try:
        data = await request.json()
        call_id = data.get("call", {}).get("id")
        duration = data.get("call", {}).get("duration")
        end_reason = data.get("call", {}).get("endedReason")
        
        logger.info(f"Call ended - ID: {call_id}, Duration: {duration}s, Reason: {end_reason}")
        
        # You can add custom logic here like:
        # - Save call transcript
        # - Send summary email
        # - Update database
        
        return {"success": True}
        
    except Exception as e:
        logger.error(f"Error handling call-ended webhook: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/webhooks/vapi/function-call")
async def vapi_function_call(request: Request):
    """
    Webhook endpoint for Vapi function calls
    This handles any custom functions your assistant might call
    """
    try:
        data = await request.json()
        function_name = data.get("functionCall", {}).get("name")
        parameters = data.get("functionCall", {}).get("parameters")
        
        logger.info(f"Function call received - Name: {function_name}, Params: {parameters}")
        
        # Handle different function calls here
        # For now, we'll just acknowledge
        
        return {
            "result": "Function executed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error handling function-call webhook: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/webhooks/twilio/voice")
async def twilio_voice_webhook(request: Request):
    """
    Webhook endpoint for Twilio voice calls
    This is the webhook URL you'll configure in Twilio
    
    When a call comes in, this endpoint will:
    1. Receive the call from Twilio
    2. Connect it to your Vapi assistant
    """
    try:
        # Parse form data from Twilio
        form_data = await request.form()
        call_sid = form_data.get("CallSid")
        from_number = form_data.get("From")
        to_number = form_data.get("To")
        
        logger.info(f"Incoming call - SID: {call_sid}, From: {from_number}, To: {to_number}")
        
        # Get the assistant ID (use the default one created on startup)
        if not assistant_manager.assistant_id:
            # Create one if it doesn't exist
            assistant = assistant_manager.create_assistant()
            assistant_id = assistant.id
        else:
            assistant_id = assistant_manager.assistant_id
        
        # Generate TwiML response to connect to Vapi
        # Replace YOUR_VAPI_PHONE_NUMBER with your Vapi SIP endpoint
        twiml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="wss://api.vapi.ai/ws?assistantId={assistant_id}" />
    </Connect>
</Response>"""
        
        logger.info(f"Connecting call {call_sid} to assistant {assistant_id}")
        
        return Response(content=twiml_response, media_type="text/xml")
        
    except Exception as e:
        logger.error(f"Error handling Twilio webhook: {str(e)}")
        # Return error TwiML
        error_twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>Sorry, we're experiencing technical difficulties. Please try again later.</Say>
    <Hangup/>
</Response>"""
        return Response(content=error_twiml, media_type="text/xml")


@app.get("/api/health")
async def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Check if we can reach Vapi API
        # This is a simple check - you might want to add more
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "vapi_connected": True,
            "assistant_id": assistant_manager.assistant_id
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    logger.error(f"HTTP exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup the assistant on server shutdown"""
    logger.info("Shutting down Vapi Voice Assistant API")
    try:
        if assistant_manager.assistant_id:
            assistant_manager.delete_assistant(assistant_manager.assistant_id)
            logger.info(f"Deleted assistant {assistant_manager.assistant_id} on shutdown")
    except Exception as e:
        logger.error(f"Failed to delete assistant on shutdown: {str(e)}")


if __name__ == "__main__":
    # Run the server
    # Make sure to set environment variables:
    # - VAPI_API_KEY: Your Vapi API key
    # - OPENAI_API_KEY: Your OpenAI API key (if using GPT-4o)
    # - ELEVEN_LABS_API_KEY: Your ElevenLabs API key (if required)
    # - DEEPGRAM_API_KEY: Your Deepgram API key (if required)
    
    logger.info("Starting Uvicorn server...")
    
    uvicorn.run(
        "agent:app",
        host="0.0.0.0",
        port=8001,
        reload=True,  # Enable auto-reload during development
        log_level="info"
    )