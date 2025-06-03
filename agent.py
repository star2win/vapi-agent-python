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
                "content": """You are a friendly Q&A assistant.
                The caller is calling from {{caller_number}}.
                At the start of every call, greet them by saying 'Thank you for calling from {{caller_number}}' before proceeding to help them.
                Your role is to answer questions helpfully and conversationally.
                Keep your responses concise but informative.
                If you don't know something, be honest about it.
                Always maintain a warm and professional tone."""
            },
            {
                "role": "assistant",
                "content": "Thank you for calling from {{caller_number}}. How may I help you?"
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

    "first_message": "Thank you for calling from {{caller_number}}. How may I help you?",
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
                        "assistant_id": None,
                        "squad_id": None,
                        "server": {
                            "url": f"{SERVER_URL}/webhooks/vapi"
                        }
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


@app.post("/webhooks/vapi")
async def vapi_webhook_handler(request: Request):
    """
    Unified webhook endpoint for all Vapi events.
    This endpoint receives all webhook events from Vapi and dispatches them
    to the appropriate handler function based on the 'type' field in the payload.
    """
    try:
        data = await request.json()
        event_type = data.get("message", {}).get("type")
        call_id = data.get("message", {}).get("call", {}).get("id") # Attempt to get call_id for logging
        
        # Extract from_number and to_number from the call object
        call_data = data.get("message", {}).get("call", {})
        from_number = call_data.get("customer", {}).get("number")
        to_number = call_data.get("phoneNumber", {}).get("number")

        logger.info(f"Received Vapi webhook event: {event_type}" + (f", Call ID: {call_id}" if call_id else ""))
        if from_number:
            logger.info(f"  From: {from_number}")
        if to_number:
            logger.info(f"  To: {to_number}")

        # Dispatch based on event type
        if event_type == "function-call":
            # Assuming you have a function to handle function-call logic
            # Reuse logic from your existing vapi_function_call function
            logger.info(f"Handling function-call event for Call ID: {call_id}")
            function_name = data.get("functionCall", {}).get("name")
            parameters = data.get("functionCall", {}).get("parameters")
            logger.info(f"Function call received - Name: {function_name}, Params: {parameters}")
            # For function calls, you need to return the result or actions for the assistant
            # For now, return a placeholder success message. Implement your actual function logic here.
            return {"result": "Function executed successfully"}

        # Add more elif blocks here for other event types you want to handle
        elif event_type == "conversation-update":
             logger.info(f"Handling conversation-update event for Call ID: {call_id}")
             # Process conversation updates
             return {"success": True, "message": "Conversation update received"}

        elif event_type == "end-of-call-report":
             logger.info(f"Handling end-of-call-report event for Call ID: {call_id}")
             # Process end of call report
             return {"success": True, "message": "End of call report received"}

        elif event_type == "hang":
             logger.info(f"Handling hang event for Call ID: {call_id}")
             # Process hang event
             return {"success": True, "message": "Hang event received"}

        elif event_type == "speech-update":
             logger.info(f"Handling speech-update event for Call ID: {call_id}")
             # Process speech update
             return {"success": True, "message": "Speech update received"}

        elif event_type == "status-update":
             logger.info(f"Handling status-update event for Call ID: {call_id}")
             # Process status update
             return {"success": True, "message": "Status update received"}

        elif event_type == "tool-calls":
             logger.info(f"Handling tool-calls event for Call ID: {call_id}")
             # Process tool calls
             return {"success": True, "message": "Tool calls received"}

        elif event_type == "transfer-destination-request":
             logger.info(f"Handling transfer-destination-request event for Call ID: {call_id}")
             # Process transfer destination request
             return {"success": True, "message": "Transfer destination request received"}

        elif event_type == "user-interrupted":
             logger.info(f"Handling user-interrupted event for Call ID: {call_id}")
             # Process user interrupted event
             return {"success": True, "message": "User interrupted event received"}

        # Add handler for assistant-request event
        elif event_type == "assistant-request":
            logger.info(f"Handling assistant-request event for Call ID: {call_id}")
            try:
                # Extract caller's phone number
                caller_number = data.get("message", {}).get("call", {}).get("customer", {}).get("number")
                logger.info(f"Caller number for assistant-request: {caller_number}")

                if not caller_number:
                    logger.error("Caller number not found in assistant-request payload.")
                    # Return an error response that Vapi will speak to the caller
                    return JSONResponse(content={"error": "Unable to get caller information."})

                # OPTION 1: Return existing assistant ID with overrides
                logger.debug(f"Returning assistantId {assistant_manager.assistant_id} with overrides for Call ID: {call_id}")
                return JSONResponse(content={
                    "assistantId": assistant_manager.assistant_id,  # Use your created assistant
                    "assistantOverrides": {
                        "variableValues": {
                            "caller_number": caller_number # Provide the caller number for substitution
                        }
                    }
                })

            except Exception as e:
                logger.error(f"Error handling assistant-request event: {str(e)}")
                # Return an error response that Vapi will speak to the caller
                return JSONResponse(content={"error": "An error occurred while setting up the assistant."})

        # If the event type is not handled, log a warning and return a success response
        logger.warning(f"Received unhandled Vapi webhook event type: {event_type}" + (f", Call ID: {call_id}" if call_id else ""))
        return {"success": True, "message": f"Event type {event_type} received but not handled"}

    except Exception as e:
        logger.error(f"Error handling Vapi webhook: {str(e)}")
        # Return an error response. Vapi might retry failed webhooks.
        return JSONResponse(status_code=500, content={"error": str(e)}) # Vapi might expect 2xx response even on handler error, verify this.


@app.post("/api/health")
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