from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_cpp import Llama
import time
import os
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Airavata GGUF Chat API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
llm_model = None

class ChatRequest(BaseModel):
    message: str
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

class ChatResponse(BaseModel):
    response: str
    latency_ms: float
    tokens_generated: int
    tokens_per_second: float

def load_model():
    """Load the GGUF model on startup"""
    global llm_model
    
    # Model path - adjust this to your GGUF model path
    model_path = "models/airavata.Q4_K_M.gguf"  # Update with your actual model path
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    try:
        logger.info("Loading GGUF model...")
        llm_model = Llama(
            model_path=model_path,
            n_ctx=2048,  # Context window
            n_threads=4,  # CPU threads (adjust based on your CPU)
            n_gpu_layers=0,  # Set to 0 for CPU-only, increase if GPU available
            verbose=False
        )
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise e

@app.on_event("startup")
async def startup_event():
    """Load model when the app starts"""
    load_model()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if llm_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.post("/generate", response_model=ChatResponse)
async def generate_response(request: ChatRequest):
    """Generate response from the model"""
    if llm_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start_time = time.time()
        
        # Format the prompt
        system_prompt = """You are Airavata, a helpful AI assistant that can understand and respond in both Hindi and English. Respond naturally and helpfully to the user's query."""
        
        formatted_prompt = f"""<|system|>
{system_prompt}
<|user|>
{request.message}
<|assistant|>
"""
        
        # Generate response
        response = llm_model(
            formatted_prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            echo=False,
            stop=["<|user|>", "<|system|>"]
        )
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        # Extract the generated text
        generated_text = response['choices'][0]['text'].strip()
        tokens_generated = response['usage']['completion_tokens']
        tokens_per_second = tokens_generated / (latency_ms / 1000) if latency_ms > 0 else 0
        
        return ChatResponse(
            response=generated_text,
            latency_ms=round(latency_ms, 2),
            tokens_generated=tokens_generated,
            tokens_per_second=round(tokens_per_second, 2)
        )
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Airavata GGUF Chat API is running!", "docs": "/docs"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)