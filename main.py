
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from llama_cpp import Llama
import time
import os
import json
from typing import Optional, Iterator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Airavata GGUF Chat API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    ttft_ms: float

def load_model():
    global llm_model
    model_path = "models/airavata.Q4_K_M.gguf"
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    try:
        logger.info("Loading GGUF model...")
        llm_model = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=0,
            verbose=False
        )
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise e

@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/health")
async def health_check():
    if llm_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

def generate_stream(request: ChatRequest) -> Iterator[str]:
    if llm_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    system_prompt = """You are Airavata, a helpful AI assistant that can understand and respond in both Hindi and English. Respond naturally and helpfully to the user's query."""
    
    formatted_prompt = f"""<|system|>
{system_prompt}
<|user|>
{request.message}
<|assistant|>
"""
    
    start_time = time.time()
    first_token_time = None
    tokens_generated = 0
    generated_text = ""
    
    try:
        stream = llm_model(
            formatted_prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            echo=False,
            stop=["<|user|>", "<|system|>"],
            stream=True
        )
        
        for token_data in stream:
            if first_token_time is None:
                first_token_time = time.time()
                ttft_ms = (first_token_time - start_time) * 1000
                
                yield f"data: {json.dumps({'type': 'ttft', 'ttft_ms': round(ttft_ms, 2)})}\n\n"
            
            token_text = token_data['choices'][0]['text']
            generated_text += token_text
            tokens_generated += 1
            
            yield f"data: {json.dumps({'type': 'token', 'text': token_text})}\n\n"
        
        end_time = time.time()
        total_latency_ms = (end_time - start_time) * 1000
        tokens_per_second = tokens_generated / (total_latency_ms / 1000) if total_latency_ms > 0 else 0
        
        final_metrics = {
            'type': 'complete',
            'response': generated_text.strip(),
            'latency_ms': round(total_latency_ms, 2),
            'tokens_generated': tokens_generated,
            'tokens_per_second': round(tokens_per_second, 2),
            'ttft_ms': round((first_token_time - start_time) * 1000, 2) if first_token_time else 0
        }
        
        yield f"data: {json.dumps(final_metrics)}\n\n"
        
    except Exception as e:
        logger.error(f"Error in stream generation: {str(e)}")
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

@app.post("/stream")
async def stream_response(request: ChatRequest):
    return StreamingResponse(
        generate_stream(request),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

@app.post("/generate", response_model=ChatResponse)
async def generate_response(request: ChatRequest):
    if llm_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start_time = time.time()
        
        system_prompt = """You are Airavata, a helpful AI assistant that can understand and respond in both Hindi and English. Respond naturally and helpfully to the user's query."""
        
        formatted_prompt = f"""<|system|>
{system_prompt}
<|user|>
{request.message}
<|assistant|>
"""
        
        first_token_time = None
        tokens_generated = 0
        generated_text = ""
        
        stream = llm_model(
            formatted_prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            echo=False,
            stop=["<|user|>", "<|system|>"],
            stream=True
        )
        
        for token_data in stream:
            if first_token_time is None:
                first_token_time = time.time()
            
            token_text = token_data['choices'][0]['text']
            generated_text += token_text
            tokens_generated += 1
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        tokens_per_second = tokens_generated / (latency_ms / 1000) if latency_ms > 0 else 0
        ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else 0
        
        return ChatResponse(
            response=generated_text.strip(),
            latency_ms=round(latency_ms, 2),
            tokens_generated=tokens_generated,
            tokens_per_second=round(tokens_per_second, 2),
            ttft_ms=round(ttft_ms, 2)
        )
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Airavata GGUF Chat API is running!", "docs": "/docs"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)