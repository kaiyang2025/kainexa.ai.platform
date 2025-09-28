from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="Kainexa API",
    version="0.1.0",
    description="Kainexa AI Platform API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

# 루트 경로
@app.get("/")
def root():
    return {"message": "Kainexa API is running!", "version": "0.1.0"}

# Health check - 루트 레벨
@app.get("/health")
def health():
    return {"status": "healthy", "service": "kainexa-core"}

# Chat endpoint - 루트 레벨
@app.post("/chat")
def chat(request: ChatRequest):
    return ChatResponse(response=f"Echo: {request.message}")

# API v1 프리픽스 버전도 추가
@app.get("/api/v1/health")
def health_v1():
    return {"status": "healthy", "service": "kainexa-core", "version": "0.1.0"}

@app.post("/api/v1/chat")
def chat_v1(request: ChatRequest):
    return ChatResponse(response=f"Echo: {request.message}")

# Swagger 문서는 /docs에서 자동 제공
# ReDoc 문서는 /redoc에서 자동 제공

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
