import os
import uuid
import tempfile
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from analyzer import VideoTalentAnalyzer

# --- Lifespan for startup/shutdown ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: verify environment variables
    required_vars = ["AZURE_SPEECH_KEY", "AZURE_SERVICE_REGION", "OPENAI_API_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        print(f"⚠️  Warning: Missing environment variables: {missing}")
    else:
        print("✅ All required environment variables are set")
    yield
    # Shutdown: cleanup if needed
    print("🔄 Shutting down...")


app = FastAPI(
    title="Video Talent Analyzer API",
    description="Analyzes video content to generate candidate profiles using AI",
    version="1.0.0",
    lifespan=lifespan
)


# --- Response Models ---
class AnalysisResponse(BaseModel):
    success: bool
    profile: Optional[str] = None
    transcript: Optional[str] = None
    pronunciation_metrics: Optional[dict] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str


# --- Endpoints ---
@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint for Cloud Run"""
    return HealthResponse(status="healthy", version="1.0.0")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(status="healthy", version="1.0.0")


def cleanup_temp_file(file_path: str):
    """Background task to clean up temporary files"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"🧹 Cleaned up: {file_path}")
    except Exception as e:
        print(f"⚠️  Error cleaning up {file_path}: {e}")


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(..., description="Video file to analyze (MP4, MOV, etc.)")
):
    """
    Analyze a video file to generate a candidate profile.
    
    This endpoint:
    1. Extracts audio from the video
    2. Transcribes content using OpenAI Whisper
    3. Analyzes pronunciation with Azure AI
    4. Generates a candidate profile with GPT-4
    
    Supported formats: MP4, MOV, AVI, MKV, WebM
    Max recommended size: 500MB
    """
    
    # Validate file type
    allowed_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'}
    file_ext = os.path.splitext(video.filename or "")[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file_ext}'. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Create temporary file path
    # Use /tmp for Cloud Run (it's the only writable directory)
    temp_dir = tempfile.gettempdir()
    unique_id = str(uuid.uuid4())
    temp_video_path = os.path.join(temp_dir, f"video_{unique_id}{file_ext}")
    
    try:
        # Save uploaded file to disk
        print(f"📥 Receiving video: {video.filename}")
        
        with open(temp_video_path, "wb") as buffer:
            # Read in chunks to handle large files
            chunk_size = 1024 * 1024  # 1MB chunks
            while True:
                chunk = await video.read(chunk_size)
                if not chunk:
                    break
                buffer.write(chunk)
        
        file_size_mb = os.path.getsize(temp_video_path) / (1024 * 1024)
        print(f"📁 Video saved: {file_size_mb:.2f} MB")
        
        # Run analysis
        analyzer = VideoTalentAnalyzer(temp_video_path, temp_dir=temp_dir)
        result = analyzer.run()
        
        # Schedule cleanup in background
        background_tasks.add_task(cleanup_temp_file, temp_video_path)
        
        return AnalysisResponse(
            success=True,
            profile=result["profile"],
            transcript=result["transcript"],
            pronunciation_metrics=result["pronunciation_metrics"]
        )
        
    except Exception as e:
        # Ensure cleanup on error
        background_tasks.add_task(cleanup_temp_file, temp_video_path)
        
        print(f"❌ Error analyzing video: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing video: {str(e)}"
        )


@app.post("/analyze/transcript-only", response_model=AnalysisResponse)
async def analyze_transcript_only(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(..., description="Video file to transcribe")
):
    """
    Lightweight endpoint that only transcribes the video without pronunciation analysis.
    Faster and uses fewer resources.
    """
    
    allowed_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'}
    file_ext = os.path.splitext(video.filename or "")[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file_ext}'. Allowed: {', '.join(allowed_extensions)}"
        )
    
    temp_dir = tempfile.gettempdir()
    unique_id = str(uuid.uuid4())
    temp_video_path = os.path.join(temp_dir, f"video_{unique_id}{file_ext}")
    
    try:
        with open(temp_video_path, "wb") as buffer:
            chunk_size = 1024 * 1024
            while True:
                chunk = await video.read(chunk_size)
                if not chunk:
                    break
                buffer.write(chunk)
        
        analyzer = VideoTalentAnalyzer(temp_video_path, temp_dir=temp_dir)
        analyzer.extract_audio()
        analyzer.transcribe_content()
        transcript = analyzer.transcript_text
        analyzer.clean_up()
        
        background_tasks.add_task(cleanup_temp_file, temp_video_path)
        
        return AnalysisResponse(
            success=True,
            transcript=transcript
        )
        
    except Exception as e:
        background_tasks.add_task(cleanup_temp_file, temp_video_path)
        raise HTTPException(
            status_code=500,
            detail=f"Error transcribing video: {str(e)}"
        )


# --- Run with uvicorn for local development ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)

