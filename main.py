import os
import uuid
import tempfile
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
import requests

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
        print(f"WARNING: Missing environment variables: {missing}")
    else:
        print("All required environment variables are set")
    yield
    # Shutdown: cleanup if needed
    print("Shutting down...")


app = FastAPI(
    title="Video Talent Analyzer API",
    description="Analyzes video content from Supabase Storage to generate candidate profiles using AI",
    version="2.0.0",
    lifespan=lifespan
)


# --- Models ---
class AnalysisRequest(BaseModel):
    video_url: str


class AnalysisResponse(BaseModel):
    success: bool
    profile: Optional[str] = None
    transcript: Optional[str] = None
    pronunciation_metrics: Optional[dict] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str


# --- Helper Functions ---
def cleanup_temp_file(file_path: str):
    """Background task to clean up temporary files"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Cleaned up: {file_path}")
    except Exception as e:
        print(f"Error cleaning up {file_path}: {e}")


def download_from_supabase(url: str, output_path: str) -> bool:
    """
    Download a file from a Supabase Storage signed URL.
    """
    try:
        response = requests.get(url, stream=True, timeout=600)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return os.path.exists(output_path) and os.path.getsize(output_path) > 0

    except Exception as e:
        print(f"Error downloading file: {str(e)}")
        return False


# --- Endpoints ---
@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint for Cloud Run"""
    return HealthResponse(status="healthy", version="2.0.0")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(status="healthy", version="2.0.0")


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_video(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks
):
    """
    Analyze a video from a Supabase Storage signed URL

    Provide a Supabase Storage signed URL and get a complete candidate analysis.

    **Supported URL format:**
    - `https://<project>.supabase.co/storage/v1/object/sign/<bucket>/<path>?token=<jwt>`

    **What you get:**
    - Full transcript of the video
    - Pronunciation analysis metrics
    - AI-generated candidate profile

    **Supported video formats:** MP4, MOV, AVI, MKV, WebM, M4V
    """
    
    video_url = request.video_url.strip()
    
    if not video_url:
        raise HTTPException(
            status_code=400,
            detail="video_url is required"
        )
    
    # Validate it looks like a URL
    if not video_url.startswith(('http://', 'https://')):
        raise HTTPException(
            status_code=400,
            detail="Invalid URL. Must start with http:// or https://"
        )
    
    # Determine file extension (default to mp4)
    file_ext = ".mp4"
    url_path = video_url.split('?')[0]
    if '.' in url_path.split('/')[-1]:
        extracted_ext = os.path.splitext(url_path)[1].lower()
        allowed_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'}
        if extracted_ext in allowed_extensions:
            file_ext = extracted_ext
    
    temp_dir = tempfile.gettempdir()
    unique_id = str(uuid.uuid4())
    temp_video_path = os.path.join(temp_dir, f"video_{unique_id}{file_ext}")
    
    try:
        # Download video
        print(f"Downloading video from: {video_url[:80]}...")
        
        success = download_from_supabase(video_url, temp_video_path)

        if not success:
            raise HTTPException(
                status_code=400,
                detail="Failed to download video. Make sure the Supabase signed URL is valid and has not expired."
            )
        
        file_size_mb = os.path.getsize(temp_video_path) / (1024 * 1024)
        print(f"Video downloaded: {file_size_mb:.2f} MB")
        
        # Run analysis
        print("Starting video analysis...")
        analyzer = VideoTalentAnalyzer(temp_video_path, temp_dir=temp_dir)
        result = analyzer.run()
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_file, temp_video_path)
        
        print("Analysis complete!")
        
        return AnalysisResponse(
            success=True,
            profile=result["profile"],
            transcript=result["transcript"],
            pronunciation_metrics=result["pronunciation_metrics"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        background_tasks.add_task(cleanup_temp_file, temp_video_path)
        
        print(f"Error analyzing video: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing video: {str(e)}"
        )


# --- Run with uvicorn for local development ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
