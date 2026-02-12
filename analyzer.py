import os
import tempfile
import moviepy as mp
from openai import OpenAI
import azure.cognitiveservices.speech as speechsdk
from pydub import AudioSegment
from pydub.utils import make_chunks


class VideoTalentAnalyzer:
    """
    Analyzes video content to generate candidate profiles.
    
    Uses:
    - MoviePy for audio extraction
    - OpenAI Whisper for transcription
    - Azure Cognitive Services for pronunciation assessment
    - GPT-4 for profile generation
    """
    
    def __init__(self, video_path: str, temp_dir: str = None):
        """
        Initialize the analyzer.
        
        Args:
            video_path: Path to the video file
            temp_dir: Directory for temporary files (default: system temp)
        """
        self.video_path = video_path
        self.temp_dir = temp_dir or tempfile.gettempdir()
        
        # Generate unique audio path to avoid conflicts
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        self.audio_path = os.path.join(self.temp_dir, f"{video_basename}_audio.wav")
        
        # Initialize OpenAI client
        self.client = OpenAI()
        
        # State
        self.transcript_text = ""
        self.pronunciation_metrics = {}
        
        # Get Azure credentials from environment
        self.azure_speech_key = os.getenv("AZURE_SPEECH_KEY")
        self.azure_service_region = os.getenv("AZURE_SERVICE_REGION")

    def extract_audio(self) -> bool:
        """
        Extracts audio from the video file and saves it as a WAV file.
        
        Returns:
            bool: True if successful, False otherwise
        """
        print(f"[1/4] Extracting audio from video...")
        try:
            clip = mp.VideoFileClip(self.video_path)
            clip.audio.write_audiofile(
                self.audio_path, 
                codec='pcm_s16le',
                fps=16000,  # 16kHz for Azure compatibility
                logger=None
            )
            clip.close()
            print(" -> Audio extracted successfully.")
            return True
        except Exception as e:
            print(f"Error extracting audio: {e}")
            raise RuntimeError(f"Failed to extract audio: {e}")

    def transcribe_content(self) -> str:
        """
        Uses OpenAI Whisper to transcribe the audio content.
        Automatically splits large files into chunks.
        
        Returns:
            str: The transcription text
        """
        print("[2/4] Transcribing content with OpenAI Whisper...")
        
        MAX_FILE_SIZE = 24 * 1024 * 1024  # 24 MB to be safe
        
        try:
            file_size = os.path.getsize(self.audio_path)
            
            if file_size <= MAX_FILE_SIZE:
                print(f" -> File size: {file_size / (1024*1024):.2f} MB (within limit)")
                with open(self.audio_path, "rb") as audio_file:
                    transcription = self.client.audio.transcriptions.create(
                        model="whisper-1", 
                        file=audio_file
                    )
                self.transcript_text = transcription.text
                print(" -> Transcription complete.")
            else:
                print(f" -> File size: {file_size / (1024*1024):.2f} MB (exceeds limit)")
                print(" -> Splitting audio into chunks...")
                
                audio = AudioSegment.from_wav(self.audio_path)
                duration_ms = len(audio)
                bytes_per_ms = file_size / duration_ms
                
                target_chunk_size = 20 * 1024 * 1024
                chunk_duration_ms = int(target_chunk_size / bytes_per_ms)
                
                chunks = make_chunks(audio, chunk_duration_ms)
                print(f" -> Split into {len(chunks)} chunks")
                
                full_transcript = []
                for i, chunk in enumerate(chunks):
                    print(f" -> Transcribing chunk {i+1}/{len(chunks)}...")
                    
                    chunk_path = os.path.join(self.temp_dir, f"temp_chunk_{i}.wav")
                    chunk.export(chunk_path, format="wav")
                    
                    with open(chunk_path, "rb") as audio_file:
                        transcription = self.client.audio.transcriptions.create(
                            model="whisper-1", 
                            file=audio_file
                        )
                    full_transcript.append(transcription.text)
                    
                    # Clean up chunk immediately
                    if os.path.exists(chunk_path):
                        os.remove(chunk_path)
                
                self.transcript_text = " ".join(full_transcript)
                print(" -> Transcription complete (all chunks processed).")
                
            return self.transcript_text
            
        except Exception as e:
            print(f"Error during transcription: {e}")
            raise RuntimeError(f"Failed to transcribe audio: {e}")

    def analyze_accent_with_azure(self) -> dict:
        """
        Uses Azure Cognitive Services to assess pronunciation.
        
        Returns:
            dict: Pronunciation metrics (Accuracy, Fluency, Prosody, Overall)
        """
        print("[3/4] Analyzing accent and fluency with Azure AI...")
        
        if not self.azure_speech_key or not self.azure_service_region:
            print(" -> Azure credentials not configured, skipping accent analysis")
            self.pronunciation_metrics = {
                "Accuracy": None,
                "Fluency": None,
                "Prosody": None,
                "Overall": None,
                "note": "Azure credentials not configured"
            }
            return self.pronunciation_metrics
        
        try:
            speech_config = speechsdk.SpeechConfig(
                subscription=self.azure_speech_key, 
                region=self.azure_service_region
            )
            audio_config = speechsdk.audio.AudioConfig(filename=self.audio_path)

            pronunciation_config = speechsdk.PronunciationAssessmentConfig(
                reference_text="", 
                grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
                granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme,
                enable_miscue=True
            )

            recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config, 
                audio_config=audio_config,
                language="en-US"
            )

            pronunciation_config.apply_to(recognizer)
            result = recognizer.recognize_once_async().get()

            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                pronunciation_result = speechsdk.PronunciationAssessmentResult(result)
                
                self.pronunciation_metrics = {
                    "Accuracy": pronunciation_result.accuracy_score,
                    "Fluency": pronunciation_result.fluency_score,
                    "Prosody": pronunciation_result.prosody_score,
                    "Overall": pronunciation_result.pronunciation_score
                }
                print(f" -> Azure Analysis: Fluency {self.pronunciation_metrics['Fluency']}/100")
            
            elif result.reason == speechsdk.ResultReason.NoMatch:
                print(" -> No speech could be recognized for accent analysis.")
                self.pronunciation_metrics = {"Overall": 0, "Fluency": 0}
                
            else:
                error_msg = getattr(result.cancellation_details, 'error_details', 'Unknown error')
                print(f" -> Azure Error: {error_msg}")
                self.pronunciation_metrics = {"Overall": 0, "Fluency": 0, "error": error_msg}
                
            return self.pronunciation_metrics
            
        except Exception as e:
            print(f"Error during accent analysis: {e}")
            self.pronunciation_metrics = {"error": str(e)}
            return self.pronunciation_metrics

    def generate_final_profile(self) -> str:
        """
        Uses GPT-4 to generate a candidate profile based on transcript and metrics.
        
        Returns:
            str: The generated profile
        """
        print("[4/4] Generating Candidate Insights with GPT-4...")

        metrics_str = f"""
        Technical English Analysis (0-100 scale):
        - Fluency Score: {self.pronunciation_metrics.get('Fluency', 'N/A')}
        - Pronunciation Accuracy: {self.pronunciation_metrics.get('Accuracy', 'N/A')}
        - Overall Score: {self.pronunciation_metrics.get('Overall', 'N/A')}
        (Note: >85 is Native-like, >70 is Advanced/Professional, <60 is Intermediate)
        """

        system_prompt = """
        You are an expert Technical Recruiter and Talent Analyst.
        Your goal is to analyze a video transcript and technical audio metrics to create a candidate profile.
        
        You will receive:
        1. A transcript of what the candidate said.
        2. Technical metrics regarding their English accent and fluency (from Azure AI).

        OUTPUT FORMAT (Strictly follow this):
        [Candidate Name (or 'Candidate' if unknown)]
        - [Current Role / Years of Experience (infer if not explicit)]
        - [Key specific achievement or technical strength]
        - [Financial metrics found (e.g., Revenue growth, P&L management)]
        - [Mission or motivation mentioned]
        - [English Level]: Combine the Azure metrics and the transcript vocabulary. (e.g., "Excellent English (Fluency: 95/100)", or "Good English with minor accent").
        - [Communication Style]: Brief conclusion based on clarity and organization.
        - [Seniority]: (e.g., "Looks Senior", "Mid-Level", etc.) based on achievements.
        """

        user_prompt = f"""
        TRANSCRIPT:
        {self.transcript_text}

        AUDIO METRICS:
        {metrics_str}

        Please generate the profile.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.4
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating profile: {e}")
            raise RuntimeError(f"Failed to generate profile: {e}")

    def clean_up(self):
        """Remove temporary audio files."""
        # Remove main audio file
        if os.path.exists(self.audio_path):
            os.remove(self.audio_path)
        
        # Remove any leftover chunk files
        for file in os.listdir(self.temp_dir):
            if file.startswith('temp_chunk_') and file.endswith('.wav'):
                try:
                    os.remove(os.path.join(self.temp_dir, file))
                except:
                    pass
        
        print("Temporary files cleaned.")

    def run(self) -> dict:
        """
        Run the complete analysis pipeline.
        
        Returns:
            dict: Contains profile, transcript, and pronunciation_metrics
        """
        try:
            self.extract_audio()
            self.transcribe_content()
            self.analyze_accent_with_azure()
            profile = self.generate_final_profile()
            
            return {
                "profile": profile,
                "transcript": self.transcript_text,
                "pronunciation_metrics": self.pronunciation_metrics
            }
        finally:
            self.clean_up()


# --- CLI for local testing ---
if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    
    load_dotenv()
    
    if len(sys.argv) < 2:
        print("Usage: python analyzer.py <video_path>")
        sys.exit(1)
    
    video_file = sys.argv[1]
    
    if not os.path.exists(video_file):
        print(f"File {video_file} not found.")
        sys.exit(1)
    
    analyzer = VideoTalentAnalyzer(video_file)
    result = analyzer.run()
    
    print("\n" + "="*40)
    print("FINAL CANDIDATE ANALYSIS")
    print("="*40)
    print(result["profile"])

