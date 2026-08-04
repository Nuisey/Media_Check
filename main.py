# Start Command for Render.com: uvicorn main:app --host 0.0.0.0 --port $PORT

import os
import json
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Media Fact Checker API")

# Enable CORS (useful if frontend is hosted separately, though we serve it together)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Serve the static HTML file
@app.get("/")
async def read_index():
    # Looks for index.html in the same directory as main.py
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="index.html not found. Make sure it's in the same directory.")
    return FileResponse(index_path)

class AnalyzeRequest(BaseModel):
    url: str

@app.post("/api/analyze")
async def analyze_video(request: AnalyzeRequest):
    url = request.url
    supadata_api_key = os.environ.get("SUPADATA_API_KEY")
    
    if not supadata_api_key:
        raise HTTPException(status_code=500, detail="Supadata API key not configured on server")
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API key not configured on server")

    # 1. Fetch transcript from Supadata API
    transcript_text = ""
    try:
        # Note: Adjust endpoint if Supadata updates its Instagram transcript API
        supadata_url = "https://api.supadata.ai/v1/transcript"
        headers = {"x-api-key": supadata_api_key}
        params = {"url": url}
        
        async with httpx.AsyncClient() as client:
            response = await client.get(supadata_url, headers=headers, params=params, timeout=30.0)
            
        if response.status_code == 200:
            data = response.json()
            # Try to safely extract transcript from standard Supadata JSON format
            if "content" in data:
                transcript_text = data["content"]
            elif "transcript" in data:
                transcript_text = data["transcript"]
            elif "text" in data:
                transcript_text = data["text"]
            else:
                transcript_text = str(data)
        else:
            raise HTTPException(status_code=response.status_code, detail=f"Failed to fetch transcript: {response.text}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with Supadata: {str(e)}")

    if not transcript_text or transcript_text == "{}":
        raise HTTPException(status_code=400, detail="Could not extract transcript from the provided URL.")

    # 2. Analyze transcript using Gemini API
    try:
        def generate_with_fallback(prompt_text):
            config = {"response_mime_type": "application/json"}
            try:
                model = genai.GenerativeModel(model_name="gemini-3.6-flash", generation_config=config)
                return model.generate_content(prompt_text)
            except Exception as e:
                print(f"Warning: gemini-3.6-flash failed ({e}). Falling back to gemini-3.5-flash.")
                fallback_model = genai.GenerativeModel(model_name="gemini-3.5-flash", generation_config=config)
                return fallback_model.generate_content(prompt_text)

        platform = "YouTube video" if "youtube.com" in url or "youtu.be" in url else "Instagram Reel"
        
        extract_prompt = f"""
        You are an expert analyst. Read the entire transcript from a {platform} and extract EVERY single factual claim made. 
        CRITICAL INSTRUCTIONS:
        1. Maintain chronological order. The main claims should flow from the beginning to the end of the video.
        2. Maintain a deep hierarchy. Identify the overarching main claims, and for each main claim, extract the nuanced sub-claims, sub-sub-claims, etc. made to support it.

        Transcript: {transcript_text}
        
        Respond in ONLY valid JSON matching this schema:
        {{
          "extracted_claims": [
            {{
              "claim": "Main claim text",
              "sub_claims": [
                {{
                  "claim": "Sub claim text",
                  "sub_claims": []
                }}
              ]
            }}
          ]
        }}
        """
        extract_response = generate_with_fallback(extract_prompt)
        try:
            extracted_json = json.loads(extract_response.text)
            claims_list = extracted_json.get("extracted_claims", [])
        except json.JSONDecodeError:
            claims_list = []
            
        prompt = f"""
        You are an expert fact-checker. Fact-check the following hierarchical chronological claims made in a {platform}.
        Claims to check: {json.dumps(claims_list)}
        Original Transcript for context: {transcript_text}
        
        CRITICAL INSTRUCTION: You MUST preserve the exact hierarchical structure (claims, sub-claims, sub-sub-claims, etc.) and chronological order provided.
        
        You must respond in ONLY valid JSON matching this exact schema:
        {{
          "summary": "A 2-sentence overview of the video's core message.",
          "claims": [
            {{
              "claim": "The specific claim made",
              "verdict": "True | False | Misleading | Unverified",
              "explanation": "Brief explanation of why the verdict was given",
              "sub_claims": [
                 // Same structure for any nested sub-claims
              ]
            }}
          ]
        }}
        """
        
        ai_response = generate_with_fallback(prompt)
        
        # Validate that we can parse the JSON before sending
        result_json = json.loads(ai_response.text)
        return result_json
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse AI response as JSON")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during AI analysis: {str(e)}")
