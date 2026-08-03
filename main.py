# Start Command for Render.com: uvicorn main:app --host 0.0.0.0 --port $PORT

import asyncio
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
        model = genai.GenerativeModel(
            model_name="gemini-3.6-flash", # Using gemini-3.6-flash as requested
            generation_config={
                "response_mime_type": "application/json",
            }
        )
        
        platform = "YouTube video" if "youtube.com" in url or "youtu.be" in url else "Instagram Reel"
        
        async def run_single():
            prompt = f"""
            You are an expert fact-checker. Analyze the following transcript from a {platform}. 
            CRITICAL INSTRUCTION: You must read the ENTIRE transcript from start to finish. Extract and fact-check EVERY single verifiable claim made throughout the entire video. Do not stop early.
            
            Transcript:
            {transcript_text}
            
            You must respond in ONLY valid JSON matching this exact schema:
            {{
              "summary": "A 2-sentence overview of the video's core message.",
              "claims": [
                {{
                  "claim": "The specific claim made in the video",
                  "verdict": "True | False | Misleading | Unverified",
                  "explanation": "Brief explanation of why the verdict was given"
                }}
              ]
            }}
            """
            resp = await model.generate_content_async(prompt)
            return json.loads(resp.text)

        async def run_two_step():
            extract_prompt = f"""
            You are an expert analyst. Read the entire transcript from a {platform} and extract EVERY single factual claim made. Do not stop early.
            Transcript: {transcript_text}
            
            Respond in ONLY valid JSON matching this schema:
            {{ "extracted_claims": ["claim 1", "claim 2"] }}
            """
            extract_resp = await model.generate_content_async(extract_prompt)
            try:
                extracted_json = json.loads(extract_resp.text)
                claims_list = extracted_json.get("extracted_claims", [])
            except json.JSONDecodeError:
                claims_list = []
                
            prompt = f"""
            You are an expert fact-checker. Fact-check the following claims made in a {platform}.
            Claims to check: {json.dumps(claims_list)}
            Original Transcript for context: {transcript_text}
            
            You must respond in ONLY valid JSON matching this exact schema:
            {{
              "summary": "A 2-sentence overview of the video's core message.",
              "claims": [
                {{
                  "claim": "The specific claim made in the video",
                  "verdict": "True | False | Misleading | Unverified",
                  "explanation": "Brief explanation of why the verdict was given"
                }}
              ]
            }}
            """
            check_resp = await model.generate_content_async(prompt)
            return json.loads(check_resp.text)

        single_res, two_step_res = await asyncio.gather(run_single(), run_two_step())
        
        return {
            "single": single_res,
            "two_step": two_step_res
        }
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse AI response as JSON")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during AI analysis: {str(e)}")
