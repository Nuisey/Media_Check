# MediaCheck - AI Media Fact Checker

MediaCheck is an AI-powered fact-checking web application designed to analyze and verify claims made in social media videos (Instagram Reels and YouTube videos). It extracts transcripts, breaks them down into hierarchical claims, and uses Google's Gemini models to provide fact-checking verdicts and evidence.
<img width="1367" height="473" alt="image" src="https://github.com/user-attachments/assets/699e5bd5-01a6-47cc-8b78-7fd214e8063c" />

## Features
- **Video Transcript Extraction:** Uses the Supadata API to fetch transcripts from YouTube or Instagram URLs.
- **Hierarchical Claim Extraction:** Utilizes Gemini AI to break down transcripts into deeply nested, logical claims and sub-claims rather than just chronological lists.
- **AI Fact-Checking:** Fact-checks every claim and sub-claim, categorizing them as True, False, Misleading, or Unverified, along with explanations and evidence strength ratings.
- **Modern UI:** A sleek, glassmorphic frontend built with Tailwind CSS, supporting deeply nested claims with expandable evidence sections.

## Tech Stack
- **Backend:** Python, FastAPI, Uvicorn, HTTPX
- **Frontend:** HTML, Tailwind CSS, Vanilla JavaScript
- **AI Integration:** Google Generative AI (Gemini 3.6 Flash / 3.5 Flash fallback)
- **External APIs:** Supadata (Transcript extraction)

## Prerequisites
- Python 3.8+
- [Google Gemini API Key](https://aistudio.google.com/)
- [Supadata API Key](https://supadata.ai/)

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd Media_Fact_Checker
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   Create a `.env` file in the root directory and add your API keys:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   SUPADATA_API_KEY=your_supadata_api_key_here
   ```

4. **Run the application:**
   ```bash
   uvicorn main:app --reload
   ```

5. **Access the application:**
   Open your browser and navigate to `http://localhost:8000`.

## Deployment
The application is configured to run easily on cloud platforms like Render. The start command is:
```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

## How It Works
1. User submits a YouTube or Instagram video URL via the frontend.
2. The FastAPI backend calls the Supadata API to extract the video's transcript.
3. The transcript is passed to Gemini, which first extracts all factual claims into a structured JSON hierarchy.
4. Gemini is called a second time to fact-check each claim in the hierarchy.
5. The structured results are sent back to the frontend, which renders the verdicts (True, False, Misleading, Unverified), explanations, and evidence strength.
