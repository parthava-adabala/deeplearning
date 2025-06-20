from fastapi import FastAPI
from pydantic import BaseModel
from sentiment_analysis import analyze_sentiment

app = FastAPI()

class TextToAnalyze(BaseModel):
    text: str

@app.post("/analyze/")
async def analyze(data: TextToAnalyze):
    return analyze_sentiment(data.text) 