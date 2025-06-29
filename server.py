from cefr_bert_classifier import CEFRTextAnalyzer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
import os
import sys
import asyncio

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Global variable to store the model
model = None


def load_model():
    """Load the CEFR model at startup"""
    global model
    print("🚀 Loading CEFR Text Analyzer Model...")

    try:
        model = CEFRTextAnalyzer(
            model_name='bert-base-uncased',
            max_length=128,
            batch_size=1,  # Single prediction
            learning_rate=2e-5
        )

        model_path = 'cefr_bert_model.pth'
        if os.path.exists(model_path):
            print(f"📁 Loading model from {model_path}...")
            model.load_model(model_path)
            print("✅ Model loaded successfully!")
        else:
            print("❌ Model file not found. Please train the model first.")
            print("Run: python train.py")
            return False

        print("🎯 CEFR Model is ready!")
        return True

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        model = None
        return False


# Load model at startup
print("🌟 Initializing CEFR Text Analyzer API Server...")
model_loaded = load_model()

# Create FastAPI app
app = FastAPI(
    title="CEFR Text Analyzer API",
    description="API for classifying English text proficiency levels according to CEFR standards (A1-C2)",
    version="1.0.0"
)

# Request model


class PredictionRequest(BaseModel):
    sentences: List[str]

# Response model


class PredictionResult(BaseModel):
    cefr: str
    sentence: str
    confidence: float


class PredictionResponse(BaseModel):
    predictions: List[PredictionResult]


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "CEFR Text Analyzer API",
        "status": "ready",
        "version": "1.0.0",
        "description": "POST to /predict with sentences to get CEFR level predictions"
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    global model
    if model is None:
        return {"status": "unhealthy", "model_loaded": False, "error": "Model not loaded"}
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=List[PredictionResult])
async def predict_cefr_levels(request: PredictionRequest):
    """
    Predict CEFR levels for input sentences

    Args:
        request: PredictionRequest containing list of sentences

    Returns:
        List of predictions with CEFR level, sentence, and confidence score
    """
    global model

    if model is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Please check server logs and ensure model file exists.")

    if not request.sentences:
        raise HTTPException(status_code=400, detail="No sentences provided")

    try:
        predictions = []

        for sentence in request.sentences:
            if not sentence or not sentence.strip():
                # Skip empty sentences
                continue

            # Get prediction from model
            cefr_level, confidence = model.predict_text(sentence.strip())

            prediction_result = PredictionResult(
                cefr=cefr_level,
                sentence=sentence.strip(),
                confidence=float(confidence)
            )

            predictions.append(prediction_result)

        if not predictions:
            raise HTTPException(
                status_code=400, detail="No valid sentences to process")

        return predictions

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/levels")
async def get_cefr_levels():
    """Get information about CEFR levels"""
    return {
        "levels": {
            "A1": {
                "name": "Beginner",
                "description": "Basic everyday expressions and very simple phrases"
            },
            "A2": {
                "name": "Elementary",
                "description": "Simple sentences on familiar topics and routine matters"
            },
            "B1": {
                "name": "Intermediate",
                "description": "Clear communication on familiar matters and personal interests"
            },
            "B2": {
                "name": "Upper Intermediate",
                "description": "Complex texts and abstract topics with good fluency"
            },
            "C1": {
                "name": "Advanced",
                "description": "Flexible and effective language use for social, academic and professional purposes"
            },
            "C2": {
                "name": "Proficient",
                "description": "Very high level with precise, nuanced expression and full command of language"
            }
        }
    }

if __name__ == "__main__":
    if not model_loaded:
        print("❌ Cannot start server: Model failed to load")
        print("Please ensure:")
        print("1. Run 'python train.py' to train the model first")
        print("2. Check that 'cefr_bert_model.pth' exists")
        sys.exit(1)

    print("🌟 CEFR Text Analyzer API Server")
    print("=" * 50)
    print("📍 Server will run on: http://localhost:5050")
    print("📝 POST to /predict with JSON: {'sentences': ['your text here']}")
    print("🔍 GET /levels for CEFR level information")
    print("💚 GET /health for health check")
    print("=" * 50)

    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=5050,
            log_level="info"
        )
    except Exception as e:
        print(f"❌ Server failed to start: {e}")
        sys.exit(1)
