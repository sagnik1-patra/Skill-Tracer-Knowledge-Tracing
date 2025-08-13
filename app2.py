
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, conlist
from typing import List, Optional, Union
from predictor import predict, model_info
from recommender import recommend

app = FastAPI(title="SkillTracer API (with Recommender)", version="1.1.0",
              description="Predict next correctness AND recommend next items.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

class HistoryEvent(BaseModel):
    skill: str = Field(..., description="Skill/concept")
    correct: int = Field(..., ge=0, le=1)

class PredictRequest(BaseModel):
    history: List[Union[HistoryEvent, conlist(Union[str, int], min_items=2, max_items=2)]]
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0)

class PredictResponse(BaseModel):
    probability: Optional[float]
    threshold: float
    predicted_class: int
    note: Optional[str] = None

class RecommendRequest(BaseModel):
    history: List[Union[HistoryEvent, conlist(Union[str, int], min_items=2, max_items=2)]]
    top_k: int = Field(5, ge=1, le=20)
    target_low: float = Field(0.60, ge=0.0, le=1.0)
    target_high: float = Field(0.75, ge=0.0, le=1.0)
    min_item_count: int = Field(30, ge=1, description="Only consider items seen at least this many times in training")

@app.get("/health")
def health():
    return {"status": "ok", "model": model_info()}

@app.post("/predict", response_model=PredictResponse)
def predict_route(body: PredictRequest):
    try:
        normalized = []
        for e in body.history:
            if isinstance(e, list) or isinstance(e, tuple):
                normalized.append({"skill": str(e[0]), "correct": int(e[1])})
            else:
                normalized.append({"skill": e.skill, "correct": e.correct})
        return predict(normalized, threshold=body.threshold)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/recommend")
def recommend_route(body: RecommendRequest):
    try:
        normalized = []
        for e in body.history:
            if isinstance(e, list) or isinstance(e, tuple):
                normalized.append({"skill": str(e[0]), "correct": int(e[1])})
            else:
                normalized.append({"skill": e.skill, "correct": e.correct})
        recs = recommend(normalized, top_k=body.top_k,
                         target_low=body.target_low, target_high=body.target_high,
                         min_item_count=body.min_item_count)
        return {"recommendations": recs}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def root():
    return {
        "hello": "SkillTracer API (with Recommender)",
        "docs": "/docs",
        "try_predict": {
            "history": [["Algebra",1], ["Algebra",0], ["Fractions",1]],
            "threshold": 0.5
        },
        "try_recommend": {
            "history": [["Algebra",1], ["Algebra",0], ["Fractions",1]],
            "top_k": 5, "target_low": 0.60, "target_high": 0.75
        }
    }

# Run: uvicorn app2:app --host 0.0.0.0 --port 8000
