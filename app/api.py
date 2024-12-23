from fastapi import FastAPI
from src.infer import rank_all_queries
from src.infer import rank_query_string
from src.infer import rank_query_string
import os

app = FastAPI() 

@app.get("/")
def home():
    return {"message": "AI-Powered Search Ranking API"}

@app.get("/rank")
def rank():
    data_path = "data/preprocessed_data.npz"
    model_path = "models/ranking_model.pth"

    # Check if files exist
    if not os.path.exists(data_path):
        return {"error": f"Data file not found at {data_path}"}
    if not os.path.exists(model_path):
        return {"error": f"Model file not found at {model_path}"}

    # Proceed with ranking
    results = rank_all_queries(data_path, model_path)
    return {"rankings": results if results else "No rankings available"}  

@app.get("/rank_query")
def rank_q(query: str):
    model_path = "models/bert_ranking_model.pth"
    original_path = "data/questions.csv"

    # Check if files exist
    if not os.path.exists(data_path):
        return {"error": f"Data file not found at {data_path}"}
    if not os.path.exists(model_path):
        return {"error": f"Model file not found at {model_path}"}

    # Proceed with ranking
    results = find_similar_questions(
        query_question=query,
        model_path=model_path,
        questions_path=original_path,
        n=5, 
        max_questions=1000
    )
    return {"rankings": results if results else "No rankings available"} 