from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="CSV AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ok for task
    allow_methods=["*"],
    allow_headers=["*"],
)

CSV_PATH = "data/data.csv"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Load CSV ONCE (important)
df = pd.read_csv(CSV_PATH)
# df = df.sort_values(by=["Trade #", "Type"])
@app.get("/health")
def health():
    return {"status": "ok", "rows": len(df)}

@app.get("/data")
def get_data():
    return {
        "columns": df.columns.tolist(),
        "rows": df.to_dict(orient="records"),
    }

@app.post("/chat")
def chat(payload: dict):
    question = payload.get("question", "")
    history = payload.get("history", [])

    if not question:
        return {"answer": "Question is empty."}

    # --- Build messages properly (PER REQUEST) ---
    messages = [
        {
            "role": "system",
            "content": (
                "You are a trading data assistant. "
                "Answer ONLY using the provided CSV data. "
                "Do NOT use external knowledge or assumptions. "

                "Format your response using simple, clean HTML only. "
                "Use the following tags when appropriate: "
                "<h4>, <p>, <ul>, <li>, <strong>. "

                "Structure rules: "
                "- Start with a short summary inside <p>. "
                "- Use <h4> for section titles. "
                "- Use <ul><li> for grouped metrics. "
                "- Highlight important numbers using <strong>. "
                "- Do NOT use markdown. "
                "- Do NOT include <html>, <body>, or <script> tags. "
            ),
        }
    ]

    # --- Inject conversation history ---
    for msg in history:
        messages.append({
            "role": msg["role"],      # "user" or "assistant"
            "content": msg["content"]
        })

    # --- Add CSV context ONLY to current question ---
    context = df.to_json(orient="records")

    METADATA = {
        "total_rows": len(df),
        "columns": df.columns.tolist(),
        "min_price": float(df["Price USD"].min()),
        "max_price": float(df["Price USD"].max()),
    }

    messages.append({
        "role": "user",
        "content": (
            f"CSV Metadata:\n{METADATA}\n\n"
            f"CSV Data:\n{context}\n\n"
            f"Question:\n{question}"
        )
    })

    body = {
        "model": "openai/gpt-4o-mini",
        "messages": messages,
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "CSV AI Task",
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        json=body,
        headers=headers,
        timeout=30,
    )

    response.raise_for_status()

    answer = response.json()["choices"][0]["message"]["content"]
    return {"answer": answer}