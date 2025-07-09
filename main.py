from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from llama_index.core import StorageContext, load_index_from_storage
import requests

app = FastAPI()
#app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL ="deepseek-r1"  # or deepseek-coder, etc.

index = load_index_from_storage(StorageContext.from_defaults(persist_dir="./storage"))
query_engine = index.as_query_engine()

conversation_history = []

def build_prompt(user_input: str, context_text: str) -> str:
    history = ""
    for entry in conversation_history[-5:]:
        history += f"User: {entry['user']}\nBot: {entry['bot']}\n"
    return f"{history}\nRetrieved Info:\n{context_text}\nUser: {user_input}\nBot:"

def query_ollama(prompt: str) -> str:
    try:
        res = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            headers={"Content-Type": "application/json"}
        )
        res.raise_for_status()
        return res.json().get("response", "No response generated.")
    except Exception as e:
        return f"Error querying Ollama: {e}"

@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat", response_class=JSONResponse)
async def chat(user_input: str = Form(...)):
    response_obj = query_engine.query(user_input)
    retrieved_context = str(response_obj)
    prompt = build_prompt(user_input, retrieved_context)
    answer = query_ollama(prompt)
    conversation_history.append({"user": user_input, "bot": answer})
    return {"response": answer}
