from pathlib import Path
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.rag import retrieve, draft_grounded_answer

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "app" / "templates"
STATIC_DIR = BASE_DIR / "app" / "static"

app = FastAPI(title="SY335C RAG Assistant")

# Mount static files: /static/style.css
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"result": None})


@app.get("/ask", response_class=HTMLResponse)
def ask_page(request: Request):
    # supports opening /ask directly
    return templates.TemplateResponse(request=request, name="index.html", context={"result": None})


@app.post("/ask", response_class=HTMLResponse)
def ask(request: Request, query: str = Form(...)):
    evidence = retrieve(query)
    result = draft_grounded_answer(query, evidence)
    return templates.TemplateResponse(request=request, name="index.html", context={"result": result, "query": query})
