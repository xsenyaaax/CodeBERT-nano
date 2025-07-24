from src.app.interface import demo
from src.utils.inference import predict
from src.config.config import language_logos
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import gradio as gr

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def index(request: Request):
    # Just redirect straight to gradio page
    return RedirectResponse(url="/gradio/")

@app.route(
    "/attention-viz",
)
async def attention_viz(request: Request):
    code = request.query_params.get("code", "")
    max_tokens = request.query_params.get("max_tokens")

    if not code.strip():
        return HTMLResponse("<h2>No code provided.</h2>", status_code=400)
    max_tokens = int(max_tokens) if max_tokens and max_tokens.isdigit() else None

    ret = predict(code, return_attention=True, max_tokens_in_attn=max_tokens)
    print(f"{ret['attention_html']=}")
    logo_path = language_logos[ret["label"]]
    return templates.TemplateResponse(
        "attention.html",
        {
            "request": request,
            "code": code,
            "label": ret["label"],
            "attention_html": ret["attention_html"],
            "logo_path": logo_path,
            "max_tokens": max_tokens or 30
        },
    )


gr.mount_gradio_app(app, demo, path="/gradio")


