from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import pickle
import io
import matplotlib.pyplot as plt
import seaborn as sns
import uuid

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    print(f"Failed to load model: {e}")

@app.get("/", response_class=HTMLResponse)
async def form_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/", response_class=HTMLResponse)
async def handle_upload(request: Request, file: UploadFile = File(...)):

    if file.content_type != "text/csv":
        return templates.TemplateResponse("index.html", {"request": request, "result": "File must be CSV."})

    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        predictions = model.predict(df)
        prediction_list = predictions.tolist()

        # Feature Importances
        plt.figure(figsize=(10, 6))
        sns.barplot(x=model.feature_importances_, y=df.columns)
        plt.title("Feature Importances")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plot1_path = f"static/feature_importance_{uuid.uuid4().hex}.png"
        plt.tight_layout()
        plt.savefig(plot1_path)
        plt.close()

        # Actual vs Predicted (dùng giá trị giả lập nếu không có target)
        actual = [0] * len(predictions)  # giả sử không có giá trị thực
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=actual, y=predictions)
        plt.plot([min(predictions), max(predictions)], [min(predictions), max(predictions)], 'r--')
        plt.title("Actual vs Predicted")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plot2_path = f"static/actual_vs_predicted_{uuid.uuid4().hex}.png"
        plt.tight_layout()
        plt.savefig(plot2_path)
        plt.close()

        return templates.TemplateResponse("result.html", {
            "request": request,
            "plot1": "/" + plot1_path,
            "plot2": "/" + plot2_path,
            "predictions": predictions.tolist()
        })
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "result": f"Error: {str(e)}"})
