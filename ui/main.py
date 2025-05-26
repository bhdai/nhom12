import traceback

from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import pickle
import io
import numpy as np
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

def extract_tire_size(x):
    if pd.isna(x): return -2
    if x == 'None or Unspecified': return -1
    if 'inch' in x: return float(x.split(' ')[0])
    return float(x.replace('"', ''))

train_df = pd.read_csv('../data/TrainAndValid.csv', parse_dates=['saledate'], low_memory=False)

keeps = ['YearMade', 'ProductSize', 'Coupler_System', 'fiProductClassDesc',
       'fiSecondaryDesc', 'saleElapsed', 'fiModelDesc', 'ModelID',
       'fiModelDescriptor', 'Enclosure', 'ProductGroup', 'Tire_Size_num',
       'Coupler', 'has_Hydraulics', 'Drive_System']

def prepare_test(df_test, to_keep, train_min_date):
    df = df_test.copy()
    df['Tire_Size_num'] = df['Tire_Size'].apply(extract_tire_size)
    size_order = ['Mini', 'Compact', 'Small', 'Medium', 'Large / Medium', 'Large']
    df['ProductSize'] = pd.Categorical(df['ProductSize'], categories=size_order, ordered=True)
    idx = df['YearMade'] > df['saledate'].dt.year
    df.loc[idx, 'YearMade'] = df.loc[idx, 'saledate'].dt.year
    df['saleElapsed'] = (df['saledate'] - train_min_date).dt.days
    df.drop('saledate', axis=1, inplace=True)
    df.loc[df['YearMade'] < 1900, 'YearMade'] = 1950

    for col in ['Enclosure', 'Hydraulics']:
        mode_val = train_df[col].mode()[0]
        df[f'has_{col}'] = (~df[col].isna()).astype(int)
        df.fillna({col: mode_val}, inplace=True)

    df_processed = df[to_keep].copy()
    obj_cols = df_processed.select_dtypes(include=['object']).columns.to_list()

    for col in obj_cols:
        df_processed[col] = pd.Categorical(df_processed[col])

    cat_cols = df_processed.select_dtypes(include=['category']).columns.to_list()
    for col in cat_cols:
        df_processed[col] = df_processed[col].cat.codes + 1

    return df_processed

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
        df['saledate'] = pd.to_datetime(df['saledate'], errors='coerce')
        sale_ids = df['SalesID'].tolist()

        min_date = train_df['saledate'].min()
        df = prepare_test(df, keeps, min_date)

        predictions = model.predict(df)
        predictions = np.exp(predictions)
        prediction_list = predictions.tolist()

        # sale_ids_draw = sale_ids[:20]
        # predictions_draw = predictions[:20]
        #
        # diffs = [predictions_draw[0]] + [predictions_draw[i] - predictions_draw[i - 1] for i in range(1, len(predictions_draw))]
        # cumulative = np.cumsum(diffs)
        # df = pd.DataFrame({
        #     'SaleID': [str(sid) for sid in sale_ids_draw],
        #     'Change': diffs,
        #     'Cumulative': cumulative
        # })
        # sns.set(style="whitegrid")
        # fig, ax = plt.subplots(figsize=(10, 6))
        # colors = ['green' if v >= 0 else 'red' for v in df['Change']]
        # bottom = 0
        # for i in range(len(df)):
        #     ax.bar(df['SaleID'][i], df['Change'][i], bottom=bottom, color=colors[i])
        #     height = bottom + df['Change'][i]
        #     ax.text(i, height, f"{height:,.0f}", ha='center', va='bottom', fontsize=9)
        #     bottom += df['Change'][i]
        # ax.axhline(y=bottom, color='blue', linestyle='--', linewidth=1)
        # ax.text(len(df) - 0.5, bottom, f"Tổng: {int(bottom):,}", color='blue', va='bottom', fontsize=10)
        # ax.set_title("Biểu đồ Thác nước: Giá dự đoán theo SaleID", fontsize=14)
        # ax.set_ylabel("Giá dự đoán (VNĐ)")
        # plt.xticks(rotation=45)
        # plt.tight_layout()
        # plot1_path = f"static/feature_importance_{uuid.uuid4().hex}.png"
        # plt.savefig(plot1_path, dpi=300)
        # plt.close()

        return templates.TemplateResponse("result.html", {
            "request": request,
            # "plot1": "/" + plot1_path,
            "predictions": zip(sale_ids, prediction_list)
        })
    except Exception as e:
        error_detail = traceback.format_exc()
        print(error_detail)
        return templates.TemplateResponse("index.html", {"request": request, "result": f"Error: {str(e)}"})
