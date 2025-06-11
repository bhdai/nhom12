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

# load training data to extract parameters
train_df = pd.read_csv('../data/TrainAndValid.csv', parse_dates=['saledate'], low_memory=False)

TRAINING_PARAMS = {
    'train_min_date': train_df['saledate'].min(),
    'imputation_modes': {
        'Enclosure': train_df['Enclosure'].mode()[0],
        'Hydraulics': train_df['Hydraulics'].mode()[0]
    },
    'year_made_min_fill': 1950
}

def extract_tire_size(x):
    if pd.isna(x): return -2
    if x == 'None or Unspecified': return -1
    if 'inch' in x: return float(x.split(' ')[0])
    return float(x.replace('"', ''))

keeps = ['YearMade', 'ProductSize', 'Coupler_System', 'fiProductClassDesc',
       'fiSecondaryDesc', 'saleElapsed', 'fiModelDesc', 'ModelID',
       'fiModelDescriptor', 'Enclosure', 'ProductGroup', 'Tire_Size_num',
       'Coupler', 'has_Hydraulics', 'Drive_System'] # 'has_Enclosure' will be created

def prepare_test(df_test, to_keep_list, training_params_dict):
    df = df_test.copy()
    df['Tire_Size_num'] = df['Tire_Size'].apply(extract_tire_size)
    size_order = ['Mini', 'Compact', 'Small', 'Medium', 'Large / Medium', 'Large']
    df['ProductSize'] = pd.Categorical(df['ProductSize'], categories=size_order, ordered=True)
    idx = df['YearMade'] > df['saledate'].dt.year
    df.loc[idx, 'YearMade'] = df.loc[idx, 'saledate'].dt.year
    df['saleElapsed'] = (df['saledate'] - training_params_dict['train_min_date']).dt.days
    df.drop('saledate', axis=1, inplace=True)
    df.loc[df['YearMade'] < 1900, 'YearMade'] = training_params_dict['year_made_min_fill']

    for col in ['Enclosure', 'Hydraulics']:
        mode_val = training_params_dict['imputation_modes'][col]
        df[f'has_{col}'] = (~df[col].isna()).astype(int)
        df.fillna({col: mode_val}, inplace=True)


    # ensure all columns in to_keep_list are present, add them with NaNs if not.
    for col in to_keep_list:
        if col not in df.columns:
            df[col] = np.nan

    df_processed = df[to_keep_list].copy()
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
        # ensure 'saledate' is parsed, handling potential errors by coercing to NaT
        df['saledate'] = pd.to_datetime(df['saledate'], errors='coerce')
        
        # check for NaT in 'saledate' which are critical for 'saleElapsed'
        if df['saledate'].isnull().any():
            # handle rows with invalid sale dates, by returning an error
            return templates.TemplateResponse("index.html", {"request": request, "result": "Error: CSV contains rows with invalid or missing 'saledate'."})

        sale_ids = df['SalesID'].tolist()

        df_prepared = prepare_test(df, keeps, TRAINING_PARAMS)
        
        predictions = model.predict(df_prepared)
        predictions = np.exp(predictions)
        prediction_list = predictions.tolist()

        return templates.TemplateResponse("result.html", {
            "request": request,
            "predictions": zip(sale_ids, prediction_list)
        })
    except Exception as e:
        error_detail = traceback.format_exc()
        print(error_detail)
        return templates.TemplateResponse("index.html", {"request": request, "result": f"Error: {str(e)}"})
