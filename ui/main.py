import traceback

from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse # Added JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import pickle
import io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import uuid
import base64
from treeinterpreter import treeinterpreter
import waterfall_chart

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# global variables to store the latest predictions and data
latest_predictions_df = None
latest_df_prepared = None # To store the dataframe used for prediction
latest_df_original = None # To store the original uploaded dataframe

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
    global latest_predictions_df, latest_df_prepared, latest_df_original # Include new global vars

    if file.content_type != "text/csv":
        return templates.TemplateResponse("index.html", {"request": request, "result": "File must be CSV."})

    if model is None:
        return templates.TemplateResponse("index.html", {"request": request, "result": "Model not loaded. Please check server logs."})

    try:
        contents = await file.read()
        # read the CSV into df first
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        # keep a copy of the original df for later use
        latest_df_original = df.copy() # Store original df

        # ensure 'saledate' is parsed, handling potential errors by coercing to NaT
        df['saledate'] = pd.to_datetime(df['saledate'], errors='coerce')
        
        if df['saledate'].isnull().any():
            return templates.TemplateResponse("index.html", {"request": request, "result": "Error: CSV contains rows with invalid or missing 'saledate'."})

        sale_ids = df['SalesID'].tolist()
        first_sale_id = sale_ids[0] if sale_ids else "N/A"

        df_prepared = prepare_test(df.copy(), keeps, TRAINING_PARAMS)
        latest_df_prepared = df_prepared.copy() # Store prepared df

        predictions_array = model.predict(df_prepared)
        predictions_array = np.exp(predictions_array)
        prediction_list = predictions_array.tolist()

        # store predictions for CSV export
        latest_predictions_df = pd.DataFrame({
            'SaleID': sale_ids,
            'PredictedPrice': [round(p, 2) for p in prediction_list]
        })

        # generate Feature Importances plot
        plot1_base64 = None
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = df_prepared.columns
            feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(x=feature_importances, y=feature_importances.index)
            plt.title('Feature Importances')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.tight_layout()
            
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png')
            img_buffer.seek(0)
            plot1_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
            plt.close() # close the plot to free up memory

        # generate Waterfall chart for the first instance
        plot_waterfall_base64 = None
        if not df_prepared.empty:
            # Initial waterfall for the first row
            row_for_interpretation = df_prepared.iloc[[0]]
            if hasattr(model, 'estimators_') and hasattr(model, 'predict'):
                try:
                    prediction, bias, contributions = treeinterpreter.predict(model, row_for_interpretation.values)
                    waterfall_features = df_prepared.columns
                    plt.figure(figsize=(18, 10))
                    waterfall_chart.plot(waterfall_features, contributions[0], threshold=0.01,
                                         rotation_value=45, formatting='{:,.3f}', net_label="Net Prediction",
                                         Title=f"Prediction Breakdown for SaleID: {first_sale_id}")
                    plt.subplots_adjust(bottom=0.35)
                    plt.tight_layout()

                    img_buffer_waterfall = io.BytesIO()
                    plt.savefig(img_buffer_waterfall, format='png')
                    img_buffer_waterfall.seek(0)
                    plot_waterfall_base64 = base64.b64encode(img_buffer_waterfall.read()).decode('utf-8')
                    plt.close()
                except Exception as e_ti:
                    print(f"Error during initial tree interpretation: {e_ti}")
                    # ... error handling ...
            else:
                print("Initial: Model not compatible with treeinterpreter.")


        return templates.TemplateResponse("result.html", {
            "request": request,
            "predictions": zip(sale_ids, prediction_list), # Pass original sale_ids and predictions
            "plot1": plot1_base64,
            "plot_waterfall": plot_waterfall_base64,
            "first_sale_id": first_sale_id
        })
    except Exception as e:
        error_detail = traceback.format_exc()
        print(error_detail)
        return templates.TemplateResponse("index.html", {"request": request, "result": f"Error: {str(e)}"})

@app.get("/export_csv")
async def export_csv():
    global latest_predictions_df
    if latest_predictions_df is not None:
        output = io.StringIO()
        latest_predictions_df.to_csv(output, index=False)
        csv_data = output.getvalue()
        output.close()
        
        return StreamingResponse(
            iter([csv_data]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"}
        )
    else:
        return HTMLResponse("No data to export. Please upload a file and make predictions first.", status_code=404)

@app.get("/explain_prediction/{sale_id}", response_class=JSONResponse)
async def explain_prediction(sale_id: int, request: Request):
    global latest_df_prepared, latest_df_original

    if latest_df_prepared is None or latest_df_original is None or model is None:
        return JSONResponse(content={"error": "No data available or model not loaded. Please upload a file first."}, status_code=404)

    try:
        # find the index of the sale_id in the original dataframe
        original_row_index = latest_df_original[latest_df_original['SalesID'] == sale_id].index
        if original_row_index.empty:
            return JSONResponse(content={"error": f"SaleID {sale_id} not found in the uploaded data."}, status_code=404)
        
        row_index = original_row_index[0] # get the first matching index

        # ensure the index is within the bounds of latest_df_prepared
        if row_index >= len(latest_df_prepared):
            return JSONResponse(content={"error": f"Index for SaleID {sale_id} is out of bounds for prepared data."}, status_code=404)

        row_for_interpretation = latest_df_prepared.iloc[[row_index]]
        
        plot_waterfall_base64 = None
        actual_sale_id_for_plot = latest_df_original.iloc[row_index]['SalesID']


        if hasattr(model, 'estimators_') and hasattr(model, 'predict'):
            prediction, bias, contributions = treeinterpreter.predict(model, row_for_interpretation.values)
            waterfall_features = latest_df_prepared.columns
            
            plt.figure(figsize=(18, 10))
            waterfall_chart.plot(waterfall_features, contributions[0], threshold=0.01,
                                 rotation_value=45, formatting='{:,.3f}', net_label="Net Prediction",
                                 Title=f"Prediction Breakdown for SaleID: {actual_sale_id_for_plot}")
            plt.subplots_adjust(bottom=0.35)
            plt.tight_layout()
            
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png')
            img_buffer.seek(0)
            plot_waterfall_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
            plt.close()
            
            return JSONResponse(content={
                "plot_waterfall": plot_waterfall_base64,
                "sale_id": str(actual_sale_id_for_plot) # ensure sale_id is string for JSON
            })
        else:
            return JSONResponse(content={"error": "Model is not compatible with treeinterpreter."}, status_code=500)

    except Exception as e:
        error_detail = traceback.format_exc()
        print(f"Error in /explain_prediction/{sale_id}: {e}\\n{error_detail}")
        return JSONResponse(content={"error": f"An error occurred: {str(e)}"}, status_code=500)
