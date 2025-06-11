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
# Add a cache for file content to avoid re-reading if possible, or to pass between stages
# For simplicity, we'll use latest_df_original for now, assuming single user or dev context.

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
    # Clear previous data when returning to the main page
    global latest_df_original, latest_df_prepared, latest_predictions_df
    latest_df_original = None
    latest_df_prepared = None
    latest_predictions_df = None
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/") # This will now handle both initial upload for preview and triggering prediction
async def handle_upload_or_predict(request: Request, file: UploadFile = File(None), action: str = Form(None)):
    global latest_predictions_df, latest_df_prepared, latest_df_original

    if action == "predict":
        if latest_df_original is None:
            return templates.TemplateResponse("index.html", {"request": request, "result": "No file data found to predict. Please upload a file first."})
        
        df_for_prediction = latest_df_original.copy()
        
        try:
            # --- This is the prediction logic, moved from the original handle_upload ---
            df_for_prediction['saledate'] = pd.to_datetime(df_for_prediction['saledate'], errors='coerce')
            if df_for_prediction['saledate'].isnull().any():
                return templates.TemplateResponse("index.html", {"request": request, "result": "Error: CSV contains rows with invalid or missing 'saledate'.", "filename": getattr(latest_df_original, 'filename', 'Uploaded File')})

            sale_ids = df_for_prediction['SalesID'].tolist()
            first_sale_id = sale_ids[0] if sale_ids else "N/A"

            df_prepared = prepare_test(df_for_prediction.copy(), keeps, TRAINING_PARAMS)
            latest_df_prepared = df_prepared.copy()

            predictions_array = model.predict(df_prepared)
            predictions_array = np.exp(predictions_array)
            prediction_list = predictions_array.tolist()

            latest_predictions_df = pd.DataFrame({
                'SaleID': sale_ids,
                'PredictedPrice': [round(p, 2) for p in prediction_list]
            })

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
                plt.close()

            plot_waterfall_base64 = None
            if not df_prepared.empty:
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
            
            return templates.TemplateResponse("result.html", {
                "request": request,
                "predictions": zip(sale_ids, prediction_list),
                "plot1": plot1_base64,
                "plot_waterfall": plot_waterfall_base64,
                "first_sale_id": first_sale_id
            })
            # --- End of prediction logic ---
        except Exception as e:
            error_detail = traceback.format_exc()
            print(error_detail)
            return templates.TemplateResponse("index.html", {"request": request, "result": f"Error during prediction: {str(e)}", "filename": getattr(latest_df_original, 'filename', 'Uploaded File'), "show_predict_button": True})

    elif file: # This is the initial file upload for preview
        if file.content_type != "text/csv":
            return templates.TemplateResponse("index.html", {"request": request, "result": "File must be CSV."})
        if model is None: # Check model availability early
             return templates.TemplateResponse("index.html", {"request": request, "result": "Model not loaded. Please check server logs."})

        try:
            contents = await file.read()
            df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
            latest_df_original = df.copy() # Store the original dataframe
            setattr(latest_df_original, 'filename', file.filename) # Store filename

            # Basic preview - First few rows
            df_preview_html = latest_df_original.head().to_html(classes=["table", "table-sm", "table-bordered", "table-striped"], justify="left")
            num_rows, num_cols = latest_df_original.shape
            
            # Generate statistical summary for numeric columns
            stats_html = latest_df_original.describe().to_html(
                classes=["table", "table-sm", "table-bordered", "table-striped"],
                justify="left",
                float_format=lambda x: f"{x:.2f}" # Format floating point numbers
            )
            
            # Generate missing value analysis
            missing_data = pd.DataFrame({
                'Column': latest_df_original.columns,
                'Data Type': latest_df_original.dtypes,
                'Missing Values': latest_df_original.isna().sum(),
                '% Missing': 100 * latest_df_original.isna().sum() / len(latest_df_original)
            }).sort_values('% Missing', ascending=False)
            
            # Format the missing values percentage with 2 decimal places
            missing_data['% Missing'] = missing_data['% Missing'].apply(lambda x: f"{x:.2f}%")
            
            missing_html = missing_data.to_html(
                classes=["table", "table-sm", "table-bordered", "table-striped"],
                justify="left",
                index=False
            )
            
            # Generate data type distribution summary
            dtype_counts = latest_df_original.dtypes.value_counts().reset_index()
            dtype_counts.columns = ['Data Type', 'Count']
            
            dtype_html = dtype_counts.to_html(
                classes=["table", "table-sm", "table-bordered", "table-striped"],
                justify="left",
                index=False
            )
            
            # Generate distribution plot for YearMade (a key feature in the bulldozer dataset)
            dist_plot_base64 = None
            
            if 'YearMade' in latest_df_original.columns:
                try:
                    plt.figure(figsize=(12, 6))
                    
                    # Filter out potentially erroneous years (e.g., 1000, 1900, etc.)
                    # but still show all relevant manufacturing years
                    year_data = latest_df_original['YearMade'].copy()
                    year_data = year_data[(year_data > 1900) & (year_data <= 2025)]
                    
                    # Create histogram with kde
                    ax = sns.histplot(year_data, kde=True, bins=30)
                    
                    # Add vertical lines for key time periods in construction equipment history
                    key_years = [1950, 1970, 1990, 2000, 2010]
                    for year in key_years:
                        if year >= year_data.min() and year <= year_data.max():
                            plt.axvline(x=year, color='red', linestyle='--', alpha=0.7)
                    
                    plt.title('Distribution of YearMade (Manufacturing Year)')
                    plt.xlabel('Year')
                    plt.ylabel('Count')
                    
                    # Add summary statistics as text
                    stats_text = (f"Mean: {year_data.mean():.1f}\n"
                                 f"Median: {year_data.median()}\n"
                                 f"Min: {year_data.min()}\n"
                                 f"Max: {year_data.max()}")
                    
                    plt.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                            verticalalignment='top', bbox={'boxstyle': 'round', 'alpha': 0.5})
                    
                    plt.tight_layout()
                    
                    img_buffer = io.BytesIO()
                    plt.savefig(img_buffer, format='png')
                    img_buffer.seek(0)
                    dist_plot_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
                    plt.close()
                    
                    # Store the string name for the UI
                    plot_col = 'YearMade'
                    
                except Exception as plot_error:
                    print(f"Error generating YearMade distribution plot: {plot_error}")
            
            # Also check for other important features if YearMade is not available
            elif len(latest_df_original.select_dtypes(include=['number']).columns) > 0:
                try:
                    # Fall back to another numeric column if YearMade isn't available
                    numeric_cols = latest_df_original.select_dtypes(include=['number']).columns
                    # Prefer important bulldozer features if available
                    preferred_cols = ['ProductSize', 'ModelID', 'age', 'MachineHoursCurrentMeter']
                    plot_col = next((col for col in preferred_cols if col in numeric_cols), 
                                   next((col for col in numeric_cols if 'id' not in col.lower()), numeric_cols[0]))
                    
                    plt.figure(figsize=(10, 6))
                    sns.histplot(latest_df_original[plot_col].dropna(), kde=True)
                    plt.title(f'Distribution of {plot_col}')
                    plt.tight_layout()
                    
                    img_buffer = io.BytesIO()
                    plt.savefig(img_buffer, format='png')
                    img_buffer.seek(0)
                    dist_plot_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
                    plt.close()
                except Exception as plot_error:
                    print(f"Error generating distribution plot: {plot_error}")
            
            # Generate sale date analysis plots
            date_plot_base64 = None
            if 'saledate' in df.columns:
                try:
                    # Ensure saledate is in datetime format
                    df['saledate'] = pd.to_datetime(df['saledate'], errors='coerce')
                    
                    # Create a figure with two subplots - sales count by year and month
                    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
                    
                    # Sales count by year
                    yearly_sales = df.groupby(df['saledate'].dt.year).size()
                    yearly_sales.plot(kind='bar', ax=axes[0])
                    axes[0].set_title('Number of Sales by Year')
                    axes[0].set_xlabel('Year')
                    axes[0].set_ylabel('Count')
                    
                    # Sales count by month
                    monthly_sales = df.groupby(df['saledate'].dt.month).size()
                    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    monthly_sales.index = [month_names[i-1] for i in monthly_sales.index]
                    monthly_sales.plot(kind='bar', ax=axes[1])
                    axes[1].set_title('Number of Sales by Month')
                    axes[1].set_xlabel('Month')
                    axes[1].set_ylabel('Count')
                    
                    plt.tight_layout()
                    
                    date_buffer = io.BytesIO()
                    plt.savefig(date_buffer, format='png')
                    date_buffer.seek(0)
                    date_plot_base64 = base64.b64encode(date_buffer.read()).decode('utf-8')
                    plt.close()
                    
                    # If SalePrice or a similar column exists, show price trends over time
                    price_trend_base64 = None
                    price_col = next((col for col in df.columns if 'price' in col.lower() or 'sale' in col.lower() and col.lower() != 'saledate'), None)
                    
                    if price_col and df[price_col].dtype in [np.float64, np.int64]:
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        # Create a new column for year-month
                        df['year_month'] = df['saledate'].dt.to_period('M')
                        
                        # Group by year_month and calculate median price
                        price_over_time = df.groupby('year_month')[price_col].median()
                        
                        # Plot median price over time
                        price_over_time.plot(ax=ax)
                        ax.set_title(f'Median {price_col} Over Time')
                        ax.set_xlabel('Date')
                        ax.set_ylabel(price_col)
                        
                        plt.tight_layout()
                        
                        price_buffer = io.BytesIO()
                        plt.savefig(price_buffer, format='png')
                        price_buffer.seek(0)
                        price_trend_base64 = base64.b64encode(price_buffer.read()).decode('utf-8')
                        plt.close()
                        
                except Exception as date_plot_error:
                    print(f"Error generating date analysis plots: {date_plot_error}")
                    traceback.print_exc()
            
            return templates.TemplateResponse("index.html", {
                "request": request,
                "filename": file.filename,
                "data_preview_html": df_preview_html,
                "stats_html": stats_html,
                "missing_html": missing_html,
                "dtype_html": dtype_html,
                "dist_plot": dist_plot_base64,
                "date_plot": date_plot_base64,
                "price_trend_plot": price_trend_base64 if 'price_trend_base64' in locals() else None,
                "plot_col": plot_col if 'plot_col' in locals() else None,
                "num_rows": num_rows,
                "num_cols": num_cols,
                "show_predict_button": True # Flag to show "Proceed to Predict" button
            })
        except Exception as e:
            error_detail = traceback.format_exc()
            print(error_detail)
            return templates.TemplateResponse("index.html", {"request": request, "result": f"Error processing file for preview: {str(e)}"})
    
    # Default case if no action and no file (e.g., direct post to / without payload)
    return templates.TemplateResponse("index.html", {"request": request, "result": "Please upload a file."})


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
