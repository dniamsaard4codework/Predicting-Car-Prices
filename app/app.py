
# Import required libraries
import joblib
import numpy as np
import pandas as pd
import os
import dash
from dash import html, dcc, callback
from dash.dependencies import Input, Output, State

# Load the trained model from multiple possible paths
MODEL_PATHS = [
    "./car_price.model",  # For Docker deployment
    "./model/car_price.model",  # For root directory local development
    "../model/car_price.model",  # For app directory local development
]

model = None
for MODEL_PATH in MODEL_PATHS:
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            print(f"Model loaded successfully from {MODEL_PATH}")
            break
        except Exception as e:
            print(f"Failed to load model from {MODEL_PATH}. Error: {e}")
            continue

if model is None:
    raise RuntimeError("No valid model found in any of the expected paths")

# Initialize Dash application
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server  # for gunicorn if needed later

# Add CSS styling for better UI appearance
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Georgia:wght@400;700&display=swap');
            body {
                font-family: 'Georgia', serif !important;
                margin: 0;
                background-color: #fafafa;
            }
            .nav-bar {
                background-color: #2c3e50;
                padding: 15px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .nav-container {
                max-width: 1200px;
                margin: 0 auto;
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 0 20px;
            }
            .nav-logo {
                color: white;
                font-size: 24px;
                font-weight: bold;
                text-decoration: none;
            }
            .nav-links {
                display: flex;
                gap: 30px;
            }
            .nav-link {
                color: white;
                text-decoration: none;
                padding: 10px 20px;
                border-radius: 5px;
                transition: background-color 0.3s;
            }
            .nav-link:hover {
                background-color: #34495e;
                text-decoration: none;
                color: white;
            }
            .nav-link.active {
                background-color: #3498db;
            }
            .container {
                max-width: 1000px;
                margin: 0 auto;
                padding: 30px 20px;
            }
            .instruction-card {
                background: white;
                border-radius: 10px;
                padding: 30px;
                margin-bottom: 30px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .form-container {
                background: white;
                border-radius: 10px;
                padding: 30px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Create navigation bar component
def create_navbar(current_page):
    return html.Div([
        html.Div([
            html.A("Car Price Predictor", href="/", className="nav-logo"),
            html.Div([
                html.A("Instructions", 
                      href="/", 
                      className=f"nav-link {'active' if current_page == 'home' else ''}"),
                html.A("Predict Price", 
                      href="/predict", 
                      className=f"nav-link {'active' if current_page == 'predict' else ''}"),
            ], className="nav-links")
        ], className="nav-container")
    ], className="nav-bar")

# Create instructions page layout
def instructions_layout():
    return html.Div([
        create_navbar('home'),
        html.Div([
            html.Div([
                html.H1("Welcome to Car Price Prediction System", 
                       style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '30px'}),
                html.H2("by Dechathon Niamsa-ard [st126235]",
                        style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': '30px'}),
                html.H3("How the Prediction System Works", style={'color': '#34495e', 'marginBottom': '20px'}),
                html.P([
                    "Car price prediction system uses advanced machine learning (XGBoost) trained on car market data. "
                    "The model analyzes multiple factors including car specifications, condition, and market trends to provide "
                    "accurate price estimates for used cars."
                ], style={'lineHeight': '1.6', 'fontSize': '16px', 'color': '#555'}),
                
                html.H3("Step-by-Step Instructions", style={'color': '#34495e', 'marginTop': '30px', 'marginBottom': '20px'}),
                html.Ol([
                    html.Li("Navigate to the 'Predict Price' page using the navigation bar above"),
                    html.Li("Fill in the car details you know in the input form"),
                    html.Li("Don't worry if you don't have all information - you can skip any field"),
                    html.Li("For missing fields, system uses smart imputation techniques to fill reasonable defaults"),
                    html.Li("Click the 'Predict Price' button to submit your data"),
                    html.Li("The predicted price will appear below the form within moments")
                ], style={'lineHeight': '1.8', 'fontSize': '16px', 'color': '#555'}),
                
                html.H3("Missing Data Handling", style={'color': '#34495e', 'marginTop': '30px', 'marginBottom': '20px'}),
                html.P([
                    "System intelligently handles missing information using the trained pipeline's imputation strategies learned from market data:"
                ], style={'lineHeight': '1.6', 'fontSize': '16px', 'color': '#555'}),
                html.Ul([
                    html.Li("Numerical fields (Year, Kilometers, Owner, Engine, Power): Uses median from training data"),
                    html.Li("Mileage: Uses mean from training data"),
                    html.Li("Categorical fields (Fuel Type, Transmission, Brand): Uses most frequent from training data"),
                    html.Li("All imputation values are learned from the actual car market data during model training")
                ], style={'lineHeight': '1.8', 'fontSize': '16px', 'color': '#555'}),
                
                html.H3("Important Notes", style={'color': '#34495e', 'marginTop': '30px', 'marginBottom': '20px'}),
                html.Div([
                    html.P("• Model is trained specifically on Petrol and Diesel vehicles", 
                           style={'color': '#e74c3c', 'fontWeight': 'bold', 'marginBottom': '10px'}),
                    html.P("• Predictions are estimates based on historical market data and current trends", 
                           style={'color': '#555', 'marginBottom': '10px'}),
                    html.P("• The more accurate information you provide, the better the prediction", 
                           style={'color': '#555'})
                ]),
                
                html.Div([
                    html.A("Start Predicting Prices →", 
                           href="/predict",
                           style={
                               'display': 'inline-block',
                               'padding': '15px 30px',
                               'backgroundColor': '#3498db',
                               'color': 'white',
                               'textDecoration': 'none',
                               'borderRadius': '5px',
                               'fontSize': '18px',
                               'textAlign': 'center',
                               'marginTop': '30px'
                           })
                ], style={'textAlign': 'center'})
                
            ], className="instruction-card")
        ], className="container")
    ])

# Create helper functions for form elements
def labeled_input(label, id_, type_="number", placeholder="", **kwargs):
    return html.Div([
        html.Label(label, style={"marginBottom": "8px", "display": "block", "color": "#2c3e50", "fontWeight": "bold"}),
        dcc.Input(
            id=id_, 
            type=type_, 
            placeholder=placeholder, 
            style={
                "width": "100%", 
                "padding": "12px", 
                "border": "2px solid #bdc3c7",
                "borderRadius": "5px",
                "fontSize": "16px",
                "fontFamily": "Georgia, serif"
            },
            **kwargs
        )
    ], style={"marginBottom": "20px"})

def labeled_dropdown(label, id_, options, value=None):
    return html.Div([
        html.Label(label, style={"marginBottom": "8px", "display": "block", "color": "#2c3e50", "fontWeight": "bold"}),
        dcc.Dropdown(
            id=id_, 
            options=[{"label": o, "value": o} for o in options], 
            value=value, 
            clearable=True,
            placeholder="Select or leave blank for pipeline imputation",
            style={
                "fontSize": "16px",
                "fontFamily": "Georgia, serif"
            }
        )
    ], style={"marginBottom": "20px"})

# Create prediction page layout
def prediction_layout():
    return html.Div([
        create_navbar('predict'),
        html.Div([
            html.Div([
                html.H1("Car Price Prediction", 
                       style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
                html.P("Fill in the details you know. Leave fields blank if you don't have the information.", 
                      style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': '30px', 'fontSize': '16px'}),
                
                html.Div([
                    html.Div([
                        labeled_input("Year of Manufacture", "year", placeholder="e.g., 2016 (leave blank for median from data)", min=1980, max=2030, step=1),
                        labeled_input("Kilometers Driven", "km", placeholder="e.g., 55000 (leave blank for median from data)", min=0, step=100),
                        labeled_dropdown("Fuel Type", "fuel", ["Petrol", "Diesel"]),
                        labeled_dropdown("Transmission", "transmission", ["Manual", "Automatic"]),
                        labeled_dropdown("Number of Previous Owners", "owner", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"]),
                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                    
                    html.Div([
                        labeled_input("Mileage (kmpl)", "mileage", placeholder="e.g., 18.5 (leave blank for mean from data)", min=0, step=0.1),
                        labeled_input("Engine Displacement (CC)", "engine", placeholder="e.g., 1197 (leave blank for median from data)", min=200, step=1),
                        labeled_input("Max Power (bhp)", "power", placeholder="e.g., 82 (leave blank for median from data)", min=10, step=1),
                        labeled_dropdown("Brand", "brand", [
                            "Maruti","Hyundai","Honda","Toyota","Skoda","BMW","Audi","Mercedes-Benz","Ford",
                            "Volkswagen","Mahindra","Tata","Renault","Chevrolet","Nissan","Kia","Jeep",
                            "Land Rover","Ashok Leyland","Datsun","Fiat","Jaguar","Mini","Mitsubishi","Porsche","Volvo","Others"
                        ]),
                        html.Div(style={'marginBottom': '20px'})  # Spacing
                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '4%'})
                ]),
                
                html.Button("Predict Price", 
                           id="predict", 
                           n_clicks=0, 
                           style={
                               "width": "100%",
                               "padding": "15px 20px",
                               "fontSize": "18px",
                               "color": "white",
                               "backgroundColor": "#4a9d5b",
                               "border": "none",
                               "borderRadius": "5px",
                               "cursor": "pointer",
                               "marginTop": "20px",
                               "fontFamily": "Georgia, serif"
                           }),
                
                html.Div(id="result-section", style={'marginTop': '30px'})
                
            ], className="form-container")
        ], className="container")
    ])

# Set up main app layout with URL routing
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# Handle URL routing to display correct page
@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/predict':
        return prediction_layout()
    else:  # Default to instructions page
        return instructions_layout()

# Handle price prediction when button is clicked
@app.callback(
    Output("result-section", "children"),
    Input("predict", "n_clicks"),
    State("year", "value"),
    State("km", "value"),
    State("fuel", "value"),
    State("transmission", "value"),
    State("owner", "value"),
    State("mileage", "value"),
    State("engine", "value"),
    State("power", "value"),
    State("brand", "value"),
)
def predict_price(n_clicks, year, km, fuel, transmission, owner, mileage, engine, power, brand):
    if not n_clicks:
        return html.Div([
            html.P("Fill in the form above and click 'Predict Price' to get your estimate.", 
                  style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '16px'})
        ])
    
    # Map owner text to numeric values for model input
    owner_mapping = {
        'First Owner': 1,
        'Second Owner': 2,
        'Third Owner': 3,
        'Fourth & Above Owner': 4
    }
    
    # Convert owner text to number if provided
    owner_num = owner_mapping.get(owner) if owner is not None else None

    # Map Brand for Land Rover and Ashok Leyland, otherwise keep original brand
    brand_mapping = {
        'Land Rover': 'Land',
        'Ashok Leyland': 'Ashok'
    }
    if brand is not None:
        if brand in brand_mapping:
            brand = brand_mapping.get(brand, brand)
    else:
        brand = None

    # Prepare input data for model prediction
    # Leave missing values as NaN/None - the pipeline will handle imputation
    row = pd.DataFrame([{
        "year": float(year) if year is not None else np.nan,
        "km_driven": float(km) if km is not None else np.nan,
        "fuel": str(fuel) if fuel is not None else None,
        "transmission": str(transmission) if transmission is not None else None,
        "owner": float(owner_num) if owner_num is not None else np.nan,  # Model expects numeric owner
        "engine": float(engine) if engine is not None else np.nan,  # Model expects numeric engine
        "max_power": float(power) if power is not None else np.nan,  # Model expects numeric max_power
        "brand": str(brand) if brand is not None else None,
        "mileage": float(mileage) if mileage is not None else np.nan,  # Model expects numeric mileage
    }])

    # Track which fields are missing for user feedback
    imputed_fields = []
    for col in row.columns:
        if pd.isna(row.at[0, col]) or row.at[0, col] is None:
            imputed_fields.append(col)

    # Make prediction and convert from log scale to price
    try:
        # Debug information for troubleshooting
        print("Input row shape:", row.shape)
        print("Input row columns:", list(row.columns))
        print("Input row values:", row.iloc[0].to_dict())
        print("Imputed fields:", imputed_fields)
        
        pred_log = float(model.predict(row)[0])
        price = float(np.exp(pred_log))
        
        print(f"Predicted log price: {pred_log:.4f}")
        print(f"Predicted price: {price:.2f}")
        
        # Display prediction results with styling
        result_content = [
            html.H2(f"Estimated Price: {price:,.0f}", 
                   style={'textAlign': 'center', 'color': '#27ae60', 'fontSize': '32px', 'marginBottom': '20px'}),
        ]
        
        # Show imputation information if fields were auto-filled
        if imputed_fields:
            imputation_mapping = {
                "year": "Year → Median from training data",
                "km_driven": "Kilometers → Median from training data", 
                "owner": "Owner → Median from training data",
                "mileage": "Mileage → Mean from training data",
                "engine": "Engine → Median from training data",
                "max_power": "Max Power → Median from training data",
                "fuel": "Fuel Type → Most frequent from training data",
                "transmission": "Transmission → Most frequent from training data",
                "brand": "Brand → Most frequent from training data"
            }
            
            result_content.append(
                html.Div([
                    html.H4("Note: Missing Information Handled by Pipeline", 
                           style={'color': '#f39c12', 'marginBottom': '15px'}),
                    html.P("The following fields were automatically filled using the trained pipeline's imputation strategy:", 
                          style={'color': '#7f8c8d', 'marginBottom': '10px'}),
                    html.Ul([
                        html.Li(imputation_mapping.get(field, f"{field} → pipeline default")) 
                        for field in imputed_fields
                    ], style={'color': '#7f8c8d', 'lineHeight': '1.5'})
                ], style={
                    'backgroundColor': '#fef9e7', 
                    'padding': '20px', 
                    'borderRadius': '5px',
                    'border': '1px solid #f39c12',
                    'marginTop': '20px'
                })
            )
        
        result_content.extend([
            html.Hr(style={'margin': '30px 0'}),
            html.Div([
                html.P("Model trained on Petrol & Diesel vehicles only.", 
                      style={'color': '#e74c3c', 'textAlign': 'center', 'fontSize': '14px', 'fontWeight': 'bold'})
            ])
        ])
        
        return html.Div(result_content, style={
            'backgroundColor': 'white', 
            'padding': '30px', 
            'borderRadius': '10px',
            'boxShadow': '0 2px 10px rgba(0,0,0,0.1)',
            'border': '3px solid #27ae60'
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        print(f"Model type: {type(model)}")
        return html.Div([
            html.H3("Prediction Failed", style={'color': '#e74c3c', 'textAlign': 'center'}),
            html.P(f"Error: {str(e)}", style={'color': '#7f8c8d', 'textAlign': 'center'}),
            html.P("Please check your input data and try again.", style={'color': '#7f8c8d', 'textAlign': 'center'})
        ], style={
            'backgroundColor': '#fdf2f2', 
            'padding': '20px', 
            'borderRadius': '5px',
            'border': '2px solid #e74c3c',
            'marginTop': '20px'
        })

# Start the application
if __name__ == "__main__":
    # Get configuration from environment variables
    port = int(os.environ.get("PORT", 8050))
    debug = os.environ.get("DEBUG", "True").lower() == "true"
    
    print(f"Starting Car Price Prediction App on port {port}")
    print(f"Debug mode: {'ON' if debug else 'OFF'}")
    
    app.run(host="0.0.0.0", port=port, debug=debug)
