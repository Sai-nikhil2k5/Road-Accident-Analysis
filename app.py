import pandas as pd
import numpy as np
import dash
from dash import html, dcc, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import joblib
import logging
import sys

if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

logger = logging.getLogger(__name__)
logging.getLogger('werkzeug').setLevel(logging.WARNING)
logging.getLogger('dash').setLevel(logging.WARNING)

class DataLoader:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataLoader, cls).__new__(cls)
            cls._instance._load_data()
        return cls._instance
    def _load_data(self):
        try:
            logger.info("Starting data loading process...")
            self.df = pd.read_csv("reg.csv")
            self.df['Count'] = pd.to_numeric(self.df['Count'], errors='coerce')
            self.df = self.df.dropna(subset=['Count'])
            categorical_cols = [
                'Million Plus Cities',
                'Cause category',
                'Cause Subcategory',
                'Outcome of Incident'
            ]
            for col in categorical_cols:
                self.df[col] = self.df[col].astype('category')
            if (self.df['Count'] % 1 == 0).all():
                self.df['Count'] = self.df['Count'].astype('int32')
            else:
                self.df['Count'] = self.df['Count'].astype('float32')
            self.df['City Avg Count'] = self.df.groupby(
                'Million Plus Cities', observed=True
            )['Count'].transform('mean')
            self.df['Cause Avg Count'] = self.df.groupby(
                'Cause Subcategory', observed=True
            )['Count'].transform('mean')
            logger.info(f"Data successfully loaded. Final shape: {self.df.shape}")
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            raise

class ModelLoader:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance._load_models()
        return cls._instance
    def _load_models(self):
        try:
            logger.info("Starting model loading process...")
            self.ct = joblib.load('column_transformer.joblib')
            self.rf_model = joblib.load('random_forest_model.joblib')
            self.xgb_model = joblib.load('xgboost_model.joblib')
            logger.info("All models loaded successfully")
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise

data_loader = DataLoader()
model_loader = ModelLoader()

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Road Accident Analysis "

def create_kpi_card(title, value):
    return html.Div([
        html.H4(title, className='kpi-title'),
        html.P(value, className='kpi-value')
    ], className='kpi-card')

def create_dropdown(id, label, options, value=None):
    return html.Div([
        html.Label(label, className='dropdown-label'),
        dcc.Dropdown(
            id=id,
            options=[{'label': str(opt), 'value': opt} for opt in sorted(options)],
            value=value,
            className='dropdown-select'
        )
    ], className='dropdown-group')

app.layout = html.Div([
    # Storage for session data
    dcc.Store(id='app-state'),
    
    # Header section
    html.Header([
        html.H1("Road Accident Analysis", className='app-title'),
    ], className='app-header'),
    
    # Main navigation tabs (removed National Overview)
    dcc.Tabs(id='main-tabs', value='analysis', children=[
        dcc.Tab(label='City Analysis', value='analysis'),
        dcc.Tab(label='Prediction Tool', value='prediction')
    ], className='main-tabs'),
    
    # Tab content container
    html.Div(id='tab-content', className='tab-content')
], className='app-container')

@app.callback(
    Output('tab-content', 'children'),
    Input('main-tabs', 'value')
)
def render_tab_content(active_tab):
    """Dynamically render content based on active tab"""
    df = data_loader.df
    
    if active_tab == 'analysis':
        return html.Div([
            create_dropdown(
                id='city-selector',
                label="Select City:",
                options=df['Million Plus Cities'].unique(),
                value='Delhi'
            ),
            
            html.Div(id='city-kpis', className='kpi-container'),
            
            dcc.Graph(id='cause-category-chart', className='graph-container'),
            dcc.Graph(id='subcause-chart', className='graph-container'),
            dcc.Graph(id='outcome-chart', className='graph-container'),
            dcc.Graph(id='sunburst-chart', className='graph-container')
        ])
    
    elif active_tab == 'prediction':
        return html.Div([
            html.H3("Accident Scenario Predictor", className='predictor-title'),
            
            create_dropdown(
                id='predict-city',
                label="Select City:",
                options=df['Million Plus Cities'].unique()
            ),
            
            create_dropdown(
                id='predict-category',
                label="Cause Category:",
                options=df['Cause category'].unique()
            ),
            
            create_dropdown(
                id='predict-subcategory',
                label="Cause Subcategory:",
                options=df['Cause Subcategory'].unique()
            ),
            
            create_dropdown(
                id='predict-outcome',
                label="Expected Outcome:",
                options=df['Outcome of Incident'].unique()
            ),
            
            html.Button(
                "Predict Accident Count", 
                id='predict-button', 
                className='predict-button'
            ),
            
            html.Div(id='prediction-results', className='prediction-results'),
            html.Div(id='confidence-indicator', className='confidence-indicator')
        ], className='prediction-container')

@app.callback(
    Output('city-kpis', 'children'),
    Output('cause-category-chart', 'figure'),
    Output('subcause-chart', 'figure'),
    Output('outcome-chart', 'figure'),
    Output('sunburst-chart', 'figure'),
    Input('city-selector', 'value')
)
def update_city_analysis(selected_city):
    try:
        df = data_loader.df
        city_data = df[df['Million Plus Cities'] == selected_city]
        total_accidents = city_data['Count'].sum()
        avg_per_day = city_data['Count'].mean()
        deadliest_cause = city_data.groupby('Cause category', observed=True)['Count'].sum().idxmax()
        kpis = [
            create_kpi_card("Total Accidents", f"{total_accidents:,}"),
            create_kpi_card("Daily Average", f"{avg_per_day:.1f}"),
            create_kpi_card("Deadliest Cause", deadliest_cause)
        ]
        cause_fig = px.bar(
            city_data.groupby('Cause category', observed=True)['Count'].sum().reset_index(),
            x='Cause category', y='Count',
            title=f'Accidents by Cause Category in {selected_city}',
            color='Cause category'
        )
        subcause_fig = px.bar(
            city_data.groupby('Cause Subcategory', observed=True)['Count'].sum().nlargest(10).reset_index(),
            y='Cause Subcategory', x='Count',
            orientation='h',
            title=f'Top 10 Dangerous Subcauses in {selected_city}',
            color='Count',
            color_continuous_scale='reds'
        )
        outcome_fig = px.pie(
            city_data,
            names='Outcome of Incident', values='Count',
            title=f'Outcome Distribution in {selected_city}',
            hole=0.4
        )
        sunburst_fig = px.sunburst(
            city_data,
            path=['Cause category', 'Cause Subcategory', 'Outcome of Incident'],
            values='Count',
            title=f'Accident Hierarchy in {selected_city}',
            color='Count',
            color_continuous_scale='thermal',
            height=800
        )
        sunburst_fig.update_layout(
            margin=dict(t=50, b=20, l=20, r=20),
            uniformtext=dict(minsize=12, mode='hide')
        )
        return kpis, cause_fig, subcause_fig, outcome_fig, sunburst_fig
    except Exception as e:
        logger.error(f"Error in city analysis: {str(e)}")
        return [], go.Figure(), go.Figure(), go.Figure(), go.Figure()

@app.callback(
    Output('top-cities-chart', 'figure'),
    Output('worst-causes-chart', 'figure'),
    Output('national-outcome-chart', 'figure'),
    Input('city-selector', 'value')
)
def update_national_analysis(_):
    try:
        df = data_loader.df
        top_cities = df.groupby('Million Plus Cities', observed=True)['Count'].sum().nlargest(10).reset_index()
        top_cities_fig = px.bar(
            top_cities,
            x='Million Plus Cities', y='Count',
            title='Cities with Highest Accident Counts',
            color='Count',
            color_continuous_scale='blues'
        )
        worst_causes = df.groupby(['Cause category', 'Cause Subcategory'], observed=True)['Count'].sum().nlargest(10).reset_index()
        worst_causes_fig = px.bar(
            worst_causes,
            x='Count', y='Cause Subcategory',
            orientation='h',
            title='Most Dangerous Accident Causes',
            color='Cause category'
        )
        outcome_fig = px.pie(
            df,
            names='Outcome of Incident', values='Count',
            title='National Accident Outcome Distribution',
            hole=0.3
        )
        return top_cities_fig, worst_causes_fig, outcome_fig
    except Exception as e:
        logger.error(f"Error in national analysis: {str(e)}")
        return go.Figure(), go.Figure(), go.Figure()

@app.callback(
    Output('prediction-results', 'children'),
    Output('confidence-indicator', 'children'),
    Input('predict-button', 'n_clicks'),
    State('predict-city', 'value'),
    State('predict-category', 'value'),
    State('predict-subcategory', 'value'),
    State('predict-outcome', 'value'),
    prevent_initial_call=True
)
def predict_accident_count(n_clicks, city, category, subcategory, outcome):
    if not all([city, category, subcategory, outcome]):
        return "Please select all parameters", ""
    try:
        df = data_loader.df
        ct = model_loader.ct
        rf_model = model_loader.rf_model
        xgb_model = model_loader.xgb_model
        input_df = pd.DataFrame([[city, category, subcategory, outcome]],
                              columns=['Million Plus Cities', 'Cause category',
                                      'Cause Subcategory', 'Outcome of Incident'])
        city_avg = df[df['Million Plus Cities'] == city]['Count'].mean()
        cause_avg = df[df['Cause Subcategory'] == subcategory]['Count'].mean()
        input_df['City Avg Count'] = city_avg if not pd.isna(city_avg) else 0
        input_df['Cause Avg Count'] = cause_avg if not pd.isna(cause_avg) else 0
        input_encoded = ct.transform(input_df)
        rf_pred = np.expm1(rf_model.predict(input_encoded))[0]
        xgb_pred = np.expm1(xgb_model.predict(input_encoded))[0]
        avg_pred = (rf_pred + xgb_pred) / 2
        std_dev = np.std([rf_pred, xgb_pred])
        confidence = max(0, min(100, 100 - (std_dev / avg_pred * 50 if avg_pred > 0 else 0)))
        prediction_display = [
            html.H4("Predicted Accident Count:"),
            html.P(f"{avg_pred:,.0f}", className='prediction-value'),
            html.P(f"Model Range: {min(rf_pred, xgb_pred):,.0f} - {max(rf_pred, xgb_pred):,.0f}", 
                  className='prediction-range')
        ]
        confidence_display = [
            html.H4("Model Confidence:"),
            html.Div([
                html.Div(style={'width': f'{confidence:.0f}%'}, className='confidence-bar')
            ], className='confidence-meter'),
            html.P(f"{confidence:.0f}% confidence (±{std_dev:.1f})", className='confidence-value')
        ]
        return prediction_display, confidence_display
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return f"Prediction error: {str(e)}", ""

if __name__ == '__main__':
    try:
        logger.info("Starting application server...")
        app.run(debug=False, host='127.0.0.1', port=8050)
    except Exception as e:
        logger.critical(f"Application failed to start: {str(e)}")
        raise
