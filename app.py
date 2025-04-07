# Combined Dash App: SDOH Visualizations + Health Score Predictor

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import dash
from dash import dcc, html, Input, Output
import json
import requests
import os

# Load data
SDOH_2010 = pd.read_csv('SDOH_2010_Final.csv')
SDOH_2015 = pd.read_csv('SDOH_2015_Final.csv')
SDOH_2020 = pd.read_csv('SDOH_2020_Final.csv')

# Add year and merge datasets
for df, year in zip([SDOH_2010, SDOH_2015, SDOH_2020], [2010, 2015, 2020]):
    df['Year'] = year

SDOH_all = pd.concat([SDOH_2010, SDOH_2015, SDOH_2020])
SDOH_all['CountyFIPS'] = SDOH_all['CountyFIPS'].astype(str).str.zfill(5)

# Load GeoJSON for Texas counties
texas_geojson = requests.get("https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json").json()

# Load pre-trained model
rf_loaded = joblib.load("health_score_model.joblib")

# Define SDOH categories and color mapping
sdo_categories = {
    "Economic Stability": ["Median_Household_Income", "Unemployment_Rate", "Income_Inequality_Gini",
                           "Households_on_Food_Stamps", "Households_Income_Below_10K",
                           "Population_Income_Above_200PCT_Poverty", "Poverty_Rate"],
    "Education": ["High_School_Graduation_Rate", "Bachelor_Degree_Rate", "Less_Than_High_School_Edu",
                 "Graduate_Degree_Rate", "Youth_Not_in_School_or_Work"],
    "Healthcare Access": ["Uninsured_Rate_Under_64", "Hospitals_with_Ambulance_per_1K", "Median_Distance_to_ER",
                         "Median_Distance_to_Clinic", "Advanced_Nurses_per_1K", "Primary_Care_Shortage_Score",
                         "Households_No_Vehicle"],
    "Neighborhood and Environment": ["Median_Home_Value", "Percent_Homes_Built_Pre_1979", "Days_Heat_Index_Above_100F",
                                     "Storm_Injuries_Total"],
    "Community Context": ["Elderly_Living_Alone", "Single_Parent_Households", "Non_Citizen_Population",
                          "Limited_English_Households"]
}

category_colors = {
    "Economic Stability": "#FF746C",
    "Education": "#f3ea76",
    "Healthcare Access": "#53abff",
    "Neighborhood and Environment": "#7edb8e",
    "Community Context": "#c9b0e9"
}

# Define feature ranges for sliders
feature_ranges = {
    "Median_Household_Income": (30931, 105956),
    "Graduate_Degree_Rate": (0, 20),
    "Median_Distance_to_ER": (0.3, 31.48),
    "Elderly_Living_Alone": (4.9, 20.97),
    "Median_Home_Value": (61800, 378500)
}

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Texas Health Dashboard"

app.layout = html.Div([
    html.H1("Texas County Health Dashboard", style={"textAlign": "center"}),

    # --- Choropleth Map Section ---
    html.Div([
        html.H2("Geospatial Health Data"),
        dcc.RadioItems(
            id="map-type",
            options=[
                {"label": "Adult Health Score", "value": "score"},
                {"label": "Max SDOH Category", "value": "sdoh"}
            ],
            value="score",
            labelStyle={"display": "inline-block", "marginRight": "15px"}
        ),
        dcc.Slider(
            id="year-slider",
            min=2010,
            max=2020,
            step=5,
            marks={2010: "2010", 2015: "2015", 2020: "2020"},
            value=2010
        ),
        dcc.Graph(id="map")
    ]),

    # --- Pie Chart ---
    html.H3("SDOH Contribution Breakdown for Selected County"),
    dcc.Graph(id="pie-chart"),

    # --- Scatterplot ---
    html.H3("Explore Relationship Between SDOH Variables and Adult Health Score"),
    html.Div([
        html.Label("Select SDOH Category"),
        dcc.Dropdown(
            id="scatter-category",
            options=[{"label": k, "value": k} for k in sdo_categories.keys()],
            value="Economic Stability",
            style={"width": "300px"}
        ),
        html.Label("Select SDOH Variable"),
        dcc.Dropdown(id="scatter-variable", style={"width": "400px", "marginTop": "10px"})
    ], style={"marginBottom": "20px"}),
    dcc.Graph(id="scatterplot"),

    # --- Correlation Heatmap ---
    html.H3("Correlation Between Selected SDOH Category and All Other Variables"),
    dcc.Dropdown(
        id="category-selector",
        options=[{"label": key, "value": key} for key in sdo_categories.keys()],
        value="Education",
        style={"width": "400px", "marginBottom": "20px"}
    ),
    dcc.Graph(id="correlation-heatmap"),

    # --- Health Score Predictor ---
    html.H2("Health Score Predictor"),
    *[
        html.Div([
            html.Label(f"{feature.replace('_', ' ')}"),
            dcc.Slider(
                id=feature,
                min=feature_ranges[feature][0],
                max=feature_ranges[feature][1],
                step=(feature_ranges[feature][1] - feature_ranges[feature][0]) / 100,
                value=np.mean(feature_ranges[feature]),
                marks={int(feature_ranges[feature][0]): str(int(feature_ranges[feature][0])),
                       int(feature_ranges[feature][1]): str(int(feature_ranges[feature][1]))},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ]) for feature in feature_ranges
    ],
    html.Br(),
    dcc.Graph(id="health_score_gauge")
])

# --- Callback Functions ---

@app.callback(
    Output("map", "figure"),
    Input("map-type", "value"),
    Input("year-slider", "value")
)
def update_map(map_type, year):
    df_filtered = SDOH_all[SDOH_all['Year'] == year]
    if map_type == "score":
        fig = px.choropleth(
            df_filtered,
            geojson=texas_geojson,
            locations="CountyFIPS",
            color="Adult_Health_Score",
            hover_name="County",
            color_continuous_scale=["#FF0000", "#FFFF00", "#32b239"],
            title=f"Adult Health Score by County in Texas ({year})",
            scope="usa"
        )
    else:
        fig = px.choropleth(
            df_filtered,
            geojson=texas_geojson,
            locations="CountyFIPS",
            color="Max_SDOH_Category",
            hover_name="County",
            color_discrete_map=category_colors,
            title=f"Max SDOH Category by County in Texas ({year})",
            scope="usa"
        )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(geo=dict(center={"lat": 31.0, "lon": -99.0}, projection_scale=7))
    return fig

@app.callback(
    Output("pie-chart", "figure"),
    Input("map", "clickData"),
    Input("year-slider", "value")
)
def update_pie_chart(clickData, year):
    if clickData is None:
        return px.pie(title="Select a county on the map")
    county = clickData["points"][0]["hovertext"]
    df_county = SDOH_all[(SDOH_all['County'] == county) & (SDOH_all['Year'] == year)]
    df_dataset = SDOH_all[SDOH_all['Year'] == year]
    if df_county.empty:
        return px.pie(title="Data unavailable")
    weights = compute_category_weights_per_county(df_county, df_dataset)
    if not weights:
        return px.pie(title=f"Invalid data for {county}")
    fig = px.pie(
        names=list(weights.keys()),
        values=list(weights.values()),
        title=f"{county} - SDOH Contributions ({year})",
        hole=0.4,
        color=list(weights.keys()),
        color_discrete_map=category_colors
    )
    fig.update_traces(textinfo="percent+label")
    return fig

@app.callback(
    Output("scatter-variable", "options"),
    Output("scatter-variable", "value"),
    Input("scatter-category", "value")
)
def update_variable_dropdown(category):
    variables = sdo_categories.get(category, [])
    options = [{"label": var, "value": var} for var in variables if var in SDOH_all.columns]
    default_value = options[0]["value"] if options else None
    return options, default_value

@app.callback(
    Output("scatterplot", "figure"),
    Input("scatter-variable", "value"),
    Input("year-slider", "value")
)
def update_scatterplot(variable, year):
    if variable is None:
        return px.scatter(title="No variable selected")
    df = SDOH_all[SDOH_all['Year'] == year]
    if variable not in df.columns or "Adult_Health_Score" not in df.columns:
        return px.scatter(title="Data not available")
    fig = px.scatter(
        df,
        x=variable,
        y="Adult_Health_Score",
        hover_name="County",
        title=f"Relationship Between {variable} and Adult Health Score ({year})",
        trendline="ols"
    )
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.update_layout(width=900, height=500)
    return fig

@app.callback(
    Output("correlation-heatmap", "figure"),
    Input("year-slider", "value"),
    Input("category-selector", "value")
)
def update_correlation_matrix(year, selected_category):
    df = SDOH_all[SDOH_all["Year"] == year].copy()
    base_vars = [v for v in sdo_categories[selected_category] if v in df.columns]
    other_vars = [v for k, lst in sdo_categories.items() if k != selected_category for v in lst if v in df.columns]
    df_clean = df[base_vars + other_vars].dropna()
    if df_clean.empty or not base_vars or not other_vars:
        return px.imshow(np.zeros((1, 1)), labels={"x": "No Data", "y": "No Data"})
    corr_matrix = df_clean[base_vars + other_vars].corr().loc[base_vars, other_vars]
    fig = px.imshow(
        corr_matrix,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title=f"Correlation Between {selected_category} and Other SDOH Variables ({year})",
        labels=dict(color="Correlation")
    )
    fig.update_layout(width=900, height=600, margin={"t": 50, "b": 50}, xaxis_tickangle=-60)
    return fig

@app.callback(
    Output("health_score_gauge", "figure"),
    [Input(feature, "value") for feature in feature_ranges]
)
def update_health_score(*values):
    user_input = pd.DataFrame([dict(zip(feature_ranges.keys(), values))])
    health_score = rf_loaded.predict(user_input)[0]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=health_score,
        title={"text": "Predicted Health Score"},
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": "blue"},
            "steps": [
                {"range": [0, 0.3], "color": "#FF0000"},
                {"range": [0.3, 0.6], "color": "#FFFF00"},
                {"range": [0.6, 1], "color": "#32b229"}
            ]
        }
    ))
    return fig

# --- Helper Function ---
def compute_category_weights_per_county(df, dataset):
    df = df.copy()
    if "Adult_Health_Score" not in df.columns:
        return None
    feature_columns = [col for cols in sdo_categories.values() for col in cols if col in df.columns]
    df_features = df[feature_columns].copy()
    imputer = SimpleImputer(strategy="median")
    df_features = pd.DataFrame(imputer.fit_transform(df_features), columns=df_features.columns)
    category_weights = {}
    for category, variables in sdo_categories.items():
        valid_vars = [var for var in variables if var in df_features.columns]
        if valid_vars:
            dataset_category = dataset[valid_vars].copy()
            dataset_category = pd.DataFrame(imputer.fit_transform(dataset_category), columns=dataset_category.columns)
            scaler = StandardScaler()
            scaler.fit(dataset_category)
            county_scaled = pd.DataFrame(scaler.transform(df_features[valid_vars]), columns=valid_vars)
            category_weights[category] = county_scaled.abs().sum(axis=1).values[0]
    total_weight = sum(category_weights.values())
    if total_weight > 0:
        category_weights = {k: (v / total_weight) * 100 for k, v in category_weights.items()}
    return category_weights

# --- Run App ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8060))  # Render sets PORT env variable
    app.run(debug=True, host="0.0.0.0", port=port)
