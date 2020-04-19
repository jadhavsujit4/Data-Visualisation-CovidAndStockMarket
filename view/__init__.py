import hashlib
import logging
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from .line_graph import line_graph_panel
from .scatter_plot import scatter_panel
from .table import table_panel
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# from .title import title_panel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Gray hex color with and without transparency
GRAY = "#d9d9d9"
GRAY_TRANSP = GRAY + "80"

# enum for mapping variables to views
title_mapping = {
"confirmed": "Confirmed Cases",
"deaths": "Deaths",
"recovered": "Recovered",
"growth": "Growth Factor",
"daily_increase": "Daily Increase",
"cumulative": "Cumulative",
"log": "Logarithmic",
"linear": "Linear",
"case_fatality": "Case Fatality Ratio",
"since_100": "Growth Since 100 Cases",
"since_10": "Growth Since 10 Cases",
}
df = pd.read_csv("stock_for_covid.csv")  


layout_parent = dict(
    autosize=True,
    automargin=True,
    hovermode="closest",
    # legend=dict(font=dict(size=10), orientation="v"),
    )

app_layout = html.Div(
    [
    html.Div([html.H1("Covid-19 and Stock Market Analysis", style={"textAlign": "center"}), dcc.Markdown('''
        CS7DS4 - Data Visualisation - Assignment 3
        ''', style={"textAlign": "center"})  ,
    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='Covid-19', children=[
            html.Div([html.H1("Covid-19 around the world", style={"textAlign": "center"}),
                html.Div(
                    id="output-clientside"
        ),  # empty Div to trigger javascript file for graph resizing
        # title_panel,
        line_graph_panel,
        scatter_panel,
        table_panel,
        ],
        id="mainContainer",
        # style={"display": "inline"}
        )        
            ], className="container"),

        dcc.Tab(label='Stock Prices', children=[
            html.Div([html.H1("Stock Market Dataset Introduction", style={'textAlign': 'center'}),
                dash_table.DataTable(
                    id='table',
                    columns=[{"name": i, "id": i} for i in df.columns],
                    data=df.iloc[0:5,:].to_dict("rows"),
                    ),
                html.H1("Facebook Stocks High vs Lows", style={'textAlign': 'center', 'padding-top': 5}),
                dcc.Dropdown(id='my-dropdown',options=[{'label': 'S&P 500', 'value': 'SNP'},{'label': 'Nifty Fifty', 'value': 'NFT'},{'label': 'Facebook', 'value': 'FB'},{'label': 'Microsoft', 'value': 'MCR'},{'label': 'Crude Oil', 'value': 'CRUD'},{'label': 'Reliance', 'value': 'REL'}],
                    multi=True,value=['FB'],style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "80%"}),
                dcc.Graph(id='highlow'),  dash_table.DataTable(
                    id='table2',
                    columns=[{"name": i, "id": i} for i in df.describe().reset_index().columns],
                    data= df.describe().reset_index().to_dict("rows"),
                    ),
                html.H1("Facebook Market Volume", style={'textAlign': 'center', 'padding-top': 5}),
                dcc.Dropdown(id='my-dropdown2',options=[{'label': 'S&P 500', 'value': 'SNP'},{'label': 'Nifty Fifty', 'value': 'NFT'},{'label': 'Facebook', 'value': 'FB'},{'label': 'Microsoft', 'value': 'MCR'},{'label': 'Crude Oil', 'value': 'CRUD'},{'label': 'Reliance', 'value': 'REL'}],
                    multi=True,value=['FB'],style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "80%"}),
                dcc.Graph(id='volume'),
                html.H1("Scatter Analysis", style={'textAlign': 'center', 'padding-top': -10}),
                dcc.Dropdown(id='my-dropdown3',
                   options=[{'label': 'S&P 500', 'value': 'SNP'},{'label': 'Nifty Fifty', 'value': 'NFT'},{'label': 'Facebook', 'value': 'FB'},{'label': 'Microsoft', 'value': 'MCR'},{'label': 'Crude Oil', 'value': 'CRUD'},{'label': 'Reliance', 'value': 'REL'}],
                   value= 'FB',
                   style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "45%"}),
                dcc.Dropdown(id='my-dropdown4',
                   options=[{'label': 'S&P 500', 'value': 'SNP'},{'label': 'Nifty Fifty', 'value': 'NFT'},{'label': 'Facebook', 'value': 'FB'},{'label': 'Microsoft', 'value': 'MCR'},{'label': 'Crude Oil', 'value': 'CRUD'},{'label': 'Reliance', 'value': 'REL'}],
                   value= 'MCR',
                   style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "45%"}),
                dcc.RadioItems(id="radiob", value= "High", labelStyle={'display': 'inline-block', 'padding': 10},
                   options=[{'label': "High", 'value': "High"}, {'label': "Low", 'value': "Low"} , {'label': "Volume", 'value': "Volume"}],
                   style={'textAlign': "center", }),
                dcc.Graph(id='scatter')
                ], className="container"),
            ]),
        # dcc.Tab(label='Covid-19 on World Map', children=[
            # html.Div([html.H1("Covid-19 on World Map", style={'textAlign': 'center'}),

                # trace2 = {
                # "type": "choropleth",
                # "locations" : df['CODE'].tolist(),
                # "z" : df['Total Cases'],
                # "text" : df['Country'],
                # "colorscale" : px.colors.sequential.SunsetMint,
                # "autocolorscale" : False,
                # "marker_line_width" : 0.5,
                # # "reversescale"=True,
                # # "marker_line_color"='darkgray',
                
                # # colorbar_tickprefix = '$',
                # "colorbar_title" : 'Total Cases',



                # # "name": str(y_axis_feature),
                # # "hoverlabel": {"bgcolor": "white", },
                # "colorbar": go.choropleth.ColorBar(title=str(y_axis_feature), xanchor="left"),
                # "colorscale": px.colors.sequential.Sunset,
                # }
                # layout = {

                # "geo": {
                # "showframe": False,
                # "showcoastlines": True,
                # "showocean": True,
                # # "landcolor": "#d8d8d8",
                # # "oceancolor": "#cef6f7",
                # "projection": go.layout.geo.Projection(type='equirectangular'),
                # },

                # "title": {"text": "<b> Covid-19 On World Map</b>",
                # "font": {"size": 15}, },
                # "margin": {'l': 40, 'b': 40, 't': 100, 'r': 10, },
                # "font": {'color': colors['text'], "family": "Roboto, sans-serif",},
                # "autosize": True,
                # }
                # fig=go.Figure(
                #     data=[
                #         type=choropleth,
                #         "locations" : df['CODE'].tolist(),
                #         "z" : df['Total Cases'],
                #         "text" : df['Country'],
                #         "colorscale" : px.colors.sequential.SunsetMint,
                #         "autocolorscale" : False,
                #         "marker_line_width" : 0.5,
                #         # "reversescale"=True,
                #         # "marker_line_color"='darkgray',
                        
                #         # colorbar_tickprefix = '$',
                #         "colorbar_title" : 'Total Cases',



                #         # "name": str(y_axis_feature),
                #         # "hoverlabel": {"bgcolor": "white", },
                #         "colorbar": go.choropleth.ColorBar(title=str(y_axis_feature), xanchor="left"),
                #         "colorscale": px.colors.sequential.Sunset,    
                #         ], 
                #     layout=["geo": {
                #         "showframe": False,
                #         "showcoastlines": True,
                #         "showocean": True,
                #         # "landcolor": "#d8d8d8",
                #         # "oceancolor": "#cef6f7",
                #         "projection": go.layout.geo.Projection(type='equirectangular'),
                #         },

                #         "title": {"text": "<b> Covid-19 On World Map</b>",
                #         "font": {"size": 15}, },
                #         "margin": {'l': 40, 'b': 40, 't': 100, 'r': 10, },
                #         "font": {'color': colors['text'], "family": "Roboto, sans-serif",},
                #         "autosize": True,])
                 
                # dcc.Graph(id="world_map")
                # fig
            # ])
        # ])
        

    ]),

])
,
])
# ,
#     html.Div(
#         id="output-clientside"
#         ),  # empty Div to trigger javascript file for graph resizing
#         # title_panel,
#         line_graph_panel,
#         scatter_panel,
#         table_panel,
#         ],
#         id="mainContainer",
#         style={"display": "flex", "flex-direction": "column"},
#         )


def get_world_map():
    fig=go.Figure(data=go.Choropleth(
                    locations = df['CODE'],
                    z = df['Total Cases'],
                    text = df['Country'],
                    colorscale = 'Mint',
                    autocolorscale=False,
                    # reversescale=True,
                    # marker_line_color='darkgray',
                    marker_line_width=0.5,
                    # colorbar_tickprefix = '$',
                    colorbar_title = 'Total Cases',
                ))

    fig.update_layout(
                    title_text='Coronavirus on World Map',
                    geo=dict(
                        showframe=False,
                        showcoastlines=False,
                        projection_type='equirectangular'
                    ),
                    annotations = [dict(
                        x=0.55,
                        y=0.1,
                        xref='paper',
                        yref='paper',
                        showarrow = False
                    )]
                )
    return fig



def get_color(country: str) -> str:
    """Return a unique color hex code based on the country name."""
    if "double" in country:
        return GRAY

        def _get_unique_int_from_str(s):
            return int.from_bytes(hashlib.sha256(s.encode()).digest(), "big") % 255

            return "#%02X%02X%02X" % (
                _get_unique_int_from_str(country[0]),
                _get_unique_int_from_str(country[1]),
                _get_unique_int_from_str(country[2]) if len(country) > 2 else 128,
                )
