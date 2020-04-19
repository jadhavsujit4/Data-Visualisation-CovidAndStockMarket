from typing import List, Dict

import logging
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas as pd
import numpy as np
import plotly.graph_objs as go
# from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
from dash.dependencies import ClientsideFunction, Input, Output, State

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

from model import get_doubling_time_ts_df, corona_country_data, corona_table_data
from view import (
    layout_parent,
    app_layout,
    line_graph,
    GRAY,
    GRAY_TRANSP,
    get_color,
    title_mapping,
)
from view.utils import registered_popovers

external_scripts = ['/assets/style.css']

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)
# this is the application/server
app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width"}],
    external_stylesheets=[dbc.themes.LUX], 
    # external_scripts=external_scripts
)
server = app.server
app.layout = app_layout


# dfrel = pd.read_csv("stock_reliance.csv")
# df = pd.read_csv("stock_data.csv")
df = pd.read_csv("stock_for_covid.csv")
df2 = pd.read_csv("dataset_Facebook.csv",";")

df_ml = df2.copy()

lb_make = LabelEncoder()
df_ml["Type"] = lb_make.fit_transform(df_ml["Type"])
df_ml = df_ml.fillna(0)

X = df_ml.drop(['like'], axis = 1).values
Y = df_ml['like'].values

X = StandardScaler().fit_transform(X)

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.30, random_state = 101)

randomforest = RandomForestRegressor(n_estimators=500,min_samples_split=10)
randomforest.fit(X_Train,Y_Train)

p_train = randomforest.predict(X_Train)
p_test = randomforest.predict(X_Test)

train_acc = r2_score(Y_Train, p_train)
test_acc = r2_score(Y_Test, p_test)

# app.layout = html.Div([html.H1("Stock Market Impacted due to Lockdowm", style={"textAlign": "center"}), dcc.Markdown('''
# Welcome to my Plotly (Dash) Data Science interactive dashboard. In order to create this dashboard have been used two different datasets. The first one is the [Huge Stock Market Dataset by Boris Marjanovic](https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs)
# and the second one is the [Facebook metrics Data Set by Moro, S., Rita, P., & Vala, B](https://archive.ics.uci.edu/ml/datasets/Facebook+metrics). This dashboard is divided in 3 main tabs. In the first one you can choose whith which other companies to compare Facebook Stock Prices to anaylise main trends.
# Using the second tab, you can analyse the distributions each of the Facebook Metrics Data Set features. Particular interest is on how paying to advertise posts can boost posts visibility. Finally, in the third tab a Machine Learning analysis of the considered datasets is proposed. 
# All the data displayed in this dashboard is fetched, processed and updated using Python (eg. ML models are trained in real time!).
# ''')  ,
#     dcc.Tabs(id="tabs", children=[
#         dcc.Tab(label='Stock Prices', children=[
# html.Div([html.H1("Dataset Introduction", style={'textAlign': 'center'}),
# dash_table.DataTable(
#     id='table',
#     columns=[{"name": i, "id": i} for i in df.columns],
#     data=df.iloc[0:5,:].to_dict("rows"),
# ),
#     html.H1("Facebook Stocks High vs Lows", style={'textAlign': 'center', 'padding-top': 5}),
#     dcc.Dropdown(id='my-dropdown',options=[{'label': 'S&P 500', 'value': 'SNP'},{'label': 'Nifty Fifty', 'value': 'NFT'},{'label': 'Facebook', 'value': 'FB'},{'label': 'Microsoft', 'value': 'MCR'},{'label': 'Crude Oil', 'value': 'CRUD'},{'label': 'Reliance', 'value': 'REL'}],
#         multi=True,value=['FB'],style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "80%"}),
#     dcc.Graph(id='highlow'),  dash_table.DataTable(
#     id='table2',
#     columns=[{"name": i, "id": i} for i in df.describe().reset_index().columns],
#     data= df.describe().reset_index().to_dict("rows"),
# ),
# html.H1("Facebook Market Volume", style={'textAlign': 'center', 'padding-top': 5}),
#     dcc.Dropdown(id='my-dropdown2',options=[{'label': 'S&P 500', 'value': 'SNP'},{'label': 'Nifty Fifty', 'value': 'NFT'},{'label': 'Facebook', 'value': 'FB'},{'label': 'Microsoft', 'value': 'MCR'},{'label': 'Crude Oil', 'value': 'CRUD'},{'label': 'Reliance', 'value': 'REL'}],
#         multi=True,value=['FB'],style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "80%"}),
#     dcc.Graph(id='volume'),
#     html.H1("Scatter Analysis", style={'textAlign': 'center', 'padding-top': -10}),
#     dcc.Dropdown(id='my-dropdown3',
#                  options=[{'label': 'S&P 500', 'value': 'SNP'},{'label': 'Nifty Fifty', 'value': 'NFT'},{'label': 'Facebook', 'value': 'FB'},{'label': 'Microsoft', 'value': 'MCR'},{'label': 'Crude Oil', 'value': 'CRUD'},{'label': 'Reliance', 'value': 'REL'}],
#                  value= 'FB',
#                  style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "45%"}),
#     dcc.Dropdown(id='my-dropdown4',
#                  options=[{'label': 'S&P 500', 'value': 'SNP'},{'label': 'Nifty Fifty', 'value': 'NFT'},{'label': 'Facebook', 'value': 'FB'},{'label': 'Microsoft', 'value': 'MCR'},{'label': 'Crude Oil', 'value': 'CRUD'},{'label': 'Reliance', 'value': 'REL'}],
#                  value= 'MCR',
#                  style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "45%"}),
#   dcc.RadioItems(id="radiob", value= "High", labelStyle={'display': 'inline-block', 'padding': 10},
#                  options=[{'label': "High", 'value': "High"}, {'label': "Low", 'value': "Low"} , {'label': "Volume", 'value': "Volume"}],
#  style={'textAlign': "center", }),
#     dcc.Graph(id='scatter')
# ], className="container"),
# ])

# @app.callback(Output('world_map', 'figure'))
# def get_ world_map(selected_dropdown):
#     fig=go.Figure(data=go.Choropleth(
#                     locations = df['CODE'],
#                     z = df['Total Cases'],
#                     text = df['Country'],
#                     colorscale = 'Mint',
#                     autocolorscale=False,
#                     # reversescale=True,
#                     # marker_line_color='darkgray',
#                     marker_line_width=0.5,
#                     # colorbar_tickprefix = '$',
#                     colorbar_title = 'Total Cases',
#                 ))

#     fig.update_layout(
#                     title_text='Coronavirus on World Map',
#                     geo=dict(
#                         showframe=False,
#                         showcoastlines=False,
#                         projection_type='equirectangular'
#                     ),
#                     annotations = [dict(
#                         x=0.55,
#                         y=0.1,
#                         xref='paper',
#                         yref='paper',
#                         showarrow = False
#                     )]
#                 )
#     return fig

@app.callback(Output('highlow', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown):
    dropdown = {"SNP": "S&P 500","NFT": "Nifty Fifty","FB": "Facebook","MCR": "Microsoft","CRUD": "Crude Oil","REL": "Reliance",}
    # {'label': 'S&P 500', 'value': 'SNP'},{'label': 'Nifty Fifty', 'value': 'NFT'},{'label': 'Facebook', 'value': 'FB'},{'label': 'Microsoft', 'value': 'MCR'},{'label': 'Crude Oil', 'value': 'CRUD'},{'label': 'Reliance', 'value': 'REL'}
    trace1 = []
    trace2 = []
    for stock in selected_dropdown:
        trace1.append(go.Scatter(x=df[df["Stock"] == stock]["Date"],y=df[df["Stock"] == stock]["High"],mode='lines',
            opacity=0.7,name=f'High {dropdown[stock]}',textposition='bottom center'))
        trace2.append(go.Scatter(x=df[df["Stock"] == stock]["Date"],y=df[df["Stock"] == stock]["Low"],mode='lines',
            opacity=0.6,name=f'Low {dropdown[stock]}',textposition='bottom center'))
    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
        'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
            height=600,title=f"High and Low Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 'step': 'month', 'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M', 'step': 'month', 'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},yaxis={"title":"Price (USD)"},     paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)')}
    return figure

@app.callback(Output('volume', 'figure'),
              [Input('my-dropdown2', 'value')])
def update_graph(selected_dropdown_value):
    dropdown = {"SNP": "S&P 500","NFT": "Nifty Fifty","FB": "Facebook","MCR": "Microsoft","CRUD": "Crude Oil","REL": "Reliance",}
    trace1 = []
    for stock in selected_dropdown_value:
        trace1.append(go.Scatter(x=df[df["Stock"] == stock]["Date"],y=df[df["Stock"] == stock]["Volume"],mode='lines',
            opacity=0.7,name=f'Volume {dropdown[stock]}',textposition='bottom center'))
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
        'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
            height=600,title=f"Market Volume for {', '.join(str(dropdown[i]) for i in selected_dropdown_value)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 'step': 'month', 'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M', 'step': 'month', 'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},yaxis={"title":"Transactions Volume"} ,   paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)')}
    return figure

@app.callback(Output('scatter', 'figure'),
              [Input('my-dropdown3', 'value'), Input('my-dropdown4', 'value'), Input("radiob", "value"),])
def update_graph(stock, stock2, radioval):
    dropdown = {"SNP": "S&P 500","NFT": "Nifty Fifty","FB": "Facebook","MCR": "Microsoft","CRUD": "Crude Oil","REL": "Reliance",}
    radio = {"High": "High Prices", "Low": "Low Prices", "Volume": "Market Volume", }
    trace1 = []
    if (stock == None) or (stock2 == None):
        trace1.append(
            go.Scatter(x= [0], y= [0],
                       mode='markers', opacity=0.7, textposition='bottom center'))
        traces = [trace1]
        data = [val for sublist in traces for val in sublist]
        figure = {'data': data,
                  'layout': go.Layout(colorway=['#FF7400', '#FFF400', '#FF0056'],
                                      height=600, title=f"{radio[radioval]}",
                                      paper_bgcolor='rgba(0,0,0,0)',
                                      plot_bgcolor='rgba(0,0,0,0)')}
    else:
        trace1.append(go.Scatter(x=df[df["Stock"] == stock][radioval][-1000:], y=df[df["Stock"] == stock2][radioval][-1000:],
                       mode='markers', opacity=0.7, textposition='bottom center'))
        traces = [trace1]
        data = [val for sublist in traces for val in sublist]
        figure = {'data': data,
            'layout': go.Layout(colorway=['#FF7400', '#FFF400', '#FF0056'],
                height=600,title=f"{radio[radioval]} of {dropdown[stock]} vs {dropdown[stock2]} Over Time (1000 iterations)",
                xaxis={"title": stock,}, yaxis={"title": stock2},     paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)')}
    return figure

# Create callback for resizing the charts
# app.clientside_callback(
#     ClientsideFunction(namespace="clientside", function_name="resize"),
#     Output("output-clientside", "children"),
#     [Input("count_graph", "figure")],
# )


@app.callback(
    Output("count_graph", "figure"),
    [
        Input("countries", "value"),
        Input("data_source", "value"),
        Input("line_graph_view", "value"),
        Input("line_graph_scaler", "value"),
        Input("date_slider", "value"),
        Input("count_graph", "hoverData"),
    ],
)
def update_time_series(
    countries: List[str],
    data_source: str,
    line_graph_view: str,
    line_graph_scaler: str,
    date_slider: int,
    hover_data: dict,
) -> Dict[str, List]:

    if line_graph_view == "trajectory":
        return update_trajectory_chart(
            countries, data_source, line_graph_scaler, date_slider, hover_data,
        )

    # special filtering for viewing "from n days setting"
    if line_graph_view in ["since_100", "since_10"]:
        df = get_doubling_time_ts_df(countries, line_graph_view, data_source)
        countries += list(filter(lambda x: "double" in x, df.index))
        x_vals = df.columns
    else:
        df = corona_country_data[data_source][line_graph_view].copy()
        x_vals = [x[:-3] for x in df.columns]

    # popualte the data output field
    data = []
    for country in countries:
        if country in df.index:
            data.append(
                dict(
                    type="scatter",
                    mode="lines",
                    name=country,
                    showlegend=True if "double" not in country else False,
                    y=df.loc[country],
                    x=x_vals,
                    line=dict(shape="spline", smoothing="2", color=get_color(country),),
                )
            )

    title = f"{title_mapping[data_source]} - {title_mapping[line_graph_view]}"
    layout_count = {
        **layout_parent,
        "title": title,
        "xaxis": {
            "title": "Development Time (days)",
            "showspikes": True,
            "spikethickness": 1,
        },
        "yaxis": {
            "title": f"Count - {title_mapping[line_graph_scaler]}",
            "type": line_graph_scaler,
            "showspikes": True,
            "spikethickness": 1,
        },
        "margin": {"l": 70, "b": 70, "r": 10, "t": 50},
    }

    return dict(data=data, layout=layout_count)


def update_trajectory_chart(
    countries: List[str],
    data_source: str,
    line_graph_scaler: str,
    date_slider: int,
    hover_data,
) -> Dict[str, List]:

    country_on_hover = None
    if hover_data:
        # divide by two because we are graphing two traces at at time (line and scatter)
        idx = hover_data["points"][0]["curveNumber"] // 2
        country_on_hover = countries[idx]

    # ts data to operate on
    df_cumulative = corona_country_data[data_source]["cumulative"]
    df_daily = corona_country_data[data_source]["daily_increase"]
    date = pd.to_datetime(df_cumulative.columns[date_slider])

    # popualte the time series data output field.
    # if the date and minimum number of cases is exceeded, then plot the scatter and line trace.
    data = []
    for country in countries:
        if country in df_cumulative.index:
            x = df_cumulative.loc[country]
            y = df_daily.loc[country]
            index = (pd.to_datetime(x.index) <= date) & (x > 50)
            trace_color = (
                get_color(country) if country == country_on_hover else GRAY_TRANSP
            )
            if True in list(index):
                data.append(
                    dict(
                        type="scatter",
                        mode="lines",
                        name=country,
                        y=y[index],
                        x=x[index],
                        customdata=country,
                        line=dict(shape="spline", smoothing="2", color=trace_color),
                        showlegend=False,
                    )
                )
                data.append(
                    dict(
                        type="scatter",
                        y=[y[index].iloc[-1]],
                        x=[x[index].iloc[-1]],
                        text=country,
                        name=country,
                        mode="markers+text",
                        textposition="top center",
                        showlegend=False,
                        marker={"size": 8, "color": get_color(country),},
                    ),
                )

    title = f"Trajectory of Covid {title_mapping[data_source]} {df_cumulative.columns[date_slider - 1]}"
    layout_count = {
        **layout_parent,
        "autosize": False,
        "title": title,
        "xaxis": {
            "title": "Total Count",
            "type": line_graph_scaler,
            "showspikes": True,
            "spikethickness": 1,
        },
        "yaxis": {
            "title": "Daily Increase",
            "type": line_graph_scaler,
            "showspikes": True,
            "spikethickness": 1,
        },
        "margin": {"l": 70, "b": 70, "r": 10, "t": 50},
    }

    return dict(data=data, layout=layout_count)


@app.callback(
    Output("scatter_plot", "figure"),
    [
        Input("countries", "value"),
        Input("scatter_x_data", "value"),
        Input("scatter_y_data", "value"),
        Input("scatter_x_scaler", "value"),
        Input("scatter_y_scaler", "value"),
        Input("min_cases_thresh", "value"),
        Input("show_labels", "value"),
    ],
)
def update_scatter_plot(
    countries: List[str],
    x_axis: str,
    y_axis: str,
    x_scaler: str,
    y_scaler: str,
    min_cases_thresh: str,
    show_labels: str,
):
    logger.info(corona_table_data["Total Cases"][0])
    corona_table_data["Total Cases"] = corona_table_data["Total Cases"].replace(',', '')
    logger.info(corona_table_data["Total Cases"][0])    
    df = corona_table_data[corona_table_data["Total Cases"].astype(float) > float(min_cases_thresh)]
    names = df["Country"] if show_labels else countries
    colors = list(
        map(lambda x: get_color(x) if x in countries else GRAY, df["Country"])
    )
    data = [
        dict(
            type="scatter",
            y=df[y_axis],
            x=df[x_axis],
            text=names,
            name=df["Country"],
            mode="markers+text",
            textposition="top center",
            showlegend=False,
            marker={
                "size": 8,
                "opacity": 1,
                "line": {"width": 0.5, "color": "white"},
                "color": colors,
            },
        )
    ]

    layout_scatter = {
        **layout_parent,
        "title": f"{y_axis} vs. {x_axis}",
        "xaxis": {
            "title": f"{x_axis} {title_mapping[x_scaler]}",
            "type": x_scaler,
            "showspikes": True,
            "spikethickness": 1,
        },
        "yaxis": {
            "title": f"{y_axis} {title_mapping[y_scaler]}",
            "type": y_scaler,
            "showspikes": True,
            "spikethickness": 1,
        },
        "margin": {"l": 70, "b": 70, "r": 10, "t": 50},
        "textposition": "top center",
    }
    return dict(data=data, layout=layout_scatter)


@app.callback(
    [Output("data_table", "selected_rows"), Output("data_table", "data")],
    [Input("countries", "value")],
)
def update_table_selection_from_country_dropdown(countries: List[str]):
    """Update the Data Table selection and sorting, based on the country dropdown."""

    # move selected countries to top of table
    df = corona_table_data.copy()
    df["new"] = list(range(1, len(df) + 1))
    df.loc[df[df.Country.isin(countries)].index, "new"] = 0
    df = df.sort_values(["new", "Total Cases"], ascending=[True, False])
    df = df.drop("new", axis=1)
    return [
        list(range(len(countries))),
        df.to_dict("records"),
    ]


@app.callback(
    dash.dependencies.Output("line_graph_view", "options"),
    [dash.dependencies.Input("data_source", "value")],
)
def hide_cases_since_dropdowns_if_case_fatalit_set(value: str):
    if value == "case_fatality":
        return line_graph.line_graph_view_options[:3]
    else:
        return line_graph.line_graph_view_options


@app.callback(
    dash.dependencies.Output("date_slider_div", "style"),
    [dash.dependencies.Input("line_graph_view", "value")],
)
def hide_date_slider_if_trajectory_not_set(value: str):
    if value == "trajectory":
        return {"display": "block"}
    else:
        return {"display": "none"}


# dynamically create callbacks for each about-info popover we created
def _toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open


for p in registered_popovers:
    app.callback(
        Output(f"popover-{p}", "is_open"),
        [Input(f"popover-target-{p}", "n_clicks")],
        [State(f"popover-{p}", "is_open")],
    )(_toggle_popover)

if __name__ == "__main__":
    # app.run_server(debug=True, port=8001)
    app.run_server(debug=True, use_reloader=True)
