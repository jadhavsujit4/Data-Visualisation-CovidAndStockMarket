import dash_html_components as html

# sources = {
#     "outworldindata": "https://ourworldindata.org/coronavirus",
#     "worldometer": "https://www.worldometers.info/coronavirus/",
#     "jhu": "https://github.com/CSSEGISandData/COVID-19",
# }

# title_panel = html.Div(
#     [
#         html.Div(
#             [
#                 html.Div(
#                     [
#                         html.H5("Source Code:"),
#                         html.H6(
#                             html.A(
#                                 "https://github.com/jadhavsujit4/covid-19AndStockMarket",
#                                 href="https://github.com/jadhavsujit4/covid-19AndStockMarket",
#                                 target="_blank",
#                             )
#                         ),
#                         html.H5("Data Sources:"),
#                     ]
#                     + [
#                         html.H6(html.A(url, href=url, target="_blank"))
#                         for url in sources.values()
#                     ],
#                 ),
#             ],
#             className="one-third column",
#         ),
#         html.Div(
#             [
#                 html.Div(
#                     [
#                         html.H1(
#                             "Covid-19 Trends",
#                             style={"margin-bottom": "0px", "margin-top": "0px"},
#                         ),
#                     ]
#                 ),
#             ],
#             className="one-half column",
#             id="title",
#         ),
#         html.Div(
#             [html.Div(),],
#             className="one-third column",
#             style={"vertical-align": "bottom"},
#         ),
#     ],
#     id="header",
#     className="row flex-display",
#     style={"margin-bottom": "0px"},
# )
