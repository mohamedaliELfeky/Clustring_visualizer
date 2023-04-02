from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc

import plotly.graph_objects as go

import helper_function as hf


def get_layout(header, cluster_type, normalize_options=['RobustScaler', 'StandardScaler', 'Normalizer']):
    buttons_layout = dbc.Row(
                                [
                                    dbc.Col(
                                        html.Div(
                                            dcc.Upload(id='upload_btn',
                                                        children=dbc.Button('Upload CSV file'),
                                                        style={"width": "200px",
                                                               "height": "50px",
                                                               "lineHeight": "50px",
                                                               "borderWidth": "1px",
                                                               "borderStyle": "dashed",
                                                               "borderRadius": "5px", "textAlign": "center", "margin": "auto"},
                                            ),
                                        ),
                                    ),
                                    dbc.Col(
                                        dbc.Button("Run", id="run_btn", n_clicks=0, className="btn btn-primary btn-lg ml-2"),
                                        width="auto",
                                    ),
                                ],
                                className="mt-4",
                            )

    graph_layout = dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Card(dcc.Graph(id='output_graph')),
                                        width=8,
                                    ),
                                    dbc.Col(
                                        dbc.Card(dcc.Graph(id='accuracy_ind')),
                                        width=4,
                                    ),
                                ],
                                className="mt-4",
                            )
    layout = html.Div(
        [
            dbc.Row(
                [
                    hf.get_header(header)
                ]
            ),
            html.Br(),
            html.Br(),
            *hf.get_inputs(cluster_type, normalize_options),
            html.Br(),
            html.Div([
                        buttons_layout
                    ],
                        style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center'}
                    ),
            html.Br(),
            graph_layout,

        ]
    )

    return layout
