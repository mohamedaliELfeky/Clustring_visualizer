# importing modules
import base64
import datetime
import io

import pandas as pd

from dash import Dash, Input, Output, State
import dash_bootstrap_components as dbc

import secondryPage as sp
import helper_function as hf
from sklearn.metrics import silhouette_score


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP])


model_type = 'Agglomerative'
cluster_type = 'k'

app.layout = sp.get_layout(cluster_type=cluster_type,
                            header= model_type + " Clustring")


def get_model(model_type, data, model_arg, norm_type):

    if model_type != 'Dbscan':
        model_arg = model_arg[0]

    if model_type == 'Dbscan':
        print('Dbscan', model_arg)
        result = hf.Dbscan(data, *model_arg, processing=norm_type)

    elif model_type == 'Gaussian_Mixture':
        result = hf.Gaussian_Mixture(data, model_arg, processing=norm_type)

    elif model_type == 'Kmeans':
        result = hf.Kmeans(data, model_arg, processing=norm_type)

    elif model_type == 'Agglomerative':
        result = hf.Agglomerative_Clustering(data, model_arg, processing=norm_type)

    else:
        return -1

    score = silhouette_score(result, result["assignments"])

    return result, score


def process_data(file_contents, model_arg, norm_type):

    df = file_contents  # pd.read_csv(file_contents)

    model_output, accuracy = get_model(model_type=model_type, data=df, model_arg=model_arg, norm_type=norm_type)

    pamaters = f'with parameters num_clusters = {model_arg[0]}'


    if model_type == 'Dbscan':
        pamaters = f'with parameters eps= {model_arg[0]}, num_samples = {model_arg[1]}'

    graph1 = hf.model_plotly(model_output, model_type, pamaters)

    graph2 = hf.get_indicator(accuracy)

    return [graph1, graph2]


def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)

    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            return pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))

        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            return pd.read_excel(io.BytesIO(decoded))

    except Exception as e:
        print(e)

        return None

state_out = None

if cluster_type != 'k':
    state_out = [State('upload_btn', 'contents'),
                 State('upload_btn', 'filename'),
                 State('norm_type', 'value'),
                 State('cluster_num', 'value'),
                 State('min_samples', 'value')]
else:
    state_out = [State('upload_btn', 'contents'),
                 State('upload_btn', 'filename'),
                 State('norm_type', 'value'),
                 State('cluster_num', 'value')]


@app.callback(
    [Output('output_graph', 'figure'),
     Output('accuracy_ind', 'figure')],
    [Input('run_btn', 'n_clicks')],
    state_out
)
def update_graphs(n_clicks, list_of_contents, list_of_names, norm_type, *cluster_arg):

    if not n_clicks:
        return {}, {}

    df = None

    if list_of_contents is not None:

        df = parse_contents(list_of_contents, list_of_names)

    if df is not None and norm_type and cluster_arg:

        return process_data(df, cluster_arg, norm_type)

    return [None, None]






if __name__ == '__main__':
    app.run_server(debug=True)