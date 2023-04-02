import dash_bootstrap_components as dbc
from dash import dcc
import plotly.graph_objects as go
from dash import html
import plotly.express as px


from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer



import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture



def get_header(header: str):

    header_component = html.H1(header, style={'display': 'flex',
                                              'justify-content': 'center',
                                              'align-items': 'center',
                                              'color': '#9191FF'
                                              })

    return header_component


def get_indicator(indicator_val=0):

    fig = go.Figure(
                    go.Indicator(
                                    mode="gauge+number",
                                    value=indicator_val,
                                    title={'text': "Silhouette Score"},
                                    domain={'x': [0, 1], 'y': [0, 1]}
                                )
                    )

    return fig


def get_graph(title, original_data, kargs):

    graph1 = go.Figure(
                        data=[
                            go.Scatter(
                                            x=original_data[:, 0],
                                            y=original_data[:, 1],
                                            mode='markers',
                                            marker=kargs
                                        )
                            ]
                     )

    graph1.update_layout(
        xaxis_title='First PCA component',
        yaxis_title='Second PCA component',
        title=title
    )

    return graph1


def add_normalizer(normalize_options):

    return [html.Div(
                    [
                        dbc.Label("Select Scaling method:",
                                  style={'font-size': '25px',
                                         'color': '#9191F1',
                                         'padding-left': 15,
                                         'padding-right': 0,
                                         'text-align': 'left'
                                         }
                                  ),
                        dcc.Dropdown(
                            id='norm_type',
                            options=normalize_options,
                            value=normalize_options[-1],
                            style={
                                'width': '200px',
                                'margin-right': '50px',
                                'margin-left': '5px'
                            }
                        )
                    ],
                    style={
                        'display': 'flex',
                        'align-items': 'center',
                        'justify-content': 'flex-start'
                    }
                )]


def k_type_cluster(method_id):

    layout = html.Div([
            dbc.Label(f"Enter {method_id}:",
                      style={'font-size': '25px',
                             'color': '#9191F1',
                             'padding-left': 15,
                             'padding-right': 0,
                             'text-align': 'left'
                             }
                      ),

            dbc.Input(
                id= 'min_samples' if method_id == 'min_samples' else 'cluster_num',
                placeholder=method_id,
                type="number",
                style={'margin-left': "10px", 'width': '20%'}
            )
        ],
            style={
                'display': 'flex',
                'align-items': 'center',
                'justify-content': 'flex-start'
            }
        )


    return [layout]


def get_inputs(method_id, normalize_options):

    out = [
            dbc.Row(
                    k_type_cluster(method_id=method_id),
                ),

            html.Br()
        ]

    if method_id == 'eps':
        out.append(
                dbc.Row(
                    k_type_cluster(method_id='min_samples')
                )
            )
    out.append(html.Br())

    out.append(
        dbc.Row(
                add_normalizer(normalize_options)
        )
    )

    return out

# style={
#             'display': 'flex',
#             'flex-direction': 'column',
#             'margin-left': '20px',
#             'margin-right': '100px',
#             'justify-content': 'flex-start'
#         }

def preprocessing(data, processing):

    if processing == 'StandardScaler':
        std = StandardScaler()
        data = std.fit_transform(data)

    elif processing == 'RobustScaler':
        rbts = RobustScaler()
        data = rbts.fit_transform(data)
    elif processing == 'Normalizer':
        nz = Normalizer()
        data = nz.fit_transform(data)
    else:
        raise Exception("Sorry, Enter availd preprocessing method form [StandardScaler, RobustScaler, Normalizer ] ")

    return pd.DataFrame(data)


def Dbscan(data, eps, num_samples, processing='StandardScaler'):

    data = preprocessing(data, processing)
    db = DBSCAN(eps=eps, min_samples=num_samples).fit(data)
    labsList = ["Noise"]
    labsList = labsList + ["Cluster " + str(i) for i in range(1, len(set(db.labels_)))]

    data["assignments"] = db.labels_

    return data


def Kmeans(data, num_clusters, processing='StandardScaler'):
    data = preprocessing(data, processing)
    km = KMeans(num_clusters)
    km.fit(data)
    data['assignments'] = km.labels_

    return data


def Agglomerative_Clustering(data, num_clusters, processing='StandardScaler'):
    data = preprocessing(data, processing)
    hac = AgglomerativeClustering(num_clusters)
    hac.fit(data)
    data['assignments'] = hac.labels_

    return data


def Gaussian_Mixture(data, num_clusters=3, processing='StandardScaler'):
    data = preprocessing(data, processing)
    gm = GaussianMixture(num_clusters)
    gm.fit(data)
    data['assignments'] = gm.predict(data)

    return data

def model_plotly(data, model, paramaters):
    df = pd.DataFrame()
    df[['pca1', 'pca2']] = data[[0, 1]].copy()
    pca = PCA(n_components=2)
    df[['pca1', 'pca2']] = pca.fit_transform(df)
    df['assignments'] = data["assignments"]

    px.defaults.template = "ggplot2"
    px.defaults.color_continuous_scale = px.colors.sequential.Blackbody
    fig = px.scatter(df, x='pca1', y='pca2', color="assignments",
                     hover_data=['assignments'],

                     labels={
                         "pca1": "Frist Component",
                         "pca2": "Second Component",
                         "assignments": "clasee"
                     })

    fig.update_layout(
        title={
            'text': f"{model} {paramaters}",
            'y': 1,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="RebeccaPurple"
        ), )

    return fig


