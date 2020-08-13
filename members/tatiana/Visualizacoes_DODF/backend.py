import base64
import io
import pathlib

import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from PIL import Image
from io import BytesIO
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import scipy.spatial.distance as spatial_distance

# get relative data folder
PATH = pathlib.Path(__file__).parent

data_dict = {
    "DODF": pd.read_csv('dodf_tsne.csv'),
    "DODF_Aposentadoria": pd.read_csv('DODF_Aposentadorias_t-SNE.csv'),
}

# Import datasets here for running the Local version

TSNE_dataset = 'scatteplot3D.csv'

with open("DODF_Explorer_intro.md", "r") as file:
    DODF_Explorer_intro_md = file.read()

with open("DODF_Explorer_description.md", "r") as file:
    DODF_Explorer_description_md = file.read()

# Methods for creating components in the layout code
def Card(children, **kwargs):
    return html.Section(children, className="card-style")


def NamedSlider(name, short, min, max, step, val, marks=None):
    if marks:
        step = None
    else:
        marks = {i: i for i in range(min, max + 1, step)}

    return html.Div(
        style={"margin": "25px 5px 30px 0px"},
        children=[
            f"{name}:",
            html.Div(
                style={"margin-left": "5px"},
                children=[
                    dcc.Slider(
                        id=f"slider-{short}",
                        min=min,
                        max=max,
                        marks=marks,
                        step=step,
                        value=val,
                    )
                ],
            ),
        ],
    )

def Legenda(dataset, app):
    contents = []
    contents.append(html.H5("LEGENDA"))
    if dataset == 'DODF_tsne':
        contents.append(html.P("0: SECRETARIA DE ESTADO DE SEGURANÇA PÚBLICA "))
        contents.append(html.P("1: SECRETARIA DE ESTADO DE CULTURA "))
        contents.append(html.P("2: SECRETARIA DE ESTADO DE FAZENDA, PLANEJAMENTO, ORÇAMENTO E GESTÃO "))
        contents.append(html.P("3: CASA CIVIL "))
        contents.append(html.P("4: SECRETARIA DE ESTADO DE OBRAS E INFRAESTRUTURA "))
        contents.append(html.P("5: SECRETARIA DE ESTADO DE EDUCAÇÃO "))
        contents.append(html.P("6: DEFENSORIA PÚBLICA DO DISTRITO FEDERAL "))
        contents.append(html.P("7: SECRETARIA DE ESTADO DE SAÚDE "))
        contents.append(html.P("8: TRIBUNAL DE CONTAS DO DISTRITO FEDERAL "))
        contents.append(html.P("9: SECRETARIA DE ESTADO DE DESENVOLVIMENTO URBANO E HABITAÇÃO "))
        contents.append(html.P("10: PODER LEGISLATIVO "))
        contents.append(html.P("11: SECRETARIA DE ESTADO DE JUSTIÇA E CIDADANIA "))
        contents.append(html.P("12: SECRETARIA DE ESTADO DE TRANSPORTE E MOBILIDADE "))
        contents.append(html.P("13: CONTROLADORIA GERAL DO DISTRITO FEDERAL "))
        contents.append(html.P("14: PODER EXECUTIVO "))
        contents.append(html.P("15: SECRETARIA DE ESTADO DE AGRICULTURA, ABASTECIMENTO E DESENVOLVIMENTO RURAL "))
        contents.append(html.P("16: SECRETARIA DE ESTADO DE ECONOMIA, DESENVOLVIMENTO, INOVAÇÃO, CIÊNCIA E TECNOLOGIA "))
        contents.append(html.P("17: SECRETARIA DE ESTADO DE DESENVOLVIMENTO ECONÔMICO "))
        contents.append(html.P("18: SECRETARIA DE ESTADO DO MEIO AMBIENTE "))

    return contents

def NamedInlineRadioItems(name, short, options, val, **kwargs):
    return html.Div(
        id=f"div-{short}",
        style={"display": "inline-block"},
        children=[
            f"{name}:",
            dcc.RadioItems(
                id=f"radio-{short}",
                options=options,
                value=val,
                labelStyle={"display": "inline-block", "margin-right": "7px"},
                style={"display": "inline-block", "margin-left": "7px"},
            ),
        ],
    )


def create_layout(app):
    # Actual layout of the app
    return html.Div(
        className="row",
        style={"max-width": "100%", "font-size": "1.5rem", "padding": "0px 0px"},
        children=[
            # Header
            html.Div(
                className="row header",
                id="app-header",
                style={"background-color": "#f9f9f9"},
                children=[
                    html.Div(
                        [
                            html.Img(
                                src=app.get_asset_url("knedle-logo4.png"),
                                className="logo",
                                id="plotly-image",
                            )
                        ],
                        className="three columns header_img",
                    ),
                    html.Div(
                        [
                            html.H3(
                                "DODF Explorer",
                                className="header_title",
                                id="app-title",
                            )
                        ],
                        className="nine columns header_title_container",
                    ),
                ],
            ),
            # Demo Description
            html.Div(
                className="row background",
                id="demo-explanation",
                style={"padding": "50px 45px"},
                children=[
                    html.Div(
                        id="description-text", children=dcc.Markdown(DODF_Explorer_intro_md)
                    ),
                    html.Div(
                        html.Button(id="learn-more-button", children=["Learn More"])
                    ),
                ],
            ),
            # Body
            html.Div(
                className="row background",
                style={"padding": "10px"},
                children=[
                    html.Div(
                        className="three columns",
                        children=[
                            Card(
                                [
                                    dcc.Dropdown(
                                        id="dropdown-dataset",
                                        searchable=False,
                                        clearable=False,
                                        options=[
                                            {
                                                "label": "DODF",
                                                "value": "DODF", #se mudar o csv para dodf, mudar aqui
                                            },
                                            {
                                                "label": "DODF - Atos de Aposentadoria",
                                                "value": "DODF_Aposentadoria", #se mudar o csv para dodf, mudar aqui
                                            },
                                        ],
                                        placeholder="Select a dataset",
                                        value="dodf_tsne",
                                    ),
                                    
                                    
                                ]
                            )
                        ],
                    ),
            
                    html.Div(
                        className="six columns",
                        children=[
                            dcc.Graph(id="graph-3d-plot-tsne", style={"height": "98vh"})
                        ],
                    ),
                    html.Div(
                        className="three columns",
                        id="euclidean-distance",
                        children=[
                            Card(
                                style={"padding": "5px"},
                                children=[
                                    html.Div(
                                        id="div-plot-click-message",
                                        style={
                                            "text-align": "center",
                                            "margin-bottom": "7px",
                                            "font-weight": "bold",
                                        },
                                    ),
                                    html.Div(id="div-plot-click-data"), #MUDEI AQUI
                                ],
                            )
                        ],
                    ),
                ],
            ),
        ],
    )

def demo_callbacks(app):
    def generate_figure_TSNE(df):
        figure = px.scatter_3d(df, x='x', y='y', z='z', color = 'int_label', size = 'size', size_max = 10)
        return figure

    # Callback function for the learn-more button
    @app.callback(
        [
            Output("description-text", "children"),
            Output("learn-more-button", "children"),
        ],
        [Input("learn-more-button", "n_clicks")],
    )
    
    def learn_more(n_clicks):
        # If clicked odd times, the instructions will show; else (even times), only the header will show
        if n_clicks is None:
            n_clicks = 0
        if (n_clicks % 2) == 1:
            n_clicks += 1
            return (
                html.Div(
                    style={"padding-right": "15%"},
                    children=[dcc.Markdown(DODF_Explorer_description_md)],
                ),
                "Close",
            )
        else:
            n_clicks += 1
            return (
                html.Div(
                    style={"padding-right": "15%"},
                    children=[dcc.Markdown(DODF_Explorer_intro_md)],
                ),
                "Learn More",
            )

    @app.callback(
        Output("graph-3d-plot-tsne", "figure"),
        [
            Input("dropdown-dataset", "value"),
        ],
    )
    def display_3d_scatter_plot(dataset):
        if dataset:

            
            if dataset == "DODF":
                data = pd.read_csv('dodf_tsne.csv')
                df = pd.DataFrame(data)
            elif dataset == "DODF_Aposentadoria":
                data = pd.read_csv('DODF_Aposentadorias_t-SNE.csv')
                df = pd.DataFrame(data)
            else:
                data = pd.read_csv('dodf_tsne.csv')
                df = pd.DataFrame(data)
                    
            figure = generate_figure_TSNE(df) 

            return figure
    
    @app.callback(
    Output("div-plot-click-data", "children"), [Input("graph-3d-plot-tsne", "clickData"),Input("dropdown-dataset", "value"),]
    )
    def display_nodedata(clickData,dataset):  
         
        if dataset == "DODF_Aposentadoria":
            data = pd.read_csv('DODF_Aposentadorias_t-SNE.csv')
            df = pd.DataFrame(data)  
        else:
            data = pd.read_csv('dodf_tsne.csv')
            df = pd.DataFrame(data)

        contents = "Clique em um ponto para ver o texto do ato aqui"

        if clickData:

            XYZ = {}
            XYZ['x'] = clickData["points"][0]['x']
            XYZ['y'] = clickData["points"][0]['y']
            XYZ['z'] = clickData["points"][0]['z']
            
            achar_indice = (
                df.loc[:, "x":"z"].eq(XYZ).all(axis=1)
            )
            # Retrieve the index of the point clicked, given it is present in the set
            if achar_indice.any():
                clicked_idx = df[achar_indice].index[0] 

                conteudo = df['conteudo'][clicked_idx]
                label = df['labels'][clicked_idx] 

                contents = []
            
                if dataset == "DODF_Aposentadoria":
                    ref_anomes = df['REF_ANOMES'][clicked_idx]
                    data_dodf = df['DATA_DODF'][clicked_idx]
                    num_dodf = df['NUM_DODF'][clicked_idx]
                    pagina_dodf = df['PAGINA_DODF'][clicked_idx]
                    tipo_dodf = df['TIPO_DODF'][clicked_idx]
                    ato = df['ATO'][clicked_idx]
                    empresa_ato = df['EMPRESA_ATO'][clicked_idx]
                    cod_matricula_ato = df['COD_MATRICULA_ATO'][clicked_idx]
                    cod_matricula_sigrh = df['COD_MATRICULA_SIGRH'][clicked_idx]
                    cpf = df['CPF'][clicked_idx]
                    nome_ato = df['NOME_ATO'][clicked_idx]
                    cargo = df['CARGO'][clicked_idx]
                    classe = df['CLASSE'][clicked_idx]
                    padrao = df['PADRAO'][clicked_idx]
                    quadro = df['QUADRO'][clicked_idx]
                    processo = df['PROCESSO'][clicked_idx]
                    fund_legal = df['FUND_LEGAL'][clicked_idx]


                    contents.append(html.H5(label))
                    contents.append(html.P("REF_ANOMES: " + str(ref_anomes)))
                    contents.append(html.P("DATA_DODF: " + str(data_dodf)))
                    contents.append(html.P("NUM_DODF: " + str(num_dodf)))
                    contents.append(html.P("PAGINA_DODF: " + str(pagina_dodf)))
                    contents.append(html.P("TIPO_DODF: " + str(tipo_dodf)))
                    contents.append(html.P("ATO: " + str(ato))) 
                    contents.append(html.P("EMPRESA_ATO: " + str(empresa_ato)))
                    contents.append(html.P("COD_MATRICULA_ATO: " + str(cod_matricula_ato)))
                    contents.append(html.P("COD_MATRICULA_SIGRH: " + str(cod_matricula_sigrh))) 
                    contents.append(html.P("CPF: " + str(cpf))) 
                    contents.append(html.P("NOME_ATO: " + str(nome_ato))) 
                    contents.append(html.P("CARGO: " + str(cargo))) 
                    contents.append(html.P("CLASSE: " + str(classe))) 
                    contents.append(html.P("PADRAO: " + str(padrao))) 
                    contents.append(html.P("QUADRO: " + str(quadro))) 
                    contents.append(html.P("PROCESSO: " + str(processo))) 
                    contents.append(html.P("FUND_LEGAL: " + str(fund_legal)))

                else: 
                    contents.append(html.H5(label))
                    contents.append(html.P(conteudo))

        return contents

    @app.callback(
        Output("div-plot-click-message", "children"),
        [   
            Input("graph-3d-plot-tsne", "clickData"), 
            Input("dropdown-dataset", "value"),
        ],
    )
    def display_click_message(clickData, dataset):
        # Displays message shown when a point in the graph is clicked, depending whether it's an image or word
        if clickData: 
            return "Ato Selecionado"
        else:
            return "Clique em um ponto para visualizar o texto do ato selecionado."