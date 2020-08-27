import io
import pathlib

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

import pandas as pd
import plotly.express as px


# get relative data folder
PATH = pathlib.Path(__file__).parent

data_dict = {
    "DODF2D": pd.read_csv('DODF_tsne_2D.csv'),
    "DODF3D": pd.read_csv('dodf_tsne.csv'),
    "DODF_Aposentadoria2D": pd.read_csv('DODF_Aposentadorias_tsne_2D.csv'),
    "DODF_Aposentadoria3D": pd.read_csv('DODF_Aposentadorias_tsne_3D.csv'),
    "DODF_Editais2D": pd.read_csv('DODF_Editais_tsne_2D_v2.csv'),
    "DODF_Editais3D": pd.read_csv('DODF_Editais_tsne_3D.csv'),
    "DODF_Exoneracoes2D": pd.read_csv('DODF_Exoneracoes_tsne_2D_v2.csv'),
    "DODF_Exoneracoes3D": pd.read_csv('DODF_Exoneracoes_tsne_3D.csv'),
}

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
                style={"background-color": "#1F2132"},
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
                style={"padding": "50px 45px 0px 45px"},
                children=[
                    html.Div(id="description-text", children=[
                        html.H4(
                            # className="three columns",
                            style={"text-align": "center"},
                            children=["About us"],
                        ),
                        html.H6("The scatter plot below is the result of running the t-SNE algorithm on DODF's datasets, resulting in 2D and 3D visualizations of the documents."),
                        html.H6("Official publications such as the Diario Oficial do Distrito Federal (DODF) are sources of information on all official government acts. Although these documents are rich in knowledge, analysing these texts manually by specialists is a complex and unfeasible task considering the growing volume of documents, the result of the frequent number of publications in the Distrito Federal Government's (GDF) communication vehicle."),
                        html.H6("DODF Explorer aims to facilitate the visualization of such information using unsupervised machine learning methods and data visualization techniques. This is one of the tools developed by the KnEDLe Project. To learn more about us, click on 'Learn More' below.")
                    ]),
                    html.Div(
                        html.Button(id="learn-more-button", children=[
                            html.A("Learn More", href='https://unb-knedle.github.io/', target="_blank")
                            ])
                    ),
                    html.Hr(),
                ],
            ),

            html.Div(
                className="row background",
                id="menu-huge",
                children=[
                    html.H4(
                        # className="three columns",
                        style={"text-align": "center"},
                        children=["Explore our datasets!"],
                    ),
                    # Body
                    html.Div(
                        className="row background",
                        style={"padding": "5px 5px 0px 0px"},
                        children=[
                            html.Div(
                                className="six columns",
                                children=[
                                    html.Div(
                                        className="row background",
                                        id="menu",
                                        style={"padding": "5px 20px 0 0",},
                                        children=[
                                            html.Div(
                                                className="three columns",
                                                style={"display": "grid", #Esse elemento alinha os itens em grade
                                                    "grid-template-columns": "repeat(3, auto)", 
                                                    "place-items": "start",
                                                },
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
                                                                        "label": "DODF - Aposentadoria",
                                                                        "value": "DODF_Aposentadoria", 
                                                                    },
                                                                    {
                                                                        "label": "DODF - Editais",
                                                                        "value": "DODF_Editais", 
                                                                    },
                                                                    {
                                                                        "label": "DODF - Exoneracoes",
                                                                        "value": "DODF_Exoneracoes", 
                                                                    },
                                                                ],
                                                                placeholder="Select a dataset",
                                                                value="DODF",
                                                            )
                                                        ]
                                                    ),
                                                    Card(
                                                        [
                                                            dcc.Dropdown(
                                                                id="dropdown-dimension",
                                                                searchable=False,
                                                                clearable=False,
                                                                options=[
                                                                    {
                                                                        "label": "Bidimensional (2D)",
                                                                        "value": "2D", #se mudar o csv para dodf, mudar aqui
                                                                    },
                                                                    {
                                                                        "label": "Tridimensional (3D)",
                                                                        "value": "3D", #se mudar o csv para dodf, mudar aqui
                                                                    },
                                                                ],
                                                                placeholder="Select a dimensionality",
                                                                value="2D",
                                                            ),
                                                        ]
                                                    ),
                                                    
                                                    #html.Button(id="tutorial-button", children=["Need help?"])
                                                ],
                                            ),
                                        ],
                                    ),
                                    
                                    dcc.Graph(id="graph-3d-plot-tsne", style={"height": "92vh"})
                                ],
                            ),
                            html.Div(
                                className="six columns",
                                id="circos-control-tabs",
                                children=[
                                    dcc.Tabs(id='circos-tabs', value='what-is', children=[                                       

                                        dcc.Tab(
                                            label='Dataset',
                                            value='data',
                                            children=html.Div(className='control-tab', children=[
                                                Card(
                                                    style={"padding": "20px",  "align-content": "center", "text-align": "center"},
                                                    children=[
                                                        html.Div(id="dataset"),
                                                    ],
                                                )
                                            ])
                                        ),

                                        dcc.Tab(
                                            label='Labels',
                                            value='table',
                                            children=html.Div(className='control-tab', children=[
                                                Card(
                                                    style={"padding": "5px"},
                                                    children=[
                                                        html.Div(id="legenda"),
                                                    ],
                                                )
                                            ])
                                        ),

                                        dcc.Tab(
                                            label='Selected point',
                                            value='graph',
                                            children=html.Div(className='control-tab', children=[
                                                Card(
                                                    style={"padding": "5px"},
                                                    children=[
                                                        html.Div(id="div-plot-click-data"),
                                                    ],
                                                )
                                            ]),
                                        ),

                                    ])
                                ]),       
                            
                        ],
                    ),
                ]
            )
        ],
    )

def demo_callbacks(app):
    def generate_figure_TSNE(df, dimension):
        if dimension == '2D':
            figure = px.scatter(df, x='x', y='y', color = 'int_label', size = 'size', size_max = 5)
            
        
        elif dimension == '3D':
            figure = px.scatter_3d(df, x='x', y='y', z='z', color = 'int_label', size = 'size', size_max = 10)
            figure.update_traces(marker=dict(size=5,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
        return figure

    @app.callback(
        Output("legenda", "children"),
        [
            Input("dropdown-dataset", "value"),
        ],
    )
    def display_legenda(dataset):
        contents = []
        contents.append(html.H5("Labels"))
        if dataset == 'DODF':
            contents.append(html.Ol(
                children=[
                    (html.Li("SECRETARIA DE ESTADO DE SEGURANÇA PÚBLICA ")),
                    (html.Li("SECRETARIA DE ESTADO DE CULTURA ")),
                    (html.Li("SECRETARIA DE ESTADO DE FAZENDA, PLANEJAMENTO, ORÇAMENTO E GESTÃO ")),
                    (html.Li("CASA CIVIL ")),
                    (html.Li("SECRETARIA DE ESTADO DE OBRAS E INFRAESTRUTURA ")),
                    (html.Li("SECRETARIA DE ESTADO DE EDUCAÇÃO ")),
                    (html.Li("DEFENSORIA PÚBLICA DO DISTRITO FEDERAL ")),
                    (html.Li("SECRETARIA DE ESTADO DE SAÚDE ")),
                    (html.Li("TRIBUNAL DE CONTAS DO DISTRITO FEDERAL ")),
                    (html.Li("SECRETARIA DE ESTADO DE DESENVOLVIMENTO URBANO E HABITAÇÃO ")),
                    (html.Li("PODER LEGISLATIVO ")),
                    (html.Li("SECRETARIA DE ESTADO DE JUSTIÇA E CIDADANIA ")),
                    (html.Li("SECRETARIA DE ESTADO DE TRANSPORTE E MOBILIDADE ")),
                    (html.Li("CONTROLADORIA GERAL DO DISTRITO FEDERAL ")),
                    (html.Li("PODER EXECUTIVO ")),
                    (html.Li("SECRETARIA DE ESTADO DE AGRICULTURA, ABASTECIMENTO E DESENVOLVIMENTO RURAL ")),
                    (html.Li("SECRETARIA DE ESTADO DE ECONOMIA, DESENVOLVIMENTO, INOVAÇÃO, CIÊNCIA E TECNOLOGIA ")),
                    (html.Li("SECRETARIA DE ESTADO DE DESENVOLVIMENTO ECONÔMICO ")),
                    (html.Li("SECRETARIA DE ESTADO DO MEIO AMBIENTE ")),
                ],
                start="0",
            ))
        elif dataset =='DODF_Aposentadoria':
            contents.append(html.Ol(
                children=[
                    (html.Li("ADMINISTRACAO REGIONAL DE CEILANDIA")),
                    (html.Li("ADMINISTRACAO REGIONAL DE AGUAS CLARAS")),
                    (html.Li("ADMINISTRACAO REGIONAL DE BRAZLANDIA")),
                    (html.Li("ADMINISTRACAO REGIONAL DO GAMA")),
                    (html.Li("ADMINISTRACAO REGIONAL DE PLANALTINA")),
                    (html.Li("ADMINISTRACAO REGIONAL DE SAMAMBAIA")),
                    (html.Li("ADMINISTRACAO REGIONAL DE SANTA MARIA")),
                    (html.Li("ADMINISTRACAO REGIONAL DE SOBRADINHO")),
                    (html.Li("ADMINISTRACAO REGIONAL DE TAGUATINGA")),
                    (html.Li("ADMINISTRACAO REGIONAL DO CRUZEIRO")),
                    (html.Li("ADMINISTRACAO REGIONAL DO GUARA")),
                    (html.Li("ADMINISTRACAO REGIONAL DO LAGO NORTE")),
                    (html.Li("ADMINISTRACAO REGIONAL DO LAGO SUL")),
                    (html.Li("ADMINISTRACAO REGIONAL DO NUCLEO BANDEIRANTE")),
                    (html.Li("ADMINISTRACAO REGIONAL DO PLANO PILOTO")),
                    (html.Li("ADMINISTRACAO REGIONAL DO RIACHO FUNDO I")),
                    (html.Li("AGENCIA DE FISCALIZACAO DO DISTRITO FEDERAL")),
                    (html.Li("ARQUIVO PUBLICO DO DISTRITO FEDERAL")),
                    (html.Li("CONTROLADORIA GERAL DO DISTRITO FEDERAL")),
                    (html.Li("CORPO DE BOMBEIROS MILITAR DO DISTRITO FEDERAL")),
                    (html.Li("DEFENSORIA PUBLICA DO DISTRITO FEDERAL")),
                    (html.Li("DEPARTAMENTO DE ESTRADAS DE RODAGEM DO DISTRITO FEDERAL")),
                    (html.Li("DEPARTAMENTO DE TRANSITO DO DISTRITO FEDERAL")),
                    (html.Li("FUNDACAO HEMOCENTRO DE BRASILIA")),
                    (html.Li("FUNDACAO JARDIM ZOOLOGICO DE BRASILIA")),
                    (html.Li("INSTITUTO DE PREVIDENCIA DOS SERVIDORES DO DISTRITO FEDERAL")),
                    (html.Li("INSTITUTO DO MEIO AMBIENTE E DOS RECURSOS HIDRICOS DO DISTRITO FEDERAL")),
                    (html.Li("POLICIA CIVIL DO DISTRITO FEDERAL")),
                    (html.Li("POLICIA MILITAR DO DISTRITO FEDERAL")),
                    (html.Li("PROCURADORIA GERAL DO DISTRITO FEDERAL")),
                    (html.Li("SECRETARIA DE ESTADO DA AGRICULTURA, ABASTECIMENTO E DESENVOLVIMENTO RURAL DO DISTRITO FEDERAL")),
                    (html.Li("SECRETARIA DE ESTADO DA CASA CIVIL, RELACOES INSTITUCIONAIS E SOCIAIS DO DISTRITO FEDERAL")),
                    (html.Li("SECRETARIA DE ESTADO DA SEGURANCA PUBLICA E DA PAZ SOCIAL DO DISTRITO FEDERAL")),
                    (html.Li("SECRETARIA DE ESTADO DE CULTURA DO DISTRITO FEDERAL")),
                    (html.Li("SECRETARIA DE ESTADO DE DESENVOLVIMENTO ECONOMICO DO DISTRITO FEDERAL")),
                    (html.Li("SECRETARIA DE ESTADO DE DESENVOLVIMENTO SOCIAL DO DISTRITO FEDERAL")),
                    (html.Li("SECRETARIA DE ESTADO DE DESENVOLVIMENTO URBANO E HABITACAO DO DISTRITO FEDERAL")),
                    (html.Li("SECRETARIA DE ESTADO DE ECONOMIA DO DISTRITO FEDERAL")),
                    (html.Li("SECRETARIA DE ESTADO DE ECONOMIA, DESENVOLVIMENTO, INOVACAO, CIENCIA E TECNOLOGIA DO DISTRITO FEDERAL")),
                    (html.Li("SECRETARIA DE ESTADO DE EDUCACAO")),
                    (html.Li("SECRETARIA DE ESTADO DE ESPORTE E LAZER DO DISTRITO FEDERAL")),
                    (html.Li("SECRETARIA DE ESTADO DE FAZENDA DO DISTRITO FEDERAL")),
                    (html.Li("SECRETARIA DE ESTADO DE FAZENDA, PLANEJAMENTO, ORCAMENTO E GESTAO DO DISTRITO FEDERAL")),
                    (html.Li("SECRETARIA DE ESTADO DE GESTAO DO TERRITORIO E HABITACAO DO DISTRITO FEDERAL")),
                    (html.Li("SECRETARIA DE ESTADO DE INFRAESTRUTURA E SERVICOS PUBLICOS DO DISTRITO FEDERAL")),
                    (html.Li("SECRETARIA DE ESTADO DE JUSTICA E CIDADANIA DO DISTRITO FEDERAL")),
                    (html.Li("SECRETARIA DE ESTADO DE MEIO AMBIENTE")),
                    (html.Li("SECRETARIA DE ESTADO DE MOBILIDADE DO DISTRITO FEDERAL")),
                    (html.Li("SECRETARIA DE ESTADO DE OBRAS E INFRAESTRUTURA DO DISTRITO FEDERAL")),
                    (html.Li("SECRETARIA DE ESTADO DE PLANEJAMENTO, ORÇAMENTO E GESTAO DO DISTRITO FEDERAL")),
                    (html.Li("SECRETARIA DE ESTADO DE POLITICAS PARA CRIANCAS, ADOLESCENTES E JUVENTUDE DO DISTRITO FEDERAL")),
                    (html.Li("SECRETARIA DE ESTADO DE PROTECAO DA ORDEM URBANISTICA DO DISTRITO FEDERAL")),
                    (html.Li("SECRETARIA DE ESTADO DE SAUDE")),
                    (html.Li("SECRETARIA DE ESTADO DE SECRETARIA DE ESTADO DE PLANEJAMENTO, ORCAMENTO E GESTAO DO DISTRITO FEDERAL")),
                    (html.Li("SECRETARIA DE ESTADO DE SEGURANCA PUBLICA E DA PAZ SOCIAL DO DISTRITO FEDERAL")),
                    (html.Li("SECRETARIA DE ESTADO DE TRABALHO, DESENVOLVIMENTO SOCIAL, MULHERES, IGUALDADE RACIAL E DIREITOS HUMANOS DO DISTRITO FEDERAL")),
                    (html.Li("SECRETARIA DE ESTADO DE TRANSPORTE E MOBILIDADE DO DISTRITO FEDERAL")),
                    (html.Li("SECRETARIA DE ESTADO DO DESENVOLVIMENTO URBANO E HABITACAO DO DISTRITO FEDERAL")),
                    (html.Li("SECRETARIA DE ESTADO DO TRABALHO, DESENVOLVIMENTO SOCIAL, MULHERES, IGUALDADE RACIAL E DIREITOS HUMANOS DO DISTRITO FEDERAL")),
                    (html.Li("SERVICO DE LIMPEZA URBANA DO DISTRITO FEDERAL")),
                    (html.Li("TRIBUNAL DE CONTAS DO DISTRITO FEDERAL")),
                    (html.Li("VICE GOVERNADORIA")),
                ],
                start="0",
            ))

        elif dataset == 'DODF_Editais':
            contents.append(html.Ol(
                children=[
                    (html.Li("SERVIÇOS DE ENGENHARIA (OUTROS)")),
                    (html.Li("AQUISIÇÃO (OUTROS)")),
                    (html.Li("OBRA DE ENGENHARIA (OUTROS)")),
                    (html.Li("SERVIÇOS (OUTROS)")),
                    (html.Li("LOCAÇÃO DE EQUIPAMENTOS DE TI / SOFTWARE")),
                    (html.Li("SERVIÇOS DE TECNOLOGIA DA INFORMAÇÃO")),
                    (html.Li("SERVIÇOS DE CONSULTORIA")),
                    (html.Li("AQUISIÇÃO DE VEÍCULOS")),
                    (html.Li("AQUISIÇÃO DE MEDICAMENTOS E PRODUTOS HOSPITALARES")),
                    (html.Li("SERVIÇOS DE MANUTENÇÃO (MÁQ/EQUIPAMENTOS)")),
                    (html.Li("SERVIÇOS DE MANUTENÇÃO (VEÍCULOS)")),
                    (html.Li("AQUISIÇÃO DE EQUIPAMENTOS TI / SOFTWARE")),
                    (html.Li("LOCAÇÃO DE VEÍCULOS")),
                    (html.Li("SERVIÇOS DE CONSERVAÇÃO (MAO DE OBRA)")),
                    (html.Li("SERVIÇOS DE MANUTENÇÃO (PREDIAL)")),
                    (html.Li("SERVIÇOS DE VIGILÂNCIA")),
                    (html.Li("OBRA VIÁRIA")),
                    (html.Li("OBRA CONSTRUÇÃO PREDIAL")),
                    (html.Li("LOCAÇÃO (OUTROS)")),
                    (html.Li("OBRA PAVIMENTAÇÃO")),
                    (html.Li("OBRA DE SANEMENTO")),
                    (html.Li("NaN")),
                    
                ],
                start="0",
            ))
            

        elif dataset == 'DODF_Exoneracoes':
            contents.append(html.Ol(
                children=[
                    (html.Li("ADMINISTRACAO REGIONAL DA CANDANGOLANDIA")),
                    (html.Li("ADMINISTRACAO REGIONAL DA FERCAL")),
                    (html.Li("ADMINISTRACAO REGIONAL DE AGUAS CLARAS")),
                    (html.Li("ADMINISTRACAO REGIONAL DE BRAZLANDIA")),
                    (html.Li("ADMINISTRACAO REGIONAL DE CEILANDIA")),
                    (html.Li("ADMINISTRACAO REGIONAL DE PLANALTINA")),
                    (html.Li("ADMINISTRACAO REGIONAL DE SAMAMBAIA")),
                    (html.Li("ADMINISTRACAO REGIONAL DE SANTA MARIA")),
                    (html.Li("ADMINISTRACAO REGIONAL DE SAO SEBASTIAO")),
                    (html.Li("ADMINISTRACAO REGIONAL DE SOBRADINHO")),
                    (html.Li("ADMINISTRACAO REGIONAL DE SOBRADINHO II")),
                    (html.Li("ADMINISTRACAO REGIONAL DE TAGUATINGA")),
                    (html.Li("ADMINISTRACAO REGIONAL DE VICENTE PIRES")),
                    (html.Li("ADMINISTRACAO REGIONAL DO CRUZEIRO")),
                    (html.Li("ADMINISTRACAO REGIONAL DO GAMA")),
                    (html.Li("ADMINISTRACAO REGIONAL DO GUARA")),
                    (html.Li("ADMINISTRACAO REGIONAL DO ITAPOA")),
                    (html.Li("ADMINISTRACAO REGIONAL DO JARDIM BOTANICO")),
                    (html.Li("ADMINISTRACAO REGIONAL DO LAGO NORTE")),
                    (html.Li("ADMINISTRACAO REGIONAL DO LAGO SUL")),
                    (html.Li("ADMINISTRACAO REGIONAL DO NUCLEO BANDEIRANTE")),
                    (html.Li("ADMINISTRACAO REGIONAL DO PARANOA")),
                    (html.Li("ADMINISTRACAO REGIONAL DO PARK WAY")),
                    (html.Li("ADMINISTRACAO REGIONAL DO PLANO PILOTO")),
                    (html.Li("ADMINISTRACAO REGIONAL DO RECANTO DAS EMAS")),
                    (html.Li("ADMINISTRACAO REGIONAL DO RIACHO FUNDO I")),
                    (html.Li("ADMINISTRACAO REGIONAL DO RIACHO FUNDO II")),
                    (html.Li("ADMINISTRACAO REGIONAL DO SETOR COMPL. DE INDUSTRIA E ABASTECIMENTO")),
                    (html.Li("ADMINISTRACAO REGIONAL DO SETOR DE INDUSTRIA E ABASTECIMENTO")),
                    (html.Li("ADMINISTRACAO REGIONAL DO SUDOESTE/OCTOGONAL")),
                    (html.Li("ADMINISTRACAO REGIONAL DO VARJAO")),
                    (html.Li("AGENCIA DE FISCALIZACAO DO DISTRITO FEDERAL - AGEFIS")),
                    (html.Li("AGENCIA REGULADORA DE AGUAS, ENERGIA E SANEAMENTO BASICO DO DF")),
                    (html.Li("ARQUIVO PUBLICO DO DISTRITO FEDERAL")),
                    (html.Li("CASA CIVIL DO DF")),
                    (html.Li("CASA CIVIL DO DISTRITO FEDERAL")),
                    (html.Li("COMPANHIA DE DESENVOLVIMENTO HABITACIONAL DO DISTRITO FEDERAL")),
                    (html.Li("COMPANHIA DE PLANEJAMENTO DO DISTRITO FEDERAL - CODEPLAN")),
                    (html.Li("COMPANHIA DO METROPOLITANO DO DISTRITO FEDERAL - METRO-DF")),
                    (html.Li("COMPANHIA URBANIZADORA DA NOVA CAPITAL DO BRASIL - NOVACAP")),
                    (html.Li("CONTROLADORIA GERAL DO DISTRITO FEDERAL")),
                    (html.Li("CORPO DE BOMBEIROS MILITAR DO DISTRITO FEDERAL")),
                    (html.Li("DEFENSORIA PUBLICA DO DISTRITO FEDERAL")),
                    (html.Li("DEPARTAMENTO DE ESTRADAS DE RODAGEM - DER")),
                    (html.Li("DEPARTAMENTO DE TRANSITO - DETRAN")),
                    (html.Li("EMPRESA BRASILIENSE DE TURISMO - BRASILIATUR")),
                    (html.Li("EMPRESA DE ASSISTENCIA TECNICA E EXTENSAO RURAL - EMATER")),
                    (html.Li("FUNDACAO DE AMPARO AO TRABALHADOR PRESO - FUNAP")),
                    (html.Li("FUNDACAO DE APOIO A PESQUISA - FAP")),
                    (html.Li("FUNDACAO DE ENSINO E PESQUISA EM CIENCIAS DA SAUDE - FEPECS")),
                    (html.Li("FUNDACAO HEMOCENTRO DE BRASILIA - FHB")),
                    (html.Li("FUNDACAO JARDIM ZOOLOGICO DE BRASILIA")),
                    (html.Li("FUNDACAO UNIVERSIDADE ABERTA DO DISTRITO FEDERAL - FUNAB")),
                    (html.Li("GABINETE DO VICE-GOVERNADOR")),
                    (html.Li("INSTITUTO DE ASSISTENCIA A SAUDE DOS SERVIDORES DO DISTRITO FEDERAL-INAS")),
                    (html.Li("INSTITUTO DE DEFESA DO CONSUMIDOR DO DISTRITO FEDERAL-PROCON-DF")),
                    (html.Li("INSTITUTO DE PREVIDENCIA DOS SERVIDORES DO DISTRITO FEDERAL - IPREV/DF")),
                    (html.Li("INSTITUTO DO MEIO AMBIENTE E DOS REC.HIDRICOS DO DF - BSB AMBIENTAL")),
                    (html.Li("JARDIM BOTANICO DE BRASILIA")),
                    (html.Li("JUNTA COMERCIAL, INDUSTRIAL E SERVICOS DO DF - JUCIS/DF")),
                    (html.Li("POLICIA CIVIL DO DISTRITO FEDERAL")),
                    (html.Li("POLICIA MILITAR DO DISTRITO FEDERAL")),
                    (html.Li("PROCURADORIA GERAL DO DISTRITO FEDERAL")),
                    (html.Li("SECRETARIA DE ESTADO DA MICRO E PEQUENA EMPRESA E ECONOMIA SOLIDARIA DF")),
                    (html.Li("SECRETARIA DE ESTADO DA MULHER DO DISTRITO FEDERAL")),
                    (html.Li("SECRETARIA DE ESTADO DA ORDEM PUBLICA E SOCIAL DO DISTRITO FEDERAL")),
                    (html.Li("SECRETARIA DE ESTADO DAS CIDADES")),
                    (html.Li("SECRETARIA DE ESTADO DE AGRICULTURA, ABASTEC. E DESENVOLVIMENTO RURAL")),
                    (html.Li("SECRETARIA DE ESTADO DE ASSUNTOS ESTRATEGICOS DO DISTRITO FEDERAL")),
                    (html.Li("SECRETARIA DE ESTADO DE CIENCIA, TECNOLOGIA E INOVACAO DO DF")),
                    (html.Li("SECRETARIA DE ESTADO DE COMUNICACAO DO DISTRITO FEDERAL")),
                    (html.Li("SECRETARIA DE ESTADO DE COMUNICAGCO SOCIAL DO DISTRITO FEDERAL")),
                    (html.Li("SECRETARIA DE ESTADO DE CULTURA E ECONOMIA CRIATIVA DO DISTRITO FEDERAL")),
                    (html.Li("SECRETARIA DE ESTADO DE DESENVOLVIMENTO ECONOMICO DO DF")),
                    (html.Li("SECRETARIA DE ESTADO DE DESENVOLVIMENTO HUMANO E SOCIAL")),
                    (html.Li("SECRETARIA DE ESTADO DE DESENVOLVIMENTO SOCIAL DO DISTRITO FEDERAL")),
                    (html.Li("SECRETARIA DE ESTADO DE DESENVOLVIMENTO URBANO E HABITACAO DO DF")),
                    (html.Li("SECRETARIA DE ESTADO DE ECONOMIA DO DISTRITO FEDERAL")),
                    (html.Li("SECRETARIA DE ESTADO DE EDUCACAO")),
                    (html.Li("SECRETARIA DE ESTADO DE GESTAO ADMINISTRATIVA E DESBUROCRATIZACAO")),
                    (html.Li("SECRETARIA DE ESTADO DE JUSTICA E CIDADANIA")),
                    (html.Li("SECRETARIA DE ESTADO DE MEIO AMBIENTE")),
                    (html.Li("SECRETARIA DE ESTADO DE OBRAS E INFRAESTRUTURA DO DISTRITO FEDERAL")),
                    (html.Li("SECRETARIA DE ESTADO DE PLANEJAMENTO, ORCAMENTO E GESTAO")),
                    (html.Li("SECRETARIA DE ESTADO DE POLITICAS CRIANCAS, ADOLESCENTES E JUVENTUDE")),
                    (html.Li("SECRETARIA DE ESTADO DE POLITICAS MULHERES IGUALDADE RACIAL DIR HUMANOS")),
                    (html.Li("SECRETARIA DE ESTADO DE PROTECAO DA ORDEM URBANISTICA DO DF - DF LEGAL")),
                    (html.Li("SECRETARIA DE ESTADO DE PROTECAO E DEFESA CIVIL DO DISTRITO FEDERAL")),
                    (html.Li("SECRETARIA DE ESTADO DE PUBLICIDADE INSTITUCIONAL E COMUNICACAO SOCIAL")),
                    (html.Li("SECRETARIA DE ESTADO DE REGULARIZACAO DE CONDOMINIOS DO DISTRITO FEDERAL")),
                    (html.Li("SECRETARIA DE ESTADO DE RELACOES INSTITUCIONAIS E SOCIAIS")),
                    (html.Li("SECRETARIA DE ESTADO DE SAUDE")),
                    (html.Li("SECRETARIA DE ESTADO DE SEGURANCA PUBLICA DO DISTRITO FEDERAL")),
                    (html.Li("SECRETARIA DE ESTADO DE TRABALHO")),
                    (html.Li("SECRETARIA DE ESTADO DE TRANSPORTE E MOBILIDADE DO DISTRITO FEDERAL")),
                    (html.Li("SECRETARIA DE ESTADO DE TURISMO DO DISTRITO FEDERAL")),
                    (html.Li("SECRETARIA DE ESTADO DO ESPORTE E LAZER DO DISTRITO FEDERAL")),
                    (html.Li("SECRETARIA DE ESTADO EXTRAORDINARIA DA COPA 2014")),
                    (html.Li("SOCIEDADE DE TRANSPORTES COLETIVOS DE BRASILIA LTDA.")),
                    (html.Li("TRANSPORTE URBANO DO DISTRITO FEDERAL - DFTRANS")),
                    (html.Li("VICE-GOVERNADORIA")),
                ],
                start="0",
            ))

        return contents

    @app.callback(
        Output("dataset", "children"),
        [
            Input("dropdown-dataset", "value"),
        ],
    )
    def display_dataset(dataset):
        contents = []
        if dataset == 'DODF':
            contents.append(html.H3("DODF"))
            contents.append(html.P("Este conjunto de dados possui 717 instâncias, as quais representam atos que foram retirados do DODF. Cada instância possui dois atributos, que são o conteúdo e o rótulo do ato. Os rótulos foram determinados pelos órgãos responsáveis por cada ato."))
            contents.append(html.P("Os pontos no gráfico ao lado representam as instâncias deste conjunto de dados. A cor de cada ponto foi determinada de acordo com o rótulo de cada instância."))

        elif dataset =='DODF_Aposentadoria':
            contents.append(html.H3("DODF - Aposentadoria"))
            contents.append(html.P("Este conjunto de dados possui 5516 instâncias, as quais representam os dados de atos de aposentadoria publicados no DODF em 2018 e 2019. Cada instância possui 17 atributos."))
            contents.append(html.P("Os pontos no gráfico ao lado representam as instâncias deste conjunto de dados. A cor de cada ponto foi determinada de acordo com o atributo 'EMPRESA_ATO' de cada instância, o qual se refere à empresa que publicou determinado ato."))

        elif dataset == 'DODF_Editais':
            contents.append(html.H3("DODF - Editais"))
            contents.append(html.P("Este conjunto de dados possui 13872 instâncias, as quais representam os dados de editais que foram publicados no DODF entre os anos de 2013 e 2020. Cada instância possui 20 atributos."))
            contents.append(html.P("Os pontos no gráfico ao lado representam as instâncias deste conjunto de dados. A cor de cada ponto foi determinada de acordo com o atributo 'classifObjeto' de cada instância, o qual se refere ao objeto do edital referenciado por determinada instância."))
            
        elif dataset == 'DODF_Exoneracoes':
            contents.append(html.H3("DODF - Exoneracoes"))
            contents.append(html.P("Este conjunto de dados possui 45530 instâncias, as quais representam os dados de atos de exoneração que foram publicados no DODF entre os anos de 2010 e 2020. Cada instância possui 15 atributos."))
            contents.append(html.P("Os pontos no gráfico ao lado representam as instâncias deste conjunto de dados. A cor de cada ponto foi determinada de acordo com o atributo '06_ORGAO' de cada instância, o qual se refere ao órgão que publicou determinado ato ."))
            
        return contents



    @app.callback(
        Output("graph-3d-plot-tsne", "figure"),
        [
            Input("dropdown-dataset", "value"),Input("dropdown-dimension", "value")
        ],
    )
    def display_3d_scatter_plot(dataset, dimension):
        if dataset:

            
            if dataset == "DODF" and dimension == '2D':
                data = data_dict['DODF2D']
                df = pd.DataFrame(data)
            elif dataset == "DODF_Aposentadoria" and dimension == '2D':
                data = data_dict['DODF_Aposentadoria2D']
                df = pd.DataFrame(data)
            elif dataset == "DODF_Aposentadoria" and dimension == '3D':
                data = data_dict['DODF_Aposentadoria3D']
                df = pd.DataFrame(data)
            elif dataset == "DODF_Editais" and dimension == '2D':
                data = data_dict['DODF_Editais2D']
                df = pd.DataFrame(data)
            elif dataset == "DODF_Editais" and dimension == '3D':
                data = data_dict['DODF_Editais3D']
                df = pd.DataFrame(data)
            elif dataset == "DODF_Exoneracoes" and dimension == '2D':
                data = data_dict['DODF_Exoneracoes2D']
                df = pd.DataFrame(data)
            elif dataset == "DODF_Exoneracoes" and dimension == '3D':
                data = data_dict['DODF_Exoneracoes3D']
                df = pd.DataFrame(data)
            else:
                data = data_dict['DODF3D']
                df = pd.DataFrame(data)

                
            figure = generate_figure_TSNE(df, dimension) 

            return figure
    
    @app.callback(
    Output("div-plot-click-data", "children"), [Input("graph-3d-plot-tsne", "clickData"),Input("dropdown-dataset", "value"),Input("dropdown-dimension", "value"),]
    )
    def display_nodedata(clickData,dataset,dimension):  

        if dataset == "DODF" and dimension == '2D':
            data = data_dict['DODF2D']
            df = pd.DataFrame(data)
        elif dataset == "DODF_Aposentadoria" and dimension == '3D':
            data = data_dict['DODF_Aposentadoria3D']
            df = pd.DataFrame(data)  
        elif dataset == "DODF_Aposentadoria" and dimension == '2D':
            data = data_dict['DODF_Aposentadoria2D']
            df = pd.DataFrame(data)  
        elif dataset == "DODF_Editais" and dimension == '3D':
            data = data_dict['DODF_Editais3D']
            df = pd.DataFrame(data)  
        elif dataset == "DODF_Editais" and dimension == '2D':
            data = data_dict['DODF_Editais2D']
            df = pd.DataFrame(data)  
        elif dataset == "DODF_Exoneracoes" and dimension == '3D':
            data = data_dict['DODF_Exoneracoes3D']
            df = pd.DataFrame(data)  
        elif dataset == "DODF_Exoneracoes" and dimension == '2D':
            data = data_dict['DODF_Exoneracoes2D']
            df = pd.DataFrame(data)  
        else:
            data = data_dict['DODF3D']
            df = pd.DataFrame(data)

        contents = []
        
        contents.append(
            html.Div(
                children=[
                    html.H5("Clique em um ponto para visualizar o texto do ato selecionado."),

                    html.Div(
                        style={"width": "100%", "height": "80vh", "display": "grid", "place-items": "center"},
                        children=[
                            html.Img(
                                    src=app.get_asset_url("knedle-logo4.png"),
                                    style={"height": "100px", "opacity": "0.5"},
                                )
                        ]
                    )
                ]
            )
        )

        if clickData:
            if dimension == '2D':
                XY = {}
                XY['x'] = clickData["points"][0]['x']
                XY['y'] = clickData["points"][0]['y']
                achar_indice = (df.loc[:, "x":"y"].eq(XY).all(axis=1))
            
            elif dimension == '3D':
                XYZ = {}
                XYZ['x'] = clickData["points"][0]['x']
                XYZ['y'] = clickData["points"][0]['y']
                XYZ['z'] = clickData["points"][0]['z']
                achar_indice = (df.loc[:, "x":"z"].eq(XYZ).all(axis=1))
            
            # Retrieve the index of the point clicked, given it is present in the set
            if achar_indice.any():
                clicked_idx = df[achar_indice].index[0]  
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
                    contents.append(
                        html.Ul(
                            style={"list-style-type": "none",
                            "margin-bottom": "1px"},
                            className="list",
                            children=[
                                (html.Li("REF_ANOMES: " + str(ref_anomes))),
                                (html.Li("DATA_DODF: " + str(data_dodf))),
                                (html.Li("NUM_DODF: " + str(num_dodf))),
                                (html.Li("PAGINA_DODF: " + str(pagina_dodf))),
                                (html.Li("TIPO_DODF: " + str(tipo_dodf))),
                                (html.Li("ATO: " + str(ato))) ,
                                (html.Li("EMPRESA_ATO: " + str(empresa_ato))),
                                (html.Li("COD_MATRICULA_ATO: " + str(cod_matricula_ato))),
                                (html.Li("COD_MATRICULA_SIGRH: " + str(cod_matricula_sigrh))) ,
                                (html.Li("CPF: " + str(cpf))) ,
                                (html.Li("NOME_ATO: " + str(nome_ato))) ,
                                (html.Li("CARGO: " + str(cargo))) ,
                                (html.Li("CLASSE: " + str(classe))) ,
                                (html.Li("PADRAO: " + str(padrao))) ,
                                (html.Li("QUADRO: " + str(quadro))) ,
                                (html.Li("PROCESSO: " + str(processo))) ,
                                (html.Li("FUND_LEGAL: " + str(fund_legal))),
                            ]
                        )
                    )

                elif dataset == "DODF_Editais":
                    idEditais = df['idEditais'][clicked_idx]
                    autuado = df['autuado'][clicked_idx]
                    nrEdital = df['nrEdital'][clicked_idx]
                    anoEdital = df['anoEdital'][clicked_idx]
                    siglaLicitante = df['siglaLicitante'][clicked_idx]
                    nomeLicitante = df['nomeLicitante'][clicked_idx]
                    dtPublicacao = df['dtPublicacao'][clicked_idx]
                    nrDodf = df['nrDodf'][clicked_idx]
                    anoDodf = df['anoDodf'][clicked_idx]
                    dtAbertura = df['dtAbertura'][clicked_idx]
                    modalidadeLicitacao = df['modalidadeLicitacao'][clicked_idx]
                    vrEstimado = df['vrEstimado'][clicked_idx]
                    prazo = df['prazo'][clicked_idx]
                    prazoAbertura = df['prazoAbertura'][clicked_idx]
                    tpPrazo = df['tpPrazo'][clicked_idx]
                    ementaObj = df['ementaObj'][clicked_idx]
                    descObjeto = df['descObjeto'][clicked_idx]
                    nrgdf = df['nrgdf'][clicked_idx]
                    anogdf = df['anogdf'][clicked_idx]
                    classifObjeto = df['classifObjeto'][clicked_idx]


                    contents.append(html.H5(label))
                    contents.append(
                        html.Ul(
                            style={"list-style-type": "none",
                            "margin-bottom": "1px"},
                            className="list",
                            children=[
                                (html.Li("idEditais: " + str(idEditais))),
                                (html.Li("autuado: " + str(autuado))),
                                (html.Li("nrEdital: " + str(nrEdital))),
                                (html.Li("anoEdital: " + str(anoEdital))),
                                (html.Li("siglaLicitante: " + str(siglaLicitante))),
                                (html.Li("nomeLicitante: " + str(nomeLicitante))) ,
                                (html.Li("dtPublicacao: " + str(dtPublicacao))),
                                (html.Li("nrDodf: " + str(nrDodf))),
                                (html.Li("anoDodf: " + str(anoDodf))) ,
                                (html.Li("dtAbertura: " + str(dtAbertura))) ,
                                (html.Li("modalidadeLicitacao: " + str(modalidadeLicitacao))) ,
                                (html.Li("vrEstimado: " + str(vrEstimado))) ,
                                (html.Li("prazo: " + str(prazo))) ,
                                (html.Li("prazoAbertura: " + str(prazoAbertura))) ,
                                (html.Li("tpPrazo: " + str(tpPrazo))) ,
                                (html.Li("ementaObj: " + str(ementaObj))) ,
                                (html.Li("descObjeto: " + str(descObjeto))),
                                (html.Li("nrgdf: " + str(nrgdf))) ,
                                (html.Li("anogdf: " + str(anogdf))) ,
                                (html.Li("classifObjeto: " + str(classifObjeto))),
                            ]
                        )
                    )
                    

                elif dataset == "DODF_Exoneracoes":
                    NOME_DO_SERVIDOR = df['01_NOME_DO_SERVIDOR'][clicked_idx]
                    MATRICULA = df['02_MATRICULA'][clicked_idx]
                    CARGO_COMISSAO_SIMBOLO = df['03_CARGO_COMISSAO_SIMBOLO'][clicked_idx]
                    CARGO_COMISSAO = df['04_CARGO_COMISSAO'][clicked_idx]
                    LOTACAO = df['05_LOTACAO'][clicked_idx]
                    LOTACAO_SUPERIOR_1 = df['05_LOTACAO_SUPERIOR_1'][clicked_idx]
                    LOTACAO_SUPERIOR_2 = df['05_LOTACAO_SUPERIOR_2'][clicked_idx]
                    LOTACAO_SUPERIOR_3 = df['05_LOTACAO_SUPERIOR_3'][clicked_idx]
                    LOTACAO_SUPERIOR_4 = df['05_LOTACAO_SUPERIOR_4'][clicked_idx]
                    ORGAO = df['06_ORGAO'][clicked_idx]
                    VIGENCIA = df['07_VIGENCIA'][clicked_idx]
                    A_PEDIDO = df['08_A_PEDIDO'][clicked_idx]
                    CARGO_EFETIVO = df['09_CARGO_EFETIVO'][clicked_idx]
                    CARGO_EFETIVO_REFERENCIA = df['09_CARGO_EFETIVO_REFERENCIA'][clicked_idx]
                    MATRICULA_SIAPE = df['10_MATRICULA_SIAPE'][clicked_idx]
                    MOTIVO = df['11_MOTIVO'][clicked_idx]


                    contents.append(html.H5(label))
                    contents.append(
                        html.Ul(
                            style={"list-style-type": "none",
                            "margin-bottom": "1px"},
                            className="list",
                            children=[
                                (html.Li("NOME_DO_SERVIDOR: " + str(NOME_DO_SERVIDOR))),
                                (html.Li("MATRICULA: " + str(MATRICULA))),
                                (html.Li("CARGO_COMISSAO_SIMBOLO: " + str(CARGO_COMISSAO_SIMBOLO))),
                                (html.Li("CARGO_COMISSAO: " + str(CARGO_COMISSAO))),
                                (html.Li("LOTACAO_SUPERIOR_1: " + str(LOTACAO_SUPERIOR_1))),
                                (html.Li("LOTACAO_SUPERIOR_2: " + str(LOTACAO_SUPERIOR_2))) ,
                                (html.Li("LOTACAO_SUPERIOR_3: " + str(LOTACAO_SUPERIOR_3))),
                                (html.Li("LOTACAO_SUPERIOR_4: " + str(LOTACAO_SUPERIOR_4))),
                                (html.Li("ORGAO: " + str(ORGAO))) ,
                                (html.Li("VIGENCIA: " + str(VIGENCIA))) ,
                                (html.Li("A_PEDIDO: " + str(A_PEDIDO))) ,
                                (html.Li("CARGO_EFETIVO: " + str(CARGO_EFETIVO))) ,
                                (html.Li("CARGO_EFETIVO_REFERENCIA: " + str(CARGO_EFETIVO_REFERENCIA))) ,
                                (html.Li("MATRICULA_SIAPE: " + str(MATRICULA_SIAPE))) ,
                                (html.Li("MOTIVO: " + str(MOTIVO))) ,
                            ]
                        )
                    )

                else: 
                    
                    conteudo = df['conteudo'][clicked_idx]

                    contents.append(html.H5(label))
                    contents.append(html.P(conteudo))

            else:
                contents = []
                contents.append(
                    html.Div(
                        children=[
                            html.H5("Clique em um ponto para visualizar o texto do ato selecionado."),

                            html.Div(
                                style={"width": "100%", "height": "80vh", "display": "grid", "place-items": "center"},
                                children=[
                                    html.Img(
                                            src=app.get_asset_url("knedle-logo4.png"),
                                            style={"height": "100px", "opacity": "0.5"},
                                        )
                                ]
                            )
                        ]
                    )
                )

        return contents
