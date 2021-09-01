import dash

import dash_core_components as dcc
import dash_html_components as html



import pathlib
from utils import Header

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../data").resolve()




def create_layout(app):
    return html.Div(
        [
            Header(app),
            # page 6


 html.Div(
        children=[
        html.Div(
            [
                #html.Div(
                    #[
             html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            [

                                                html.Div(
                                                    [
                                                        #html.H6("Проект века"),
                                                        html.H5("Cement quality prediction and optimization"),
                                                        #html.H6("Все права  защищены"),
                                                    ],
                                                    className="page-1b",
                                                ),
                                            ],
                                            className="page-1c",
                                        )
                                    ],
                                    className="page-1d",
                                ),
                                html.Div(
                                    [
                                        html.H1(
                                            [
                                                html.Span("01.", className="page-1e"),
                                                html.Span("20"),
                                            ]
                                        ),
                                        html.H6(""),
                                    ],
                                    className="page-1f",
                                ),
                            ],
                            className="page-1g",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H6("Igor Konnov", className="page-1h"),
                                        html.P("453-264-8591"),
                                        html.P("ilq@w.ipq"),
                                    ],
                                    className="page-1i",
                                ),



                            ],
                            className="page-1j",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H6(
                                            "Introduction",
                                            className="page-1h",
                                        ),
                                        html.P( "The cement industry is very rich in data. Hundreds of parameters are collected every hour to monitor and control cement quality and operations. Most of the data collected is utilized to fulfill local standards. In rare cases, these data are also used to predict cement performance and quality."),
                                        html.P( "In real life, it is difficult to predict cement quality. This is due to the high variability of the physical and chemical characteristics of raw materials as well as different operation modes. Furthermore, there are currently no low cost tools available for quality prediction."),
                                        html.P("Machine learning (ML) provides various opportunities for cement producers, including quality optimization, cost optimization, throughput prediction, production cost monitoring, and so on. "),
                                        html.P("The idea of this study is to demonstrate how modern ML can be applied in the cement industry. The study also seeks to examine how these tools can help to optimize production cost. ")

                                    ],
                                    className="page-1k",
                                ),

                                html.Div([html.H6("Key findings", className = "page-1h"),
                                          html.P(" This ML application can optimize cement production cost and quality. This project demonstrates that a cement plant can save  0.5 – 2 $ per ton of cement, optimizing minimum parameters: gypsum, SCM, mill output, and dosage of cement additive. Is possible to predict cement quality with high accuracy despite a lack of data.")
                                         ],className="page-1k"),

                                html.Div(
                                    [
                                        html.H6(
                                            "Objective",
                                            className="page-1h",
                                        ),
                                        html.P("This study presents 3 primary objectives:"),
                                        html.P("1.To create and select a machine learning (ML) model with the ability to predict 2 days’ worth of compressive strength"),
                                        html.P("2.To determine the performance of the ML and select quality predictors "),
                                        html.P("3.To optimize cement composition to determine the lowest possible cost of the cement produced"),
                                    ],
                                    className="page-1m",
                                ),
                                html.Div(
                                    [
                                        html.H6(
                                            "Source of data",
                                            className="page-1h",
                                        ),
                                        html.P("The data are collected from an existing decommissioned cement plant."),
                                    ],
                                    className="page-1l",
                                ),
                                html.Div([    html.H6("Background information", className="page-1h" ),
                                       html.P("The cement plant produced only OP cement with 5% of limestone in an open ball mill. The plant suffered high production costs due to growing competition and pure cement reactivity - early ages (2D compressive strength)."),
                                       html.P("The mean of 2 days of compressive strength produced by the plant was 21 MPa. However, the average 2D compressive strength of cement produced by competitors in the market was approximately 24-27 MPa."),
                                       html.P("The plants required a cement additive with the ability to enhance early compressive strength. Thus, several quality improvers were tested. Several challenges were presented, such as:"),
                                       html.P("• how to assess the performance of the tested cement additives."),
                                       html.P("• what cement parameter was most important and what to control to get the desired strength."),
                                       html.P("• what the expected production cost was at the desired quality. "),
                                       html.P("• how to optimize the production cost. ")


                                           ], className="page-1l",),
                            ],
                            className="page-1n",
                        ),
                    #],
                    #className="subpage",
                #)
            ],
            className="page",
        ) ,


                 ])
],className="page",)
