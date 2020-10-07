import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd


def get_dash(server):
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    app = dash.Dash(__name__, 
                    server=server,
                    routes_pathname_prefix='/dashapp/',
                    external_stylesheets=external_stylesheets
                    )

    dfz = get_data()

    styles = get_styles()

    fig = px.bar(dfz, x="Fruit", y="Amount", color="City", barmode="group")

    app.layout = html.Div([
        # html.H6("Change the value in the text box to see callbacks in action!"),
        html.A("Go to Home Page", href="/", style=styles["button_styles"]),
        html.Div("This graph holds really cool data.", id='my-output',
                 style=styles["text_styles"]),
        html.Div(
            dcc.Graph(
                id='example-graph',
                figure=fig
            ),
            style=styles["fig_style"]
        )
    ])

    return app


# def get_data():
#     df = pd.DataFrame({
#             "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
#             "Amount": [4, 1, 2, 2, 4, 5],
#             "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
#         })
#     return df


# predicted wind_power
def get_data():
    dfz = pd.DataFrame({ "Predicted Power Output"})
    return dfz


# predicticted solar power 
def get_data():
    dfz1 = pd.DataFrame({ "Predicted Power Output"})
    return dfz1


def get_styles():
    """
    Very good for making the thing beautiful.
    """
    base_styles = {
        "text-align": "center",
        "border": "1px solid #ddd",
        "padding": "7px",
        "border-radius": "2px",
    }
    text_styles = {
        "background-color": "#eee",
        "margin": "auto",
        "width": "50%"
    }
    text_styles.update(base_styles)

    button_styles = {
        "text-decoration": "none",
    }
    button_styles.update(base_styles)

    fig_style = {
        "padding": "10px",
        "width": "80%",
        "margin": "auto",
        "margin-top": "5px"
    }
    fig_style.update(base_styles)
    return {
        "text_styles" : text_styles,
        "base_styles" : base_styles,
        "button_styles" : button_styles,
        "fig_style": fig_style,
    }
   
