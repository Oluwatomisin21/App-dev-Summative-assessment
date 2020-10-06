import flask
import dashboard

from MESSAGING_API import Clientv

from  WIND_FARM_MODEL import dfz

from  SOLAR_FARM_MODEL import dfz1

print(dfz)

print(dfz1)


server = flask.Flask(__name__)
app = dashboard.get_dash(server)

@app.server.route("/")
def index():
    # go back to home page
    return flask.render_template("index.html")


@app.server.route("/call_back_1")
def call_back_1():
    print("call back used.")

    # go back to home page
    return flask.redirect("/")


@app.server.route("/predict")
def predict():
    print("dfz")
    return flask.redirect("/")

@app.server.route("/predict")
def predict():
    print("dfz1")
    return flask.redirect("/")


@app.server.route("/store_file")
def store_file():
    #storing the file here
    return flask.redirect("/")


if __name__ == '__main__':
    app.run_server(port=5001, debug=True)
