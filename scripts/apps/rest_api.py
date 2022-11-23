from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route("/")
def hello_world():
  return "Hello, World!"

@app.route("/api/v0/html_endpoint/name=<string:name>")
def html_endpoint(name):
    
    page = f"""
    <html>
        <body>
        Hi {name}!
        </body>
    </html>
    """

    return page

@app.route("/api/v0/json_endpoint/name=<string:name>")
def json_endpoint(name):

    return jsonify({'name':name})


@app.route('/api/v0/html_endpoint2')
def html_endpoint2():
    # if key doesn't exist, returns None
    name = request.args.get('name')

    page = f"""
    <html>
        <body>
        Hi {name}!
        </body>
    </html>
    """
    return page


@app.route('/api/v1/data_endpoint', methods=['GET', 'POST'])
def data_endpoint():
    # if key doesn't exist, returns None
    json = request.json

    return json

app.run(port=5000,debug=True)