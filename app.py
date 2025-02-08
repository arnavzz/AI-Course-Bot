from flask import Flask
app = Flask(__name__)

@app.route('/chat')
def chat():
    return 'Hello, Chat!'

if __name__ == "__main__":
    app.run(debug=True, port=5000)
