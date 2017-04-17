from predict_conversation_cluster import predict_cluster
from predict_message_sentiment import predict_sentiment

import traceback

from flask import Flask, jsonify, request
app = Flask(__name__)


@app.route('/cluster', methods=['POST'])
def get_cluster():

    request_data = request.get_json()
    text = request_data['text']
    if isinstance(text, basestring):
        cluster = predict_cluster(text)
        response = {
            'clusterId': cluster[0],
            'label': cluster[1]
        }

        return jsonify(response)
    else:
        return jsonify({
            'error': 'Not a string value'
        })

@app.route('/sentiment', methods=['POST'])
def get_sentiment():

    request_data = request.get_json()
    text = request_data['text']

    if isinstance(text, basestring):
        sentiment = predict_sentiment(text)
        response = {
            'sentiment': sentiment
        }

        return jsonify(response)
    else:
        return jsonify({
            'error': 'Not a string value'
        })

@app.errorhandler(Exception)
def unhandled_exception(e):
    response = {
        'error': traceback.format_exc(e)
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(port=5555)
    