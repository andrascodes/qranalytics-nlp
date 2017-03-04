from predict_conversation_cluster import predict_cluster
from predict_message_sentiment import predict_sentiment

import traceback

from flask import Flask, jsonify, request
app = Flask(__name__)


@app.route('/cluster', methods=['POST'])
def get_cluster():
    
    # try:
    request_data = request.get_json()

    cluster = predict_cluster(request_data['text'])
    response = {
        'clusterId': cluster[0],
        'label': cluster[1]
    }

    return jsonify(response)
    # except Exception as e:
    #     return jsonify(e)

@app.route('/sentiment', methods=['POST'])
def get_sentiment():

    # try:
    request_data = request.get_json()

    sentiment = predict_sentiment(request_data['text'])
    response = {
        'sentiment': sentiment
    }

    return jsonify(response)
    # except Exception as e:
    #     return jsonify({ 'error': str(e) })

@app.errorhandler(Exception)
def unhandled_exception(e):
    response = {
        'error': traceback.format_exc(e)
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run()
    