from flask import Flask, jsonify, request
import sys
from sklearn.externals import joblib
import pandas as pd
import traceback
import os

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hey welcome to the Titanic'

@app.route('/predict', methods=['POST'])
def predict():
    if lr:
        try:
            json_ = request.get_json()
            print(json_)

            query = pd.get_dummies(pd.DataFrame(json_))
            print(query)
            query = query.reindex(columns=model_columns, fill_value=0)
            print(query)

            prediction = list(lr.predict(query))

            return jsonify({'prediction': str(prediction)})

        except:
            return jsonify({'trace': traceback.format_exc()})

    else:
        print('Train the model first')
        return ('No model here to use')


if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except:
        port = 12345

    print(os.getcwd())

    lr = joblib.load("model.pkl")
    print('Model Loaded!')

    model_columns = joblib.load("model_columns.pkl")
    print('Model Columns Loaded!')

    app.run(port=port, debug=True)
