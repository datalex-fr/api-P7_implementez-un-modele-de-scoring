#Charger les modules
import joblib
import pandas as pd
from flask import Flask, request, jsonify
import json
from sklearn.neighbors import NearestNeighbors
import shap
#Chargement des données
model = joblib.load('classifier_lgbm_model.pkl')
threshold = joblib.load('threshold.pkl')
X_test = joblib.load('X_test.csv')
X_train = joblib.load('X_train.csv')
y_test = joblib.load('y_test.csv')
y_train = joblib.load('y_train.csv')

x_train_sample = X_train.iloc[0: 100]
y_train_sample = y_train.iloc[0: 100]


X = pd.concat([X_test, X_train], axis=0)
y = pd.concat([y_test, y_train], axis=0)

#Instance flask
app = Flask(__name__)

@app.route("/")
def index():
    return "Hello world!"


#Id liste client
@app.route('/id/')
def ids_list():
    id_list = pd.Series(list(X.index.sort_values()))
    id_list_json = json.loads(id_list.to_json())
    return jsonify({'status': 'pass',
                    'data': id_list_json})


#/data_client/?SK_ID_CURR=165690
@app.route('/data_client/')
def selected_client_data():
    selected_id_client = int(request.args.get('SK_ID_CURR'))
    x_client = X.loc[selected_id_client: selected_id_client]
    y_client = y.loc[selected_id_client: selected_id_client]
    data_x_json = json.loads(x_client.to_json())
    y_client_json = json.loads(y_client.to_json())
    return jsonify({'status': 'pass',
                    'y_client': y_client_json,
                    'data': data_x_json})
