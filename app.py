#Charger les modules
import joblib
import pandas as pd
from flask import Flask, request, jsonify
import json
from sklearn.neighbors import NearestNeighbors
import shap
#Chargement des données
model = joblib.load('classifier_lgbm_model.pkl')
threshold = joblib.load('threshold.joblib')
X_test = joblib.load('X_test.csv')
X_train = joblib.load('X_train.csv')
y_test = joblib.load('y_test.csv')
y_train = joblib.load('y_train.csv')

x_train_sample = X_train.iloc[0: 100]
y_train_sample = y_train.iloc[0: 100]


X = pd.concat([X_test, X_train], axis=0)
y = pd.concat([y_test, y_train], axis=0)
preproc_cols = X_train.columns
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


#/data_client/?SK_ID_CURR=330430
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




#chercher les 10 NN du train_set
def get_df_neigh_10(selected_id_client):
    #fit des NN
    NN = NearestNeighbors(n_neighbors=10)
    NN.fit(X_train)
    X_client = X.loc[selected_id_client: selected_id_client]
    idx = NN.kneighbors(X=X_client,
                        n_neighbors=10,
                        return_distance=False).ravel()
    nearest_client_idx = list(X_train.iloc[idx].index)
    #données et cibles des NN
    x_neigh = X_train.loc[nearest_client_idx, :]
    y_neigh = y_train.loc[nearest_client_idx]
    return x_neigh, y_neigh, X_client

#/neigh_client/?SK_ID_CURR=330430
@app.route('/neigh_client/')
def neigh_client():
    selected_id_client = int(request.args.get('SK_ID_CURR'))
    #selectionner l'id client en requete http et retourner les NN
    data_neigh, y_neigh, X_client = get_df_neigh_10(selected_id_client)
    #convertir en JSON
    data_neigh_json = json.loads(data_neigh.to_json())
    y_neigh_json = json.loads(y_neigh.to_json())
    x_customer_json = json.loads(X_client.to_json())
    #retourner le flux en json
    return jsonify({'status': 'pass',
                    'y_neigh':  y_neigh_json,
                    'data_neigh': data_neigh_json,
                    'X_client': x_customer_json})




@app.route('/shap_values/')
#shap values du client et de ses 10 nearest neighbors
#/shap_values/?SK_ID_CURR=330430
def shap_value():
    #selectionner l'id client en requete http
    selected_id_client = int(request.args.get('SK_ID_CURR'))
    #recuperer le NN
    X_neigh, y_neigh = get_df_neigh_10(selected_id_client)
    X_client = X.loc[selected_id_client: selected_id_client]
    #preparer les valeurs shap des NN + client
    shap.initjs()
    #creation du TreeExplainer avec le model
    explainer = shap.TreeExplainer(model)
    #valeurs
    expected_vals = pd.Series(list(explainer.expected_value))
    #calcule des valeurs shap du client
    shap_vals_client = pd.Series(list(explainer.shap_values(X_client)[1]))
    #calcule des valeurs shap des NN
    shap_val_neigh_ = pd.Series(list(explainer.shap_values(X_neigh)[1]))
    #pd.Series en JSON
    X_neigh_json = json.loads(X_neigh.to_json())
    expected_vals_json = json.loads(expected_vals.to_json())
    shap_val_neigh_json = json.loads(shap_val_neigh_.to_json())
    shap_vals_client_json = json.loads(shap_vals_client.to_json())
    #retourner le flux en json
    return jsonify({'status': 'pass',
                    'X_neigh_': X_neigh_json,
                    'shap_val_neigh': shap_val_neigh_json,
                    'expected_vals': expected_vals_json,
                    'shap_val_client': shap_vals_client_json})


#affichage donnée
@app.route('/all_proc_train_data/')
def all_proc_train_data():
    #recuperer toutes les données de X_train et y_train
    #retourner le flux en json
    X_train_json = json.loads(x_train_sample.to_json())
    y_train_json = json.loads(y_train_sample.to_json())
    return jsonify({'status': 'pass',
                    'X_train': X_train_json,
                    'y_train': y_train_json})


#score client 
#/score_du_client/?SK_ID_CURR=330430
@app.route('/score_du_client/')
def score_client():
    #selectionner l'id client en requete http
    client_id = int(request.args.get('SK_ID_CURR'))
    #recuperer les données du client
    X_client = X.loc[client_id:client_id]
    #calcul du score client
    score = model.predict_proba(X_client)[:, 1][0]
    return jsonify({'status': 'pass',
                    'SK_ID_CURR': client_id,
                    'score': score,
                    'thresh': threshold})


#features du model lgbm
@app.route('/feature/')
def features():
    feat = X_test.columns
    f = pd.Series(feat)
    feat_json = json.loads(f.to_json())
    return jsonify({'status': 'pass',
                    'data': feat_json})


#feature importance du model lgbm
@app.route('/feature_importance/')
def send_feat_imp():
    feat_imp = pd.Series(model.feature_importances_,
                         index=X_test.columns).sort_values(ascending=False)
    feat_imp_json = json.loads(feat_imp.to_json())
    return jsonify({'status': 'pass',
                    'data': feat_imp_json})




if __name__ == "__main__":
    app.run()

