import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from estrategia import create,start,finish,compra,venda

def update(capital,posicao,precoHoje,decisao,ultimoDia):
    if ultimoDia == False :
        if decisao > 0:
            capital, posicao = compra(capital, posicao, precoHoje, 1)
        else:
            capital, posicao = venda(capital, posicao, precoHoje, 1)
    else:
        if posicao < 0:
            capital, posicao = compra(capital, posicao, precoHoje, -posicao)
        elif posicao > 0:
            capital, posicao = venda(capital, posicao, precoHoje, posicao)
    return capital, posicao

def ensemble(base_path, ano_inicio = '1/3/2022', tipo = 'votação'):
    base = pd.read_csv('./dados/{}-indicadores.csv'.format(base_path))

    # Remover linhas com valores NaN
    base = base.dropna()

    # Preparar os retorno e o alvo
    base['Retorno'] = base['Close'].shift(-1) - base['Close']
    base['Alvo'] = base['Retorno'].apply(lambda retorno: 1 if retorno > 0 else 0)

    # Define o inicio de teste
    base['Date'] = pd.to_datetime(base['Date'])
    data_inicio_teste_str = ano_inicio #'1/3/2022'
    data_inicio_teste = pd.to_datetime(data_inicio_teste_str).date() 
    indice_data_inicio = base[base['Date'].dt.date == data_inicio_teste].index[0]

   # Dividir os dados em treinamento e teste
    X = base.drop(columns=['Date','Open','High','Low','Close','Volume','Alvo','Retorno'])
    y = base['Alvo']

    X_train, X_test = X[base['Date'].dt.date < data_inicio_teste], X[base['Date'].dt.date >= data_inicio_teste]
    y_train, y_test = y[base['Date'].dt.date < data_inicio_teste], y[base['Date'].dt.date >= data_inicio_teste]

    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    mlp_model = MLPClassifier(random_state=42)
    mlp_model.fit(X_train, y_train)

    xgb_model = XGBClassifier(random_state=42)
    xgb_model.fit(X_train, y_train)

    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)

    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train, y_train)

    if tipo == 'votação':
        
        rf_predictions = rf_model.predict(X_test)
        mlp_predictions = mlp_model.predict(X_test)
        xgb_predictions = xgb_model.predict(X_test)
        nb_predictions = nb_model.predict(X_test)
        knn_predictions = knn_model.predict(X_test)

        predictions = np.array([])

        for i in range(len(rf_predictions)):
            all_predictions = [rf_predictions[i], mlp_predictions[i], xgb_predictions[i], nb_predictions[i], knn_predictions[i]]
            final_prediction = np.bincount(all_predictions).argmax()
            predictions = np.append(predictions, final_prediction)
        
    elif tipo == 'media' or 'media ponderada':
        rf_predictions = rf_model.predict_proba(X_test)[:, 1]
        mlp_predictions = mlp_model.predict_proba(X_test)[:, 1]
        xgb_predictions = xgb_model.predict_proba(X_test)[:, 1]
        nb_predictions = nb_model.predict_proba(X_test)[:, 1]
        knn_predictions = knn_model.predict_proba(X_test)[:, 1]

        peso = [0.2, 0.2, 0.2, 0.2, 0.2]

        if tipo == 'media ponderada':
            rf_auc = roc_auc_score(y_test, rf_predictions)
            mlp_auc = roc_auc_score(y_test, mlp_predictions)
            xgb_auc = roc_auc_score(y_test, xgb_predictions)
            nb_auc = roc_auc_score(y_test, nb_predictions)
            knn_auc = roc_auc_score(y_test, knn_predictions)

            auc = rf_auc + mlp_auc + xgb_auc + nb_auc + knn_auc

            peso = [rf_auc/auc, mlp_auc/auc, xgb_auc/auc, nb_auc/auc, knn_auc/auc]

        predictions = np.array([])
        predictions = (rf_predictions * peso[0] + mlp_predictions * peso[1] + xgb_predictions * peso[2] + nb_predictions * peso[3] + knn_predictions * peso[4])
        threshold = 0.5
        predictions = (predictions >= threshold).astype(int)


    auc = roc_auc_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    print("Recall: {:.4f}\tPrecisão: {:.4f}\tAUC: {:.4f}".format(recall, precision, auc))

        
    capital = 0
    posicao = 0
    fechamentos = []
    riqueza = []

    for indice, hoje in base[base.index >= indice_data_inicio].iterrows():
            
        capital, posicao = update(capital, posicao, hoje['Close'], predictions[indice-indice_data_inicio],indice == base.index[-1])
        riquezaAtual = capital + posicao * hoje['Close']
        fechamentos.append(hoje['Close'])
        riqueza.append(float(riquezaAtual))
        preco = hoje['Close']

    print(f'Capital: {round(capital, 2)}\tAções em posse: {posicao}\tPreço: {round(preco, 2)}\tRiqueza: {round(riquezaAtual, 2)}')
        
    return fechamentos, riqueza

ensemble('PETR3.SA', tipo = 'votação')
ensemble('PETR3.SA', tipo = 'media')
ensemble('PETR3.SA', tipo = 'media ponderada') 