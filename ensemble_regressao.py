import pandas as pd
import numpy as np
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import  mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split
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

def ensembleR(base_path, ano_inicio = '1/3/2022', tipo = 'media'):
    base = pd.read_csv('./dados/{}-indicadores.csv'.format(base_path))

    # Remover linhas com valores NaN
    base = base.dropna()

    # Preparar os retorno e o alvo
    base['Retorno'] = base['Close'].shift(-1) - base['Close']

    # Define o inicio de teste
    base['Date'] = pd.to_datetime(base['Date'])
    data_inicio_teste_str = ano_inicio #'1/3/2022'
    data_inicio_teste = pd.to_datetime(data_inicio_teste_str).date() 
    indice_data_inicio = base[base['Date'].dt.date == data_inicio_teste].index[0]

   # Dividir os dados em treinamento e teste
    X = base.drop(columns=['Date','Open','High','Low','Close','Volume','Retorno'])
    y = base['Retorno']
    y = y.copy()
    y.iloc[-1] = 0

    X_train, X_test = X[base['Date'].dt.date < data_inicio_teste], X[base['Date'].dt.date >= data_inicio_teste]
    y_train, y_test = y[base['Date'].dt.date < data_inicio_teste], y[base['Date'].dt.date >= data_inicio_teste]
    
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)

    mlp_model = MLPRegressor(random_state=42)
    mlp_model.fit(X_train, y_train)
    mlp_predictions = mlp_model.predict(X_test)

    xgb_model = XGBRegressor(random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_predictions = xgb_model.predict(X_test)

    nb_model = LinearRegression()
    nb_model.fit(X_train, y_train)
    nb_predictions = nb_model.predict(X_test)

    knn_model = KNeighborsRegressor()
    knn_model.fit(X_train, y_train)
    knn_predictions = knn_model.predict(X_test)

    peso = [0.2, 0.2, 0.2, 0.2, 0.2]

    if tipo == 'media ponderada':
        X_train_train, X_train_test, y_train_train, y_train_test = train_test_split(X_train, y_train, test_size=(1 - 0.8), shuffle=False)
        
        rf_model = RandomForestRegressor(random_state=42)
        rf_model.fit(X_train_train, y_train_train)
        rf_predictions_aux = rf_model.predict(X_train_test)

        mlp_model = MLPRegressor(random_state=42)
        mlp_model.fit(X_train_train, y_train_train)
        mlp_predictions_aux = mlp_model.predict(X_train_test)

        xgb_model = XGBRegressor(random_state=42)
        xgb_model.fit(X_train_train, y_train_train)
        xgb_predictions_aux = xgb_model.predict(X_train_test)

        nb_model = LinearRegression()
        nb_model.fit(X_train_train, y_train_train)
        nb_predictions_aux = nb_model.predict(X_train_test)

        knn_model = KNeighborsRegressor()
        knn_model.fit(X_train_train, y_train_train)
        knn_predictions_aux = knn_model.predict(X_train_test)

        rf_mae = mean_absolute_error(y_train_test, rf_predictions_aux)
        mlp_mae = mean_absolute_error(y_train_test, mlp_predictions_aux)
        xgb_mae = mean_absolute_error(y_train_test, xgb_predictions_aux)
        nb_mae = mean_absolute_error(y_train_test, nb_predictions_aux)
        knn_mae = mean_absolute_error(y_train_test, knn_predictions_aux)

        mae = rf_mae + mlp_mae + xgb_mae + nb_mae + knn_mae

        peso = [rf_mae/mae, mlp_mae/mae, xgb_mae/mae, nb_mae/mae, knn_mae/mae]
        print(peso)

    predictions = np.array([])
    predictions = (rf_predictions * peso[0] + mlp_predictions * peso[1] + xgb_predictions * peso[2] + nb_predictions * peso[3] + knn_predictions * peso[4])
    threshold = 0.5
    predictions = (predictions >= threshold).astype(int)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = math.sqrt(mse)
    print("MAE: {:.4f}\tRMSE: {:.4f}".format(mae, rmse))

        
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

#ensembleR('PETR3.SA', tipo = 'media')
#ensembleR('PETR3.SA', tipo = 'media ponderada')