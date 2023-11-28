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

def baseC(base_path, ano_inicio = '1/3/2022',tipo = 'media'):
    base = pd.read_csv('./dados/{}.csv'.format(base_path))

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
    X = base.drop(columns=['Date','Open','High','Low','Close','Alvo','Retorno'])
    y = base['Alvo']
    y = y.copy()
    y.iloc[-1] = 0

    X_train, X_test = X[base['Date'].dt.date < data_inicio_teste], X[base['Date'].dt.date >= data_inicio_teste]
    y_train, y_test = y[base['Date'].dt.date < data_inicio_teste], y[base['Date'].dt.date >= data_inicio_teste]

    predictions = np.full(len(y_test), fill_value=np.argmax(np.bincount(np.asarray(y_test).astype(int))))

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