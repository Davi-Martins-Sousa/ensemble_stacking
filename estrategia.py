import pandas as pd

def create(base):
    base = pd.read_csv(base)
    return base

def start(data):
    print(f'Início do dia: {data}')

def update(data):
    print("Simulação de compra evenda")

def finish(data):
    print(f'Fim do dia: {data}')

def compra(capital, posicao, precoHoje, volume):
    posicao += volume
    capital -= volume * precoHoje - 0.00  # preço de hoje - slippage
    return capital, posicao

def venda(capital, posicao, precoHoje, volume):
    posicao -= volume
    capital += volume * precoHoje + 0.00  # preço de hoje + slippage
    return capital, posicao