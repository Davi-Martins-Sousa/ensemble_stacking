import matplotlib.pyplot as plt
from estrategiaClassificação import classificação
from estrategiaRegressão import regressao 

def main():
    print('\nAlgoritmo classificação baseline')
    fechamentos,riqueza = classificação('PETR3.SA', tipo =  'baseline')
    plt.figure(figsize=(12, 6))
    plt.plot(fechamentos, label='Fechamento', marker='')
    plt.plot(riqueza, label='Riqueza', marker='')
    plt.xlabel('Dia')
    plt.ylabel('Valor')
    plt.title('Evolução do Fechamento e Riqueza ao longo do tempo')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('graficos/grafico_algoritmo_classificação_baseline.png', format='png')