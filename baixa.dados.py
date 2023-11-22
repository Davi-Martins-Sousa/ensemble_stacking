import yfinance as yf

# Defina o período de datas desejado
start_date = '2023-01-01'
end_date = '2023-08-01'

# Símbolo da ação (PETR3 no caso da Petrobras)
stock_symbol = 'PETR3.SA'

# Baixa os dados históricos
#stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
stock_data = yf.download(stock_symbol, period="max") # interval="1h" 1m ou 1s


# Salva os dados em um arquivo CSV
csv_filename = 'dados/PETR3.SA.csv'
stock_data.to_csv(csv_filename)

print(stock_data)