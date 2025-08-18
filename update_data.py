import pandas as pd
import requests
from datetime import datetime, timedelta
import os

# --- Configuração ---
CSV_FILE = "cotacao_historica_base.csv"
API_URL_TEMPLATE = "https://www.cepea.org.br/api/indicador/dados/id/2/d/{start_date}/{end_date}"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}

def update_data_file():
    """
    Script para buscar os dados mais recentes da cotação do boi gordo
    e atualizar o ficheiro CSV local.
    """
    print(f"A iniciar a atualização do ficheiro '{CSV_FILE}'...")

    # 1. Carrega os dados existentes
    if not os.path.exists(CSV_FILE):
        print(f"Erro: O ficheiro '{CSV_FILE}' não foi encontrado. Certifique-se de que ele existe na mesma pasta.")
        return

    df = pd.read_csv(CSV_FILE)
    df['Data'] = pd.to_datetime(df['Data'])
    
    # 2. Determina o período para buscar novos dados
    last_date = df['Data'].max()
    start_date = last_date + timedelta(days=1)
    end_date = datetime.now()

    if start_date.date() >= end_date.date():
        print("Os dados já estão atualizados. Nenhuma ação necessária.")
        return

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    print(f"A procurar novos dados de {start_str} até {end_str}...")

    # 3. Busca os novos dados na API
    try:
        api_url = API_URL_TEMPLATE.format(start_date=start_str, end_date=end_str)
        response = requests.get(api_url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        data = response.json()

        if not data or not data.get('result'):
            print("Nenhum dado novo encontrado no período.")
            return

        # 4. Processa e prepara os novos dados
        new_data_df = pd.DataFrame(data['result'])
        new_data_df = new_data_df[['data', 'valor']]
        new_data_df.columns = ['Data', 'Cotação (R$/arroba)']
        new_data_df['Data'] = pd.to_datetime(new_data_df['Data'])
        new_data_df['Cotação (R$/arroba)'] = new_data_df['Cotação (R$/arroba)'].str.replace(',', '.').astype(float)

        print(f"Encontrados {len(new_data_df)} novos registos.")

        # 5. Combina os dados, remove duplicatas e salva o ficheiro atualizado
        combined_df = pd.concat([df, new_data_df]).drop_duplicates(subset=['Data'], keep='last')
        combined_df = combined_df.sort_values("Data").reset_index(drop=True)
        
        combined_df.to_csv(CSV_FILE, index=False)
        print(f"Sucesso! O ficheiro '{CSV_FILE}' foi atualizado com os novos dados.")

    except requests.exceptions.RequestException as e:
        print(f"\nErro de conexão: Não foi possível ligar à API do CEPEA.")
        print(f"Detalhes: {e}")
        print("Por favor, verifique a sua ligação à internet e tente novamente.")
    except Exception as e:
        print(f"\nOcorreu um erro inesperado: {e}")

if __name__ == "__main__":
    update_data_file()
