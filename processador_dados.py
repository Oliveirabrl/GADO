import pandas as pd
import os
from datetime import datetime

# --- Configuração ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DADOS_GERAL_PATH = os.path.join(BASE_DIR, 'dados_historicos', 'dados_geral')
OUTPUT_FILE_PATH = os.path.join(BASE_DIR, 'dados_historicos', 'arroba-sp-historico.csv')

# Mapeamento dos ficheiros .xlsx e das colunas que queremos ler
FILES_TO_PROCESS = {
    'preco_brl_arroba': 'preco_brl_arroba.xlsx',
    'preco_bezerro_brl': 'preco_bezerro_brl.xlsx',
    'preco_soja_brl': 'preco_soja_brl.xlsx',
    'preco_milho_brl': 'preco_milho_brl.xlsx',
}

def process_files():
    """
    Lê os ficheiros .xlsx, processa os dados e gera o CSV final.
    """
    print("Iniciando o processamento dos dados históricos...")
    
    all_data = []
    
    for key, filename in FILES_TO_PROCESS.items():
        file_path = os.path.join(DADOS_GERAL_PATH, filename)
        print(f"Processando {filename}...")
        
        if not os.path.exists(file_path):
            print(f"AVISO: O ficheiro {filename} não foi encontrado.")
            continue
            
        try:
            # Lê o ficheiro .xlsx, ignorando as primeiras 3 linhas de cabeçalho
            df = pd.read_excel(file_path, skiprows=3, engine='openpyxl')
            
            # Valida se as colunas necessárias existem
            required_cols = ['Data', 'À vista R$']
            if not all(col in df.columns for col in required_cols):
                print(f"ERRO: O ficheiro {filename} não contém as colunas 'Data' e 'À vista R$'.")
                continue

            # Seleciona, renomeia e formata os dados
            df = df[required_cols].copy()
            df.rename(columns={'Data': 'data', 'À vista R$': key}, inplace=True)
            
            # --- CORREÇÃO DEFINITIVA: Lê as datas no formato Dia/Mês/Ano ---
            df['data'] = pd.to_datetime(df['data'], dayfirst=True)
            
            # Filtra para ter apenas o primeiro dia de cada mês, a partir de 2015
            df = df[df['data'].dt.year >= 2015]
            df = df.sort_values('data').groupby(pd.Grouper(key='data', freq='MS')).first().reset_index()
            df[key] = pd.to_numeric(df[key], errors='coerce')
            
            all_data.append(df)

        except Exception as e:
            print(f"Ocorreu um erro inesperado ao processar o ficheiro {filename}: {e}")

    if len(all_data) < len(FILES_TO_PROCESS):
        print("Alguns ficheiros não puderam ser processados. O script será encerrado para evitar dados incompletos.")
        return

    # Junta todos os dados num único dataframe
    final_df = all_data[0]
    for df_single in all_data[1:]:
        final_df = pd.merge(final_df, df_single, on='data', how='outer')

    final_df.sort_values('data', inplace=True)
    
    # Preenche valores em falta com o último valor válido
    final_df.ffill(inplace=True)
    final_df.dropna(inplace=True)

    # --- Cálculos das Relações de Troca ---
    print("Calculando relações de troca e custos...")
    final_df['Boi com 20@/Garrote'] = (20 * final_df['preco_brl_arroba']) / final_df['preco_bezerro_brl']
    final_df['Boi com 20@/soja'] = (20 * final_df['preco_brl_arroba']) / final_df['preco_soja_brl']
    final_df['Boi com 20@/milho'] = (20 * final_df['preco_brl_arroba']) / final_df['preco_milho_brl']
    final_df['milho(80)+soja(20)'] = (final_df['preco_milho_brl'] * 26.7) + (final_df['preco_soja_brl'] * 5.6)
    
    # Formata a data para o padrão dia/mês/ano
    final_df['data'] = final_df['data'].dt.strftime('%d/%m/%Y')

    # Grava o ficheiro CSV final
    final_df.to_csv(OUTPUT_FILE_PATH, sep=';', index=False, decimal='.')
    
    print("-" * 50)
    print(f"Processamento concluído com sucesso!")
    print(f"O ficheiro '{os.path.basename(OUTPUT_FILE_PATH)}' foi atualizado com {len(final_df)} linhas.")
    print("-" * 50)

if __name__ == "__main__":
    process_files()

