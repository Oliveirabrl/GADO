import pandas as pd
import requests
from datetime import datetime, timedelta
import firebase_admin
from firebase_admin import credentials, firestore
import os
import json

# --- Configuração ---
COLLECTION_NAME = "cotacoes"
API_URL_TEMPLATE = "https://www.cepea.org.br/api/indicador/dados/id/2/d/{start_date}/{end_date}"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}

def initialize_firestore():
    """Inicializa a conexão com o Firestore usando as credenciais."""
    creds_json_str = os.getenv('FIREBASE_CREDENTIALS_JSON')
    if not creds_json_str:
        raise ValueError("A variável de ambiente FIREBASE_CREDENTIALS_JSON não está definida.")
    
    creds_dict = json.loads(creds_json_str)
    cred = credentials.Certificate(creds_dict)
    firebase_admin.initialize_app(cred)
    return firestore.client()

def update_firestore_data():
    """Busca os dados mais recentes da API e os salva no Firestore."""
    print("A iniciar a atualização de dados para o Firestore...")
    db = initialize_firestore()
    
    try:
        query = db.collection(COLLECTION_NAME).order_by("Data", direction=firestore.Query.DESCENDING).limit(1)
        last_doc = next(query.stream(), None)
        
        if last_doc:
            last_date = last_doc.to_dict()['Data'].date()
            start_date = last_date
        else:
            start_date = datetime.now().date() - timedelta(days=3*365)
        
        end_date = datetime.now().date()
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        print(f"A procurar novos dados de {start_str} até {end_str}...")

        api_url = API_URL_TEMPLATE.format(start_date=start_str, end_date=end_str)
        response = requests.get(api_url, headers=HEADERS, timeout=20)
        response.raise_for_status()
        data = response.json()

        if not data or not data.get('result'):
            print("Nenhum dado novo encontrado no período.")
            return

        new_records_count = 0
        for record in data['result']:
            record_date_str = record['data'].split(" ")[0]
            doc_ref = db.collection(COLLECTION_NAME).document(record_date_str)
            
            record_datetime = datetime.strptime(record['data'], "%Y-%m-%d %H:%M:%S")
            record_price = float(record['valor'].replace(',', '.'))
            
            doc_ref.set({
                'Data': record_datetime,
                'Cotação (R$/arroba)': record_price
            })
            new_records_count += 1
        
        print(f"Sucesso! {new_records_count} registos foram adicionados/atualizados no Firestore.")

    except Exception as e:
        print(f"\nOcorreu um erro durante a atualização: {e}")
        raise

if __name__ == "__main__":
    update_firestore_data()
