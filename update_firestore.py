import pandas as pd
import requests
from datetime import datetime, timedelta
import firebase_admin
from firebase_admin import credentials, firestore
import os
import json
import urllib.parse

# --- Configuração ---
COLLECTION_NAME = "cotacoes"
# O URL da API do CEPEA que queremos aceder
CEPEA_API_URL_TEMPLATE = "https://www.cepea.org.br/api/indicador/dados/id/2/d/{start_date}/{end_date}"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}

def initialize_firestore():
    """Inicializa a conexão com o Firestore usando as credenciais."""
    creds_json_str = os.getenv('FIREBASE_CREDENTIALS_JSON')
    if not creds_json_str:
        raise ValueError("A variável de ambiente FIREBASE_CREDENTIALS_JSON não está definida.")
    
    creds_dict = json.loads(creds_json_str)
    cred = credentials.Certificate(creds_dict)
    
    # Evita reinicializar a app se já estiver iniciada
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
        
    return firestore.client()

def update_firestore_data():
    """Busca os dados mais recentes da API através do ScrapingBee e os salva no Firestore."""
    print("A iniciar a atualização de dados para o Firestore...")
    db = initialize_firestore()
    
    scrapingbee_api_key = os.getenv('SCRAPINGBEE_API_KEY')
    if not scrapingbee_api_key:
        raise ValueError("A variável de ambiente SCRAPINGBEE_API_KEY não está definida.")

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

        # 1. Monta o URL do CEPEA
        cepea_url = CEPEA_API_URL_TEMPLATE.format(start_date=start_str, end_date=end_str)
        
        # 2. Monta o URL final para o ScrapingBee, que irá aceder ao URL do CEPEA por nós
        scrapingbee_url = f"https://app.scrapingbee.com/api/v1/?api_key={scrapingbee_api_key}&url={urllib.parse.quote(cepea_url)}"

        response = requests.get(scrapingbee_url, headers=HEADERS, timeout=60) # Aumenta o timeout
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
