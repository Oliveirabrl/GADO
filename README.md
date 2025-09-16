Simulador de Viabilidade: Gado vs. Selic

Este projeto é um dashboard interativo construído com Streamlit para analisar a viabilidade financeira da engorda de gado em comparação com investimentos na taxa Selic, além de fornecer análises históricas do mercado.
Como Atualizar os Dados Históricos

Para manter os gráficos de análise de mercado atualizados, o sistema utiliza um script que processa os dados brutos e gera o ficheiro consolidado que a aplicação utiliza.

O processo é simples:

    Adicione os Ficheiros de Dados Brutos:
    Coloque os ficheiros Excel (.xls ou .xlsx) com os históricos de preços diários dentro da pasta dados_historicos/dados_geral/. Os ficheiros devem ter os seguintes nomes:

        preco_brl_arroba.xls

        preco_bezerro_brl.xls

        preco_soja_brl.xls

        preco_milho_brl.xls

    Nota: Cada ficheiro deve conter, no mínimo, as colunas "Data" e "Último".

    Execute o Script de Processamento:
    Abra o terminal na pasta principal do projeto e execute o seguinte comando:

    python processador_dados.py

    Verifique o Resultado:
    O script irá ler os ficheiros, processar os dados e criar/atualizar automaticamente o ficheiro dados_historicos/arroba-sp-historico.csv.

    Execute o Dashboard:
    Depois de atualizar os dados, pode iniciar o dashboard normalmente:

    streamlit run main.py
