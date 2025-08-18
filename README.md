# Dashboard de Produção de Gado x Selic

Este é um dashboard interativo construído com Streamlit para analisar a viabilidade da produção de gado em comparação com um investimento na taxa Selic.

## Funcionalidades

- Visualização da cotação histórica da arroba do boi gordo (CEPEA/ESALQ).
- Simulação de investimento em produção de gado com parâmetros customizáveis.
- Comparativo de rentabilidade com a taxa Selic.
- Cálculo de TIR (Taxa Interna de Retorno) para ambas as operações.

## Como Manter os Dados Atualizados

A aplicação utiliza uma base de dados local (`cotacao_historica_base.csv`) para garantir que o dashboard funcione sempre, mesmo que a ligação à internet falhe no servidor.

Para que os dados exibidos estejam sempre atualizados, é necessário executar um script simples no seu computador local antes de enviar as alterações para o GitHub.

### Pré-requisitos

- Ter o Python instalado na sua máquina.
- Ter as bibliotecas necessárias instaladas. Pode instalá-las com o comando:
  ```bash
  pip install pandas requests
  ```

### Passos para a Atualização

1.  **Execute o Script de Atualização:**
    Abra um terminal na pasta do projeto e execute o seguinte comando:
    ```bash
    python update_data.py
    ```
    O script irá ligar-se à API do CEPEA, buscar os dados mais recentes e atualizar o ficheiro `cotacao_historica_base.csv` automaticamente.

2.  **Envie as Alterações para o GitHub:**
    Depois de executar o script, o ficheiro de dados estará atualizado. Agora, basta enviar esta alteração para o seu repositório:
    ```bash
    git add cotacao_historica_base.csv
    git commit -m "Atualiza dados da cotação"
    git push
    ```

Após o `push`, o Render irá automaticamente fazer o "deploy" da nova versão da sua aplicação com os dados mais recentes. Recomenda-se executar este processo periodicamente (ex: uma vez por semana) para manter o dashboard atualizado.
