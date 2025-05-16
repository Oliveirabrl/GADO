import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy_financial as npf
import os
import base64
from datetime import datetime, timedelta
import requests
import re

# Ajustar o layout para "wide" e maximizar o uso do espaço
st.set_page_config(layout="wide")

# Título do Dashboard (centralizado e com tamanho ajustado)
st.markdown("<h1 style='text-align: center; font-size: 3rem;'>Dashboard de Produção de Gado x Selic</h1>", unsafe_allow_html=True)

# Função para converter imagem para Base64
def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except Exception as e:
        st.error(f"Erro ao carregar a imagem {image_path}: {str(e)}")
        return None

# Função para exibir a imagem com link incorporado e frase
def display_linked_image(image_path, url, caption, width):
    if os.path.exists(image_path):
        base64_string = image_to_base64(image_path)
        if base64_string:
            image_format = "jpeg" if image_path.lower().endswith(".jpeg") else "jpg"
            st.markdown(
                f'<a href="{url}" target="_blank"><img src="data:image/{image_format};base64,{base64_string}" width="{width}"></a>',
                unsafe_allow_html=True
            )
            st.markdown(caption, unsafe_allow_html=True)
    else:
        st.error(f"Imagem não encontrada: {image_path}. Verifique o caminho do arquivo.")

# Função para carregar dados históricos do arquivo Excel do CEPEA
def load_historical_data():
    excel_file = "CEPEA_20250516120712.xls"
    try:
        # Tentar ler o arquivo Excel com diferentes motores
        try:
            df = pd.read_excel(excel_file, engine="xlrd")
        except Exception as e_xlrd:
            try:
                df = pd.read_excel(excel_file, engine="openpyxl")
            except Exception as e_openpyxl:
                st.error(f"Erro ao ler o arquivo Excel: {str(e_xlrd)} (xlrd) / {str(e_openpyxl)} (openpyxl)")
                return pd.DataFrame()
        
        # Ajustar nomes das colunas
        df.columns = df.columns.str.strip()
        if "Data" not in df.columns or "Valor" not in df.columns:
            st.error("O arquivo Excel deve conter as colunas 'Data' e 'Valor'. Verifique o arquivo CEPEA_20250516120712.xls.")
            return pd.DataFrame()
        
        # Converter a coluna 'Data' para datetime
        df["Data"] = pd.to_datetime(df["Data"], format="%d/%m/%Y", errors="coerce")
        df = df.dropna(subset=["Data"])  # Remover linhas com datas inválidas
        
        # Filtrar de 2020 até 16/05/2025
        start_date = pd.to_datetime("2020-01-01")
        end_date = pd.to_datetime("2025-05-16")
        df = df[(df["Data"] >= start_date) & (df["Data"] <= end_date)]
        
        # Renomear a coluna 'Valor' para 'Cotação (R$/arroba)'
        df = df.rename(columns={"Valor": "Cotação (R$/arroba)"})
        
        # Arredondar os valores para números inteiros
        df["Cotação (R$/arroba)"] = df["Cotação (R$/arroba)"].round(0).astype(int)
        
        # Selecionar apenas as colunas necessárias
        df = df[["Data", "Cotação (R$/arroba)"]]
        
        # Salvar como CSV para uso futuro
        df.to_csv("cotacao_historica.csv", index=False)
        return df
    
    except Exception as e:
        st.error(f"Erro ao ler o arquivo Excel: {str(e)}")
        return pd.DataFrame()

# Função para obter a cotação mais recente do widget do CEPEA e atualizar o histórico
@st.cache_data(ttl=300)  # Atualiza a cada 5 minutos (300 segundos)
def fetch_arroba_data():
    # Verificar se o CSV histórico já existe; se não, carregar do Excel
    csv_file = "cotacao_historica.csv"
    if not os.path.exists(csv_file):
        df = load_historical_data()
        if df.empty:
            # Dados de fallback (simulados) caso o Excel não seja carregado
            end_date = datetime(2025, 5, 16)
            start_date = pd.to_datetime("2020-01-01")
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            np.random.seed(42)
            base_price = 280
            trend = np.linspace(0, 50, len(dates))  # Tendência de alta ao longo dos anos
            noise = np.random.normal(0, 5, len(dates))
            prices = base_price + trend + noise
            # Arredondar os valores simulados para números inteiros
            prices = np.round(prices).astype(int)
            df = pd.DataFrame({
                "Data": dates,
                "Cotação (R$/arroba)": prices
            })
            df.to_csv(csv_file, index=False)
    else:
        df = pd.read_csv(csv_file)
        df["Data"] = pd.to_datetime(df["Data"])
        # Garantir que os valores no CSV já existente sejam inteiros
        df["Cotação (R$/arroba)"] = df["Cotação (R$/arroba)"].round(0).astype(int)

    try:
        # URL do widget do CEPEA
        widget_url = "https://www.cepea.org.br/br/widgetproduto.js.php?fonte=arial&tamanho=10&largura=400px&corfundo=dbd6b2&cortexto=333333&corlinha=ede7bf&id_indicador%5B%5D=2"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(widget_url, headers=headers)
        response.raise_for_status()

        # Extrair a tabela HTML do JavaScript
        content = response.text
        table_match = re.search(r"<table.*?</table>", content, re.DOTALL)
        if not table_match:
            st.error("Não foi possível extrair a tabela do widget do CEPEA.")
            return df, None, None

        table_html = table_match.group(0)
        # Ajustar as expressões regulares para capturar a data e o preço
        # Data: primeira célula do <tbody>
        date_match = re.search(r"<tbody>.*?<td>(\d{2}/\d{2}/\d{4})</td>", table_html, re.DOTALL)
        # Preço: terceira célula do <tbody>, dentro de <span class="maior">
        price_match = re.search(r"<td>R\$ <span class=\"maior\">([\d,]+)</span></td>", table_html)

        if not date_match or not price_match:
            st.error("Não foi possível extrair a data ou o preço do widget do CEPEA.")
            return df, None, None

        date_str = date_match.group(1)  # Ex.: "15/05/2025"
        price_str = price_match.group(1).replace(",", ".")  # Ex.: "308.00"
        date = datetime.strptime(date_str, "%d/%m/%Y")
        price = round(float(price_str))  # Arredondar para número inteiro

        # Verificar se a data já existe no DataFrame
        if date not in df["Data"].values:
            # Adicionar a nova cotação
            new_row = pd.DataFrame({"Data": [date], "Cotação (R$/arroba)": [price]})
            df = pd.concat([df, new_row], ignore_index=True)
            # Ordenar por data
            df = df.sort_values("Data")
            # Salvar no CSV
            df.to_csv(csv_file, index=False)

        return df, date_str, price

    except Exception as e:
        st.error(f"Erro ao buscar cotações do widget: {str(e)}")
        return df, None, None

# Obter os dados da cotação e a cotação diária mais recente
arroba_data, latest_date, latest_price = fetch_arroba_data()

# Filtrar para os últimos 3 anos (dinâmico)
current_date = datetime(2025, 5, 16)  # Data atual
start_date = current_date - timedelta(days=3*365)  # 3 anos atrás (aproximadamente)
arroba_data = arroba_data[arroba_data["Data"] >= start_date]

# Agregar os dados para cotações semanais (média da cotação por semana)
arroba_data.set_index("Data", inplace=True)
weekly_data = arroba_data.resample('W').mean().reset_index()
weekly_data["Cotação (R$/arroba)"] = weekly_data["Cotação (R$/arroba)"].round(0).astype(int)

# Criar o gráfico da cotação da arroba do boi gordo (semanal, apenas linha)
fig_arroba, ax_arroba = plt.subplots(figsize=(15, 4))  # Tamanho fixo para ambos os gráficos
sns.lineplot(x="Data", y="Cotação (R$/arroba)", data=weekly_data, ax=ax_arroba)  # Apenas linha
ax_arroba.set_title("Cotação da Arroba do Boi Gordo (Últimos 3 Anos)", fontsize=16)  # Título dentro do gráfico
ax_arroba.set_xlabel("Data", fontsize=12)
ax_arroba.set_ylabel("Cotação (R$/arroba)", fontsize=12)
ax_arroba.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Criar duas colunas: uma maior para os gráficos e tabela, outra menor para as logos
col_main, col_logos = st.columns([3, 1])

with col_main:
    # Exibir a cotação diária abaixo do gráfico, com destaque e crédito ao CEPEA
    if latest_date and latest_price:
        st.markdown(
            f"<p style='font-size: 1.2rem; color: #8B4513;'><span style='font-size: 1.2rem; color: #8B4513;'>Cotação Diária: {latest_date} - R$ {latest_price}</span> <span style='font-size: 0.9rem; color: #FFFFFF;'>- cotação disponibilizada pela empresa <a href='https://www.cepea.org.br/br' target='_blank' style='color: #FFFFFF; text-decoration: underline;'>CEPEA</a></span></p>",
            unsafe_allow_html=True
        )

    # Exibir o gráfico da cotação (o título agora está dentro do gráfico)
    st.pyplot(fig_arroba)

# Barra lateral com parâmetros ajustáveis (valores padrão ajustados para 1)
st.sidebar.header("Parâmetros")
selic_rate = st.sidebar.slider("Taxa Selic (% ao ano)", min_value=0.0, max_value=20.0, value=1.0, step=0.25)
period_months = st.sidebar.slider("Período de Análise (meses)", min_value=1, max_value=36, value=1)
initial_investment = st.sidebar.number_input("Investimento Inicial (R$)", min_value=0.0, value=1.0, step=1000.0)
num_heads_bought = st.sidebar.number_input("Quantidade de Cabeças Comprada", min_value=0.0, value=1.0, step=1.0)

# Cálculo e exibição da Quantidade Total de Arrobas Compradas logo após o filtro de cabeças
buy_arroba_price = st.sidebar.number_input("Preço da Arroba na Compra (R$/arroba)", min_value=0.0, value=1.0, step=1.0)
if buy_arroba_price > 0:
    total_arrobas_bought = initial_investment / buy_arroba_price
    st.sidebar.text(f"Quantidade Total de Arrobas Compradas: {total_arrobas_bought:.2f}")
else:
    total_arrobas_bought = 0
    st.sidebar.text("Quantidade Total de Arrobas Compradas: 0.00")

# Outros parâmetros
sell_arroba_price = st.sidebar.number_input("Preço da Arroba na Venda (R$/arroba)", min_value=0.0, value=1.0, step=1.0)
arrobas_gain_period = st.sidebar.number_input("Ganhos de Arrobas no Período", min_value=0.0, value=1.0, step=10.0)

# Filtro de Custo Médio por Cabeça
cost_per_head_feed = st.sidebar.number_input("Custo Médio Mensal por Cabeça (Alimentação) (R$/cabeça)", min_value=0.0, value=1.0, step=1.0)

fixed_costs = st.sidebar.number_input("Custos Fixos (R$)", min_value=0.0, value=1.0, step=100.0)

# Determinar a alíquota de IR com base no período de análise
period_days = period_months * 30
if period_days <= 180:
    default_ir_rate = 22.5
elif 181 <= period_days <= 360:
    default_ir_rate = 20.0
elif 361 <= period_days <= 720:
    default_ir_rate = 17.5
else:
    default_ir_rate = 15.0

ir_rate_options = [22.5, 20.0, 17.5, 15.0]
ir_rate_index = ir_rate_options.index(default_ir_rate)
ir_rate = st.sidebar.selectbox("Alíquota de IR na Aplicação (%)", options=ir_rate_options, index=ir_rate_index)

# Validação dos campos obrigatórios
if initial_investment <= 0 or buy_arroba_price <= 0 or sell_arroba_price <= 0:
    st.error("Por favor, preencha os campos 'Investimento Inicial', 'Preço da Arroba na Compra' e 'Preço da Arroba na Venda' com valores maiores que 0.")
else:
    # Cálculo do Lucro Total da Operação
    total_arrobas_sold = total_arrobas_bought + arrobas_gain_period
    total_revenue = total_arrobas_sold * sell_arroba_price
    
    # Cálculo do custo de alimentação por cabeça
    feed_cost = cost_per_head_feed * num_heads_bought * period_months
    total_costs = initial_investment + fixed_costs + feed_cost
    total_profit = total_revenue - total_costs

    # Calcular o lucro mensal médio para distribuição ao longo do período
    monthly_profit = total_profit / period_months

    # Função para converter taxa anual para mensal
    def annual_to_monthly_rate(annual_rate):
        return (1 + annual_rate / 100) ** (1 / 12) - 1

    # Função para obter a tendência da Selic (projeções do Relatório Focus)
    def get_selic_trend(period_months):
        months = [7, 19, 31, 43]  # Meses correspondentes a dez/2025, dez/2026, dez/2027, dez/2028 (a partir de maio/2025)
        selic_values = [15.0, 12.5, 10.5, 10.0]
        trend = []
        for month in range(period_months + 1):
            if month <= months[0]:
                value = selic_values[0]
            elif month >= months[-1]:
                value = selic_values[-1]
            else:
                for i in range(len(months) - 1):
                    if months[i] <= month < months[i + 1]:
                        t = (month - months[i]) / (months[i + 1] - months[i])
                        value = selic_values[i] + t * (selic_values[i + 1] - selic_values[i])
                        break
            trend.append(value)
        return trend

    # Função para calcular o rendimento líquido da Selic (fixa)
    def calculate_selic_return(initial_investment, selic_rate, ir_rate, period_months):
        selic_monthly_rate = annual_to_monthly_rate(selic_rate)
        ir_rate_decimal = ir_rate / 100
        monthly_net_rate = selic_monthly_rate * (1 - ir_rate_decimal)
        values = []
        for month in range(period_months + 1):
            future_value = initial_investment * (1 + monthly_net_rate) ** month
            values.append(future_value)
        return values

    # Obter tendência da Selic
    selic_trend = get_selic_trend(period_months)

    # Cálculo para a aplicação Selic (fixa)
    selic_values = calculate_selic_return(initial_investment, selic_rate, ir_rate, period_months)

    # Para a produção de gado, acumular o lucro mensalmente
    gado_values = []
    current_value = initial_investment
    gado_values.append(current_value)
    for month in range(period_months):
        current_value += monthly_profit
        gado_values.append(current_value)

    # DataFrame para o gráfico (sem a Selic com tendência)
    months = list(range(period_months + 1))
    df = pd.DataFrame({
        "Mês": months,
        "Produção de Gado (R$ Nominais)": gado_values,
        "Aplicação Selic (fixa) (R$ Nominais)": selic_values
    })

    # Calcular a TIR mensal para Selic
    selic_cash_flow = [-initial_investment] + [0] * (period_months - 1) + [selic_values[-1]]
    selic_irr_monthly = npf.irr(selic_cash_flow)
    selic_irr_monthly_pct = selic_irr_monthly * 100 if selic_irr_monthly is not None and not np.isnan(selic_irr_monthly) else None

    # Calcular a TIR mensal para Produção de Gado
    gado_cash_flow = [-initial_investment] + [0] * (period_months - 1) + [initial_investment + total_profit]
    gado_irr_monthly = npf.irr(gado_cash_flow)
    gado_irr_monthly_pct = gado_irr_monthly * 100 if gado_irr_monthly is not None and not np.isnan(gado_irr_monthly) else None

    # Gráfico da Produção de Gado x Selic (apenas linha, mesma largura)
    fig, ax1 = plt.subplots(figsize=(15, 6))  # Mesma largura do gráfico anterior

    # Plotar valores acumulados (R$) no eixo y primário (apenas linha)
    sns.lineplot(x="Mês", y="Produção de Gado (R$ Nominais)", data=df, label="Produção de Gado", ax=ax1)
    sns.lineplot(x="Mês", y="Aplicação Selic (fixa) (R$ Nominais)", data=df, label="Aplicação Selic (fixa)", ax=ax1)

    # Criar eixo y secundário para a taxa Selic (%) (apenas linha)
    ax2 = ax1.twinx()
    ax2.plot(months, selic_trend, label="Taxa Selic Projetada (%)", linestyle='--', color='red')
    ax2.set_ylabel("Taxa Selic (% ao ano)", fontsize=12)

    # Configurar rótulos e legenda
    ax1.set_xlabel("Mês\nElaborado por: OS CAPITAL", fontsize=12)
    ax1.set_ylabel("Valor (R$ Nominais)", fontsize=12)
    ax1.grid(True)

    # Título do gráfico
    ax1.set_title("Produção de Gado x Selic", fontsize=16)

    # Adicionar marca d'água
    ax1.text(0.5, 0.5, "OS CAPITAL", fontsize=40, color='green', alpha=0.2, ha='center', va='center', transform=ax1.transAxes)

    # Combinar legendas dos dois eixos
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    # Ajustar layout para evitar sobreposição
    plt.tight_layout()

    with col_main:
        # Exibir o segundo gráfico (Produção de Gado x Selic)
        st.pyplot(fig)

        # Calcular métricas para a tabela
        selic_profit = selic_values[-1] - initial_investment  # Lucro com Selic (fixa)
        gain_from_arrobas = arrobas_gain_period * sell_arroba_price  # Ganho com aumento de arrobas
        price_difference_profit = (sell_arroba_price - buy_arroba_price) * total_arrobas_bought  # Lucro/prejuízo com diferença de preço
        total_cost = fixed_costs + feed_cost  # Custo total (fixo + alimentação)

        # Criar a tabela de lucros com TIR mensal
        st.subheader("Resumo de Lucros")
        table_data = {
            "Métrica": [
                "Lucro com Selic (fixa) (R$)",
                "Ganho com Aumento de Arrobas (R$)",
                "Lucro/Prejuízo com Diferença de Preço (R$)",
                "Custo Total (R$)",
                "Lucro Total com Produção de Gado (R$)"
            ],
            "Valor": [
                f"{selic_profit:.2f}",
                f"{gain_from_arrobas:.2f}",
                f"{price_difference_profit:.2f}",
                f"{total_cost:.2f}",
                f"{total_profit:.2f}"
            ],
            "TIR Mensal (%)": [
                f"{selic_irr_monthly_pct:.2f}" if selic_irr_monthly_pct is not None else "N/A",
                "N/A",
                "N/A",
                "N/A",
                f"{gado_irr_monthly_pct:.2f}" if gado_irr_monthly_pct is not None else "N/A"
            ]
        }
        # Ajustar a largura da tabela para ser igual à dos gráficos
        st.markdown(
            """
            <style>
            .stTable {width: 100% !important; max-width: 100% !important;}
            </style>
            """,
            unsafe_allow_html=True
        )
        st.table(table_data)

# Exibir as logos na coluna da direita com ajustes
with col_logos:
    # Aumentar o tamanho da fonte dos textos das logos
    st.markdown(
        """
        <style>
        .logo-text {font-size: 1.2rem !important;}
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("<p class='logo-text'><b>OS CAPITAL</b></p>", unsafe_allow_html=True)
    display_linked_image("assets/oscapital.jpeg", "https://oscapitaloficial.com.br/", "<p class='logo-text'>VISITE NOSSO SITE</p>", 200)
    
    st.markdown("<p class='logo-text'><b>Interactive Brokers</b></p>", unsafe_allow_html=True)
    display_linked_image(
        "assets/IB_logo_stacked1.jpg",
        "https://ibkr.com/referral/edgleison239",
        "<p class='logo-text'>INVISTA EM MAIS DE 160<br>MERCADOS EM TODO O<br>MUNDO</p>",
        200
    )