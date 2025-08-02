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
import plotly.express as px # Importa a biblioteca Plotly

# Ajustar o layout para "wide" e maximizar o uso do espaço
st.set_page_config(layout="wide")

# Título do Dashboard (centralizado e com tamanho ajustado)
st.markdown("<h1 style='text-align: center; font-size: 3rem;'>Dashboard de Produção de Gado x Selic</h1>", unsafe_allow_html=True)

# --- FUNÇÕES AUXILIARES ---

def image_to_base64(image_path):
    """Converte uma imagem local para uma string Base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except Exception as e:
        st.error(f"Erro ao carregar a imagem {image_path}: {str(e)}")
        return None

def display_linked_image(image_path, url, caption, width):
    """Exibe uma imagem com um link incorporado."""
    if os.path.exists(image_path):
        base64_string = image_to_base64(image_path)
        if base64_string:
            image_format = "jpeg" if image_path.lower().endswith((".jpeg", ".jpg")) else "png"
            st.markdown(
                f'<a href="{url}" target="_blank"><img src="data:image/{image_format};base64,{base64_string}" width="{width}"></a>',
                unsafe_allow_html=True
            )
            st.markdown(caption, unsafe_allow_html=True)
    else:
        st.warning(f"Imagem não encontrada: {image_path}. Verifique o caminho do arquivo.")

# --- LÓGICA DE DADOS ---

def get_live_price():
    """
    Obtém a cotação mais recente de forma independente.
    """
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    
    try:
        widget_url = "https://www.cepea.org.br/br/widgetproduto.js.php?id_indicador%5B%5D=2"
        response = requests.get(widget_url, headers=headers, timeout=7)
        response.raise_for_status()
        content = response.text
        
        date_match = re.search(r"<td>(\d{2}/\d{2}/\d{4})</td>", content)
        price_match = re.search(r'R\$ <span class="maior">([\d,]+)</span>', content)

        if date_match and price_match:
            date_str = date_match.group(1)
            price_str = price_match.group(1).replace(",", ".")
            price = float(price_str)
            st.toast("Cotação diária obtida via Widget.")
            return date_str, price
    except Exception:
        pass

    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        start_str, end_str = start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
        api_url = f"https://www.cepea.org.br/api/indicador/dados/id/2/d/{start_str}/{end_str}"
        response = requests.get(api_url, headers=headers, timeout=7)
        response.raise_for_status()
        data = response.json()
        if data and data.get('result'):
            latest_entry = data['result'][-1]
            date = datetime.strptime(latest_entry['data'], "%Y-%m-%d %H:%M:%S")
            price = float(latest_entry['valor'].replace(",", "."))
            st.toast("Cotação diária obtida via API.")
            return date.strftime('%d/%m/%Y'), price
    except Exception:
        pass

    return None, None

@st.cache_data(ttl=3600)
def load_and_update_historical_data():
    """
    Carrega os dados HISTÓRICOS de um ficheiro CSV local e tenta atualizá-los.
    """
    base_csv_file = "cotacao_historica_base.csv"
    
    if not os.path.exists(base_csv_file):
        st.error(f"Ficheiro de dados base '{base_csv_file}' não encontrado!")
        return None
        
    df_historico = pd.read_csv(base_csv_file)
    df_historico["Data"] = pd.to_datetime(df_historico["Data"])
    df_historico["Cotação (R$/arroba)"] = pd.to_numeric(df_historico["Cotação (R$/arroba)"])
    
    try:
        last_local_date = df_historico['Data'].max()
        start_date = last_local_date + timedelta(days=1)
        end_date = datetime.now()

        if start_date.date() < end_date.date():
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            api_url = f"https://www.cepea.org.br/api/indicador/dados/id/2/d/{start_str}/{end_str}"
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(api_url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data and data.get('result'):
                    api_df = pd.DataFrame(data['result'])
                    api_df = api_df[['data', 'valor']]
                    api_df.columns = ['Data', 'Cotação (R$/arroba)']
                    api_df['Data'] = pd.to_datetime(api_df['Data'])
                    api_df['Cotação (R$/arroba)'] = api_df['Cotação (R$/arroba)'].str.replace(',', '.').astype(float)
                    df_historico = pd.concat([df_historico, api_df]).drop_duplicates(subset=['Data'], keep='last').sort_values("Data").reset_index(drop=True)
                    df_historico.to_csv(base_csv_file, index=False)
                    st.toast("Base de dados histórica atualizada.")
    except Exception:
        pass

    return df_historico


def annual_to_monthly_rate(annual_rate):
    """Converte taxa de juros anual para mensal."""
    return (1 + annual_rate / 100) ** (1 / 12) - 1

def get_selic_trend(period_months):
    """Gera uma tendência de Selic baseada em projeções."""
    months_proj = [7, 19, 31, 43] 
    selic_values_proj = [10.5, 9.5, 9.0, 8.5] 
    
    trend = np.interp(range(period_months + 1), months_proj, selic_values_proj, left=selic_values_proj[0], right=selic_values_proj[-1])
    return trend.tolist()

def calculate_selic_return(initial_investment, selic_rate, ir_rate, period_months):
    """Calcula o retorno acumulado de um investimento na Selic."""
    selic_monthly_rate = annual_to_monthly_rate(selic_rate)
    ir_rate_decimal = ir_rate / 100
    values = []
    for month in range(period_months + 1):
        gross_value = initial_investment * (1 + selic_monthly_rate) ** month
        profit = gross_value - initial_investment
        net_profit = profit * (1 - ir_rate_decimal)
        net_value = initial_investment + net_profit
        values.append(net_value)
    return values

# --- INÍCIO DA INTERFACE DO STREAMLIT ---

# Obter dados históricos para o gráfico
arroba_data = load_and_update_historical_data()

# Obter cotação diária (independente)
latest_date, latest_price = get_live_price()

# Fallback para cotação diária se a busca online falhar
if not latest_date and arroba_data is not None and not arroba_data.empty:
    st.toast("Não foi possível obter cotação online. A usar o último valor do histórico.")
    latest_row = arroba_data.iloc[-1]
    latest_date = latest_row['Data'].strftime('%d/%m/%Y')
    latest_price = float(latest_row['Cotação (R$/arroba)'])


# Preparar dados para o gráfico de cotação
if arroba_data is not None and not arroba_data.empty:
    end_date_filter = arroba_data["Data"].max()
    start_date_filter = end_date_filter - timedelta(days=3*365)
    arroba_data_filtered = arroba_data.loc[arroba_data["Data"] >= start_date_filter]

    monthly_avg_data = arroba_data_filtered.set_index('Data')['Cotação (R$/arroba)'].resample('M').mean().reset_index()

    fig_arroba = px.line(
        monthly_avg_data,
        x='Data',
        y='Cotação (R$/arroba)',
        title="Cotação Média Mensal da Arroba do Boi Gordo (Últimos 3 Anos)",
        labels={'Data': 'Data', 'Cotação (R$/arroba)': 'Cotação Média (R$/arroba)'},
        markers=True
    )
    
    fig_arroba.update_traces(
        line_color='#8B4513',
        hovertemplate='<b>Data</b>: %{x|%B de %Y}<br><b>Cotação Média</b>: R$ %{y:.2f}'
    )
    
    fig_arroba.update_layout(
        title_x=0.5,
        xaxis_title="Data",
        yaxis_title="Cotação Média (R$/arroba)",
        plot_bgcolor='rgba(255, 255, 255, 0.1)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=400
    )
    
    fig_arroba.add_annotation(
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        text="OS CAPITAL",
        showarrow=False,
        font=dict(
            size=50,
            color="green"
        ),
        opacity=0.15
    )
    
    fig_arroba.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig_arroba.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')

else:
    fig_arroba = None


# --- LAYOUT DA PÁGINA ---

col_main, col_logos = st.columns([3, 1])

with col_main:
    if latest_date and latest_price:
        st.markdown(
            f"""<div style='text-align: center; margin-bottom: 1rem;'>
                   <span style='font-size: 1.5rem; color: #8B4513; font-weight: bold;'>Cotação Diária: {latest_date} - R$ {latest_price:.2f}</span>
                   <span style='font-size: 0.9rem;'> | Fonte: <a href='https://www.cepea.org.br/br' target='_blank'>CEPEA</a></span>
               </div>""",
            unsafe_allow_html=True
        )

    if fig_arroba:
        st.plotly_chart(fig_arroba, use_container_width=True)
    elif arroba_data is None:
         st.error("Não foi possível carregar os dados históricos para exibir o gráfico.")


# --- BARRA LATERAL (INPUTS DO USUÁRIO) ---

st.sidebar.header("Parâmetros da Simulação")
initial_investment = st.sidebar.number_input("Investimento Inicial (R$)", min_value=1.0, value=100000.0, step=1000.0)
# **LAYOUT ATUALIZADO**: Custo de capital movido para baixo do investimento inicial
cost_of_capital_rate = st.sidebar.number_input("Custo de Capital (% ao ano)", min_value=0.0, value=0.0, step=0.5, format="%.2f")
num_heads_bought = st.sidebar.number_input("Quantidade de Cabeças Comprada", min_value=1, value=50, step=1)
period_months = st.sidebar.slider("Período de Análise (meses)", min_value=1, max_value=48, value=12)

st.sidebar.markdown("---")

default_buy_price = latest_price if latest_price else 290.0
buy_arroba_price = st.sidebar.number_input("Preço da Arroba na Compra (R$/arroba)", min_value=1.0, value=default_buy_price, step=0.01, format="%.2f")
sell_arroba_price = st.sidebar.number_input("Preço da Arroba na Venda (R$/arroba)", min_value=1.0, value=default_buy_price + 10, step=0.01, format="%.2f")
arrobas_gain_period = st.sidebar.number_input("Ganho de Arrobas por Cabeça no Período", min_value=0.0, value=7.0, step=0.5)

st.sidebar.markdown("---")

cost_per_head_feed = st.sidebar.number_input("Custo Mensal por Cabeça (Alimentação, etc.) (R$)", min_value=0.0, value=80.0, step=5.0)
fixed_costs = st.sidebar.number_input("Outros Custos Fixos no Período (R$)", min_value=0.0, value=5000.0, step=100.0)


st.sidebar.markdown("---")
st.sidebar.subheader("Parâmetros da Aplicação")
selic_rate = st.sidebar.slider("Taxa Selic (% ao ano)", min_value=0.0, max_value=20.0, value=10.5, step=0.25)

period_days = period_months * 30
if period_days <= 180: default_ir_rate = 22.5
elif 181 <= period_days <= 360: default_ir_rate = 20.0
elif 361 <= period_days <= 720: default_ir_rate = 17.5
else: default_ir_rate = 15.0

ir_rate_options = [22.5, 20.0, 17.5, 15.0]
ir_rate_index = ir_rate_options.index(default_ir_rate)
ir_rate = st.sidebar.selectbox("Alíquota de IR na Aplicação (%)", options=ir_rate_options, index=ir_rate_index)

if initial_investment <= 0 or buy_arroba_price <= 0 or sell_arroba_price <= 0 or num_heads_bought <= 0:
    with col_main:
        st.error("Por favor, preencha os campos de investimento, preços e quantidade de cabeças com valores maiores que 0.")
else:
    # --- CÁLCULOS PRINCIPAIS ---
    total_arrobas_bought = initial_investment / buy_arroba_price
    total_arrobas_gain = num_heads_bought * arrobas_gain_period
    total_arrobas_sold = total_arrobas_bought + total_arrobas_gain
    total_revenue = total_arrobas_sold * sell_arroba_price
    
    feed_cost = cost_per_head_feed * num_heads_bought * period_months
    
    capital_cost = initial_investment * (cost_of_capital_rate / 100) * (period_months / 12)
    
    total_costs = initial_investment + fixed_costs + feed_cost + capital_cost
    total_profit_gado = total_revenue - total_costs
    
    gado_values = np.linspace(initial_investment, initial_investment + total_profit_gado, period_months + 1).tolist()

    selic_values = calculate_selic_return(initial_investment, selic_rate, ir_rate, period_months)
    selic_profit = selic_values[-1] - initial_investment

    months = list(range(period_months + 1))
    df_comparison = pd.DataFrame({
        "Mês": months,
        "Produção de Gado (R$ Nominais)": gado_values,
        "Aplicação Selic (R$ Nominais)": selic_values
    })

    gado_cash_flow = [-initial_investment] + [0] * (period_months - 1) + [initial_investment + total_profit_gado]
    gado_irr_monthly = npf.irr(gado_cash_flow)
    gado_irr_monthly_pct = gado_irr_monthly * 100 if gado_irr_monthly is not None and not np.isnan(gado_irr_monthly) else 0

    selic_cash_flow = [-initial_investment] + [0] * (period_months - 1) + [initial_investment + selic_profit]
    selic_irr_monthly = npf.irr(selic_cash_flow)
    selic_irr_monthly_pct = selic_irr_monthly * 100 if selic_irr_monthly is not None and not np.isnan(selic_irr_monthly) else 0

    # --- GRÁFICO COMPARATIVO E TABELA DE RESULTADOS ---
    with col_main:
        st.markdown("---")
        fig_comp, ax1 = plt.subplots(figsize=(15, 6))

        sns.lineplot(x="Mês", y="Produção de Gado (R$ Nominais)", data=df_comparison, label="Produção de Gado", ax=ax1, linewidth=2.5)
        sns.lineplot(x="Mês", y="Aplicação Selic (R$ Nominais)", data=df_comparison, label=f"Aplicação Selic ({selic_rate}%)", ax=ax1, linewidth=2.5)

        ax1.set_xlabel("Mês\nElaborado por: OS CAPITAL", fontsize=12)
        ax1.set_ylabel("Valor Acumulado (R$)", fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.set_title("Comparativo de Investimentos: Produção de Gado x Selic", fontsize=16)
        
        ax1.text(0.5, 0.5, "OS CAPITAL", fontsize=50, color='gray', alpha=0.15, ha='center', va='center', transform=ax1.transAxes)
        
        selic_trend = get_selic_trend(period_months)
        ax2 = ax1.twinx()
        ax2.plot(months, selic_trend, label="Taxa Selic Projetada (%)", linestyle='--', color='red', alpha=0.7)
        ax2.set_ylabel("Taxa Selic Projetada (% ao ano)", fontsize=12)
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

        plt.tight_layout()
        st.pyplot(fig_comp)

        st.subheader("Resumo da Operação")
        
        gain_from_arrobas = total_arrobas_gain * sell_arroba_price
        price_difference_profit = (sell_arroba_price - buy_arroba_price) * total_arrobas_bought
        total_op_cost = fixed_costs + feed_cost + capital_cost

        table_data = {
            "Métrica": [
                "<b>Lucro Total com Produção de Gado (R$)</b>",
                "Lucro Total com Selic (R$)",
                "TIR Mensal - Gado (%)",
                "TIR Mensal - Selic (%)",
                "--- Detalhamento do Gado ---",
                "Ganho com Aumento de Arrobas (R$)",
                "Ganho/Perda com Diferença de Preço (R$)",
                "Custo de Capital (Juros) (R$)",
                "Custo Operacional Total (R$)",
            ],
            "Valor": [
                f"<b>R$ {total_profit_gado:,.2f}</b>",
                f"R$ {selic_profit:,.2f}",
                f"{gado_irr_monthly_pct:.2f}%",
                f"{selic_irr_monthly_pct:.2f}%",
                "",
                f"R$ {gain_from_arrobas:,.2f}",
                f"R$ {price_difference_profit:,.2f}",
                f"- R$ {capital_cost:,.2f}",
                f"- R$ {total_op_cost:,.2f}",
            ]
        }
        
        df_table = pd.DataFrame(table_data)
        st.markdown(df_table.to_html(escape=False, index=False), unsafe_allow_html=True)


# --- COLUNA DE LOGOS ---
with col_logos:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.2rem; text-align: center;'><b>OS CAPITAL</b></p>", unsafe_allow_html=True)
    display_linked_image("assets/oscapital.jpeg", "https://oscapitaloficial.com.br/", "<p style='font-size: 1rem; text-align: center;'>VISITE NOSSO SITE</p>", 200)
    
    st.markdown("---")
    
    st.markdown("<p style='font-size: 1.2rem; text-align: center;'><b>Interactive Brokers</b></p>", unsafe_allow_html=True)
    display_linked_image(
        "assets/IB_logo_stacked1.jpg",
        "https://ibkr.com/referral/edgleison239",
        "<p style='font-size: 1rem; text-align: center;'>INVISTA EM MAIS DE 160<br>MERCADOS EM TODO O<br>MUNDO</p>",
        200
    )
