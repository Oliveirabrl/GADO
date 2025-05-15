import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy_financial as npf
import os
import base64

# Título do Dashboard
st.title("Dashboard de Produção de Gado x Selic")

# Caminhos para as imagens das logos
logo_os_capital_path = "assets/oscapital.jpeg"
logo_interactive_brokers_path = "assets/IB_logo_stacked1.jpg"

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
        # Converter a imagem para Base64
        base64_string = image_to_base64(image_path)
        if base64_string:
            # Determinar o tipo de imagem (jpeg ou jpg) com base na extensão
            image_format = "jpeg" if image_path.lower().endswith(".jpeg") else "jpg"
            # Exibir a imagem com link incorporado usando HTML e Base64
            st.markdown(
                f'<a href="{url}" target="_blank"><img src="data:image/{image_format};base64,{base64_string}" width="{width}"></a>',
                unsafe_allow_html=True
            )
            # Adicionar a frase abaixo da imagem
            st.markdown(caption)
    else:
        st.error(f"Imagem não encontrada: {image_path}. Verifique o caminho do arquivo.")

# Exibir as logos com links incorporados diretamente nas imagens e frases abaixo
col1, col2 = st.columns(2)

# Proporção das larguras com base no comprimento das frases
os_capital_caption = "VISITE NOSSO SITE"
interactive_brokers_caption = "INVISTA EM MAIS DE 160 MERCADOS EM TODO O MUNDO"
os_capital_width = 200  # Largura base para OS CAPITAL
interactive_brokers_width = min(os_capital_width * (len(interactive_brokers_caption) / len(os_capital_caption)), 400)  # Proporcional, com limite de 400px

with col1:
    st.markdown("**OS CAPITAL**")
    display_linked_image(logo_os_capital_path, "https://oscapitaloficial.com.br/", os_capital_caption, os_capital_width)

with col2:
    st.markdown("**Interactive Brokers**")
    display_linked_image(logo_interactive_brokers_path, "https://ibkr.com/referral/edgleison239", interactive_brokers_caption, interactive_brokers_width)

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
# Converter meses para dias (assumindo 30 dias por mês)
period_days = period_months * 30

# Definir a alíquota com base nas regras
if period_days <= 180:
    default_ir_rate = 22.5
elif 181 <= period_days <= 360:
    default_ir_rate = 20.0
elif 361 <= period_days <= 720:
    default_ir_rate = 17.5
else:  # Acima de 721 dias
    default_ir_rate = 15.0

# Pré-selecionar a alíquota no selectbox
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

    # Gráfico
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plotar valores acumulados (R$) no eixo y primário
    sns.lineplot(x="Mês", y="Produção de Gado (R$ Nominais)", data=df, label="Produção de Gado", marker="o", ax=ax1)
    sns.lineplot(x="Mês", y="Aplicação Selic (fixa) (R$ Nominais)", data=df, label="Aplicação Selic (fixa)", marker="s", ax=ax1)

    # Criar eixo y secundário para a taxa Selic (%)
    ax2 = ax1.twinx()
    ax2.plot(months, selic_trend, label="Taxa Selic Projetada (%)", linestyle='--', color='red', marker='None')
    ax2.set_ylabel("Taxa Selic (% ao ano)")

    # Configurar rótulos e legenda
    ax1.set_xlabel("Mês\nElaborado por: OS CAPITAL")
    ax1.set_ylabel("Valor (R$ Nominais)")
    ax1.grid(True)

    # Título do gráfico
    ax1.set_title("Produção de Gado x Selic")

    # Adicionar marca d'água
    ax1.text(0.5, 0.5, "OS CAPITAL", fontsize=40, color='green', alpha=0.2, ha='center', va='center', transform=ax1.transAxes)

    # Combinar legendas dos dois eixos
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    # Ajustar layout para evitar sobreposição
    plt.tight_layout()

    # Exibir o gráfico
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
    st.table(table_data)