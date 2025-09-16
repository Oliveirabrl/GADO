import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import os
import base64
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Configuração da Página ---
st.set_page_config(layout="wide")

# --- Define o caminho base do projeto de forma segura ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Funções Auxiliares para Imagens ---
def image_to_base64(image_path):
    """Converte uma imagem local para uma string Base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except Exception:
        return None

def display_linked_image(image_path, url, caption, width):
    """Exibe uma imagem com um link incorporado."""
    full_image_path = os.path.join(BASE_DIR, image_path)
    if os.path.exists(full_image_path):
        base64_string = image_to_base64(full_image_path)
        if base64_string:
            image_format = "jpeg" if full_image_path.lower().endswith((".jpeg", ".jpg")) else "png"
            st.markdown(
                f'<a href="{url}" target="_blank"><img src="data:image/{image_format};base64,{base64_string}" width="{width}"></a>',
                unsafe_allow_html=True
            )
            st.markdown(caption, unsafe_allow_html=True)

# --- Carregamento de Dados ---
@st.cache_data
def load_data(file_name, sep=';', decimal=','):
    """Carrega um ficheiro CSV da pasta dados_historicos e limpa os nomes das colunas."""
    file_path = os.path.join(BASE_DIR, 'dados_historicos', file_name)
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, sep=sep, decimal=decimal)
            df.columns = df.columns.str.strip()
            return df
        except Exception as e:
            st.error(f"Erro ao ler o ficheiro {file_path}: {e}")
            return None
    else:
        st.warning(f"Ficheiro não encontrado: {file_path}")
        return None

# --- Função para calcular o custo da alimentação dinamicamente ---
def calcular_custo_alimentacao(df, arrobas_a_ganhar, percentual_milho, conversao_alimentar, pct_concentrado_dieta):
    KG_POR_ARROBA = 15
    RENDIMENTO_CARCACA = 0.55
    MS_MILHO = 0.88
    MS_SOJA = 0.89
    KG_SACA_MILHO = 60
    KG_SACA_SOJA = 50
    kg_carcaca_a_ganhar = arrobas_a_ganhar * KG_POR_ARROBA
    kg_peso_vivo_a_ganhar = kg_carcaca_a_ganhar / RENDIMENTO_CARCACA
    consumo_total_ms = kg_peso_vivo_a_ganhar * conversao_alimentar
    total_concentrado_ms = consumo_total_ms * (pct_concentrado_dieta / 100)
    percentual_soja = 100 - percentual_milho
    milho_ms = total_concentrado_ms * (percentual_milho / 100)
    soja_ms = total_concentrado_ms * (percentual_soja / 100)
    consumo_total_milho_kg = milho_ms / MS_MILHO
    consumo_total_soja_kg = soja_ms / MS_SOJA
    sacos_de_milho = consumo_total_milho_kg / KG_SACA_MILHO
    sacos_de_soja = consumo_total_soja_kg / KG_SACA_SOJA
    custo_total_milho = df['preco_milho_brl'] * sacos_de_milho
    custo_total_soja = df['preco_soja_brl'] * sacos_de_soja
    return custo_total_milho + custo_total_soja

# --- Título Principal ---
st.markdown("<h1 style='text-align: center; font-size: 3rem;'>Simulador de Viabilidade: Gado vs. Selic</h1>", unsafe_allow_html=True)

# --- Layout da Aplicação ---
col_main, col_logos = st.columns([3, 1])

# --- Barra Lateral com os Parâmetros ---
st.sidebar.header("Parâmetros da Simulação")
initial_investment = st.sidebar.number_input("Investimento Inicial (R$)", min_value=1.0, value=100000.0, step=1000.0)
cost_of_capital = st.sidebar.number_input("Custo de Capital (% ao ano)", min_value=0.0, value=0.0, step=0.1)
num_heads_bought = st.sidebar.number_input("Quantidade de Cabeças Comprada", min_value=1, value=50, step=1)
period_months = st.sidebar.slider("Período de Análise (meses)", 1, 48, 12)
buy_arroba_price = st.sidebar.number_input("Preço da Arroba na Compra (R$/arroba)", min_value=1.0, value=250.00, step=0.01)
sell_arroba_price = st.sidebar.number_input("Preço da Arroba na Venda (R$/arroba)", min_value=1.0, value=280.00, step=0.01)
cost_per_head_monthly = st.sidebar.number_input("Custo Mensal por Cabeça (Outros) (R$)", min_value=0.0, value=80.0, step=5.0)
fixed_costs = st.sidebar.number_input("Outros Custos Fixos no Período (R$)", min_value=0.0, value=5000.0, step=100.0)

st.sidebar.header("Parâmetros da Aplicação Selic")
selic_rate = st.sidebar.slider("Taxa Selic (% ao ano)", 0.0, 20.0, 10.5, 0.25)
ir_rate_options = [22.5, 20.0, 17.5, 15.0]
ir_rate = st.sidebar.selectbox("Alíquota de IR na Aplicação (%)", options=ir_rate_options, index=2)

# --- Conteúdo Principal (Gráficos e Tabela) ---
with col_main:
    df_export = load_data('exportacao-kg.csv')
    df_prices = load_data('arroba-sp-historico.csv')

    if df_export is not None and df_prices is not None:
        try:
            # --- PREPARAÇÃO DOS DADOS ---
            required_cols_export = ['periodo', 'kg_liquido', 'preco_brl']
            for col in required_cols_export:
                if col not in df_export.columns:
                    raise KeyError(f"Coluna '{col}' não encontrada em 'exportacao-kg.csv'.")
            
            required_cols_prices = ['data', 'preco_brl_arroba', 'preco_bezerro_brl', 'preco_soja_brl', 'preco_milho_brl', 'Boi com 20@/Garrote', 'Boi com 20@/soja', 'Boi com 20@/milho']
            for col in required_cols_prices:
                if col not in df_prices.columns:
                        raise KeyError(f"Coluna '{col}' não encontrada em 'arroba-sp-historico.csv'.")

            df_export['Data'] = pd.to_datetime(df_export['periodo'], format='%d/%m/%Y')
            df_prices['Data'] = pd.to_datetime(df_prices['data'], format='%d/%m/%Y')
            
            df_merged = pd.merge(df_export, df_prices, on="Data", how="inner")
            
            numeric_cols = ['kg_liquido', 'preco_brl_arroba', 'preco_bezerro_brl', 'preco_soja_brl', 'preco_milho_brl', 'Boi com 20@/Garrote', 'Boi com 20@/soja', 'Boi com 20@/milho']
            for col in numeric_cols:
                df_merged[col] = pd.to_numeric(df_merged[col].astype(str).str.replace(',', '.'), errors='coerce')
            df_merged.dropna(subset=numeric_cols, inplace=True)
            df_merged['kg_liquido'] = df_merged['kg_liquido'].astype(int)

            # --- SEÇÃO 1: ANÁLISE ESTRATÉGICA DE MERCADO ---
            st.markdown("---")
            st.markdown("### 1. Análise Estratégica de Mercado")

            # GRÁFICO 1: EXPORTAÇÃO VS. PREÇO DA ARROBA
            df_merged['KG'] = df_merged['kg_liquido']
            df_merged['KG_Anterior'] = df_merged.groupby(df_merged['Data'].dt.month)['KG'].shift(1)
            conditions_export = [df_merged['KG'] > df_merged['KG_Anterior'], df_merged['KG'] < df_merged['KG_Anterior']]
            choices_export = ['green', 'red']
            colors_export = np.select(conditions_export, choices_export, default='#1f77b4').tolist()
            fig_export = make_subplots(specs=[[{"secondary_y": True}]])
            fig_export.add_trace(go.Bar(x=df_merged['Data'], y=df_merged['KG'], name='KG Exportado', marker_color=colors_export), secondary_y=False)
            fig_export.add_trace(go.Scatter(x=df_merged['Data'], y=df_merged['preco_brl_arroba'], name='Preço Arroba (R$)', mode='lines', line=dict(color='orange')), secondary_y=True)
            fig_export.update_layout(title_text='Exportação Mensal (KG) vs. Preço da Arroba (R$)', plot_bgcolor='rgba(17,17,17,0.9)', paper_bgcolor='rgba(17,17,17,0.9)', font_color="white", title_x=0.5, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            fig_export.update_yaxes(title_text="<b>Quantidade Exportada</b> (KG)", secondary_y=False)
            fig_export.update_yaxes(title_text="<b>Preço da Arroba</b> (R$)", secondary_y=True, color="orange")
            st.plotly_chart(fig_export, use_container_width=True)

            # GRÁFICO DE SAZONALIDADE
            st.markdown("---")
            df_merged['mes'] = df_merged['Data'].dt.month
            # Calcula um custo base para a sazonalidade, usando os parâmetros padrão
            custo_producao_base = calcular_custo_alimentacao(df_merged, 11.0, 80, 6.5, 85.0) + df_merged['preco_bezerro_brl']
            df_merged['custo_producao_base'] = custo_producao_base
            df_sazonal = df_merged.groupby('mes')[['preco_brl_arroba', 'custo_producao_base']].mean().reset_index()
            meses = {1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun', 7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'}
            df_sazonal['mes_nome'] = df_sazonal['mes'].map(meses)
            fig_sazonal = make_subplots(specs=[[{"secondary_y": True}]])
            fig_sazonal.add_trace(go.Bar(x=df_sazonal['mes_nome'], y=df_sazonal['custo_producao_base'], name='Custo Médio de Produção'), secondary_y=False)
            fig_sazonal.add_trace(go.Scatter(x=df_sazonal['mes_nome'], y=df_sazonal['preco_brl_arroba'], name='Preço Médio da Arroba', mode='lines', line=dict(color='yellow')), secondary_y=True)
            fig_sazonal.update_layout(title_text='Análise de Sazonalidade Média (2015-Presente)', plot_bgcolor='rgba(17,17,17,0.9)', paper_bgcolor='rgba(17,17,17,0.9)', font_color="white", title_x=0.5, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            fig_sazonal.update_yaxes(title_text="<b>Custo Médio de Produção</b> (R$)", secondary_y=False)
            fig_sazonal.update_yaxes(title_text="<b>Preço Médio da Arroba</b> (R$)", secondary_y=True, color="yellow")
            st.plotly_chart(fig_sazonal, use_container_width=True)

            # --- SEÇÃO 2: SIMULAÇÃO DE CUSTO E VIABILIDADE ---
            st.markdown("---")
            st.markdown("### 2. Simulação de Custo e Viabilidade")
            
            col_diet1, col_diet2 = st.columns([1, 2])
            with col_diet1:
                arrobas_gain_head = st.number_input("Ganho de Arrobas por Cabeça", min_value=1.0, value=11.0, step=0.5, help="Quantas arrobas cada animal deve ganhar no período.")
            with col_diet2:
                percent_milho = st.slider("Índice de Alimentação (% Milho na ração)", min_value=0, max_value=100, value=80, step=5, help="A composição do concentrado entre milho e farelo de soja.")
                st.info(f"**Composição do concentrado:** {percent_milho}% Milho / {100-percent_milho}% Farelo de Soja", icon="🌽")

            st.markdown("##### Premissas de Eficiência da Dieta")
            col_eff1, col_eff2 = st.columns(2)
            with col_eff1:
                conversao_alimentar_input = st.number_input("Conversão Alimentar (kg MS / kg PV)", min_value=4.0, value=6.5, step=0.1, help="Kg de Matéria Seca (MS) necessários para ganhar 1 kg de Peso Vivo (PV). Menor = Mais eficiente.")
            with col_eff2:
                pct_concentrado_input = st.number_input("% de Concentrado na Dieta", min_value=50.0, max_value=100.0, value=85.0, step=1.0, help="Percentual da dieta total que é composta por concentrado (milho/soja).")

            custo_alimentacao = calcular_custo_alimentacao(df_merged, arrobas_gain_head, percent_milho, conversao_alimentar_input, pct_concentrado_input)
            df_merged['custo_producao'] = custo_alimentacao + df_merged['preco_bezerro_brl']
            
            PESO_FINAL_EM_ARROBAS = 28
            df_merged['receita_por_cabeca'] = df_merged['preco_brl_arroba'] * PESO_FINAL_EM_ARROBAS
            
            df_merged['margem_bruta'] = df_merged['receita_por_cabeca'] - df_merged['custo_producao']
            colors_custo = ['#DC143C' if val < 0 else '#1f77b4' for val in df_merged['margem_bruta']]

            fig_custo = go.Figure()
            fig_custo.add_trace(go.Bar(x=df_merged['Data'], y=df_merged['custo_producao'], name='Custo por Cabeça (R$)', marker_color=colors_custo))
            fig_custo.add_trace(go.Scatter(x=df_merged['Data'], y=df_merged['receita_por_cabeca'], name=f'Receita por Cabeça ({PESO_FINAL_EM_ARROBAS}@) (R$)', mode='lines', line=dict(color='yellow')))
            
            fig_custo.update_layout(title_text='Custo de Produção vs. Receita Estimada por Cabeça', plot_bgcolor='rgba(17,17,17,0.9)', paper_bgcolor='rgba(17,17,17,0.9)', font_color="white", title_x=0.5, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            fig_custo.update_yaxes(title_text="<b>Valor por Cabeça</b> (R$)")
            st.plotly_chart(fig_custo, use_container_width=True)

            # --- SEÇÃO 3: FERRAMENTAS DE TIMING ---
            st.markdown("---")
            st.markdown("### 3. Ferramentas de Timing (Compra e Venda)")

            df_merged['Bezerro_Anterior'] = df_merged.groupby(df_merged['Data'].dt.month)['preco_bezerro_brl'].shift(1)
            conditions_bezerro = [df_merged['preco_bezerro_brl'] < df_merged['Bezerro_Anterior']]
            choices_bezerro = ['pink']
            colors_bezerro = np.select(conditions_bezerro, choices_bezerro, default='#1f77b4').tolist()
            fig_bezerro = make_subplots(specs=[[{"secondary_y": True}]])
            fig_bezerro.add_trace(go.Bar(x=df_merged['Data'], y=df_merged['preco_bezerro_brl'], name='Preço Bezerro (R$)', marker_color=colors_bezerro), secondary_y=False)
            fig_bezerro.add_trace(go.Scatter(x=df_merged['Data'], y=df_merged['Boi com 20@/Garrote'], name='Boi com 20@/Garrote', mode='lines', line=dict(color='yellow')), secondary_y=True)
            media_relacao_troca = df_merged['Boi com 20@/Garrote'].mean()
            fig_bezerro.add_hline(y=media_relacao_troca, line_dash="dash", line_color="white", annotation_text=f"Média Histórica ({media_relacao_troca:.2f})", annotation_position="bottom right", secondary_y=True)
            fig_bezerro.update_layout(title_text='Sinal de Compra: Relação de Troca (Boi Gordo vs. Bezerro)', plot_bgcolor='rgba(17,17,17,0.9)', paper_bgcolor='rgba(17,17,17,0.9)', font_color="white", title_x=0.5, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            fig_bezerro.update_yaxes(title_text="<b>Preço do Bezerro</b> (R$)", secondary_y=False)
            fig_bezerro.update_yaxes(title_text="<b>Relação de Troca</b>", secondary_y=True)
            st.plotly_chart(fig_bezerro, use_container_width=True)

            st.markdown("---")
            col_std1, col_std2 = st.columns(2)
            with col_std1:
                periodo_media = st.slider("Período da Média Móvel (meses)", min_value=3, max_value=24, value=12, help="Janela de cálculo para a média e desvio padrão.")
            with col_std2:
                num_desvios = st.slider("Número de Desvios Padrão", min_value=1.0, max_value=3.0, value=2.0, step=0.5, help="Define a largura das bandas. 2.0 é o padrão de mercado.")

            df_merged['margem_media_movel'] = df_merged['margem_bruta'].rolling(window=periodo_media).mean()
            df_merged['margem_desvio_padrao'] = df_merged['margem_bruta'].rolling(window=periodo_media).std()
            df_merged['banda_superior'] = df_merged['margem_media_movel'] + (df_merged['margem_desvio_padrao'] * num_desvios)
            df_merged['banda_inferior'] = df_merged['margem_media_movel'] - (df_merged['margem_desvio_padrao'] * num_desvios)

            fig_reversao = go.Figure()
            fig_reversao.add_trace(go.Scatter(x=df_merged['Data'], y=df_merged['banda_superior'], mode='lines', line=dict(color='rgba(255,255,255,0)'), showlegend=False))
            fig_reversao.add_trace(go.Scatter(x=df_merged['Data'], y=df_merged['banda_inferior'], mode='lines', line=dict(color='rgba(255,255,255,0)'), name='Bandas de Desvio Padrão', fill='tonexty', fillcolor='rgba(255, 255, 255, 0.1)'))
            fig_reversao.add_trace(go.Scatter(x=df_merged['Data'], y=df_merged['margem_media_movel'], name='Média Móvel da Margem', mode='lines', line=dict(color='orange', dash='dash')))
            colors_margin_reversao = ['green' if val >= 0 else 'red' for val in df_merged['margem_bruta']]
            fig_reversao.add_trace(go.Bar(x=df_merged['Data'], y=df_merged['margem_bruta'], name='Margem Bruta (R$)', marker_color=colors_margin_reversao))
            fig_reversao.update_layout(title_text='Sinal de Venda: Análise de Reversão à Média da Margem', plot_bgcolor='rgba(17,17,17,0.9)', paper_bgcolor='rgba(17,17,17,0.9)', font_color="white", title_x=0.5, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            fig_reversao.update_yaxes(title_text="<b>Margem por Cabeça</b> (R$)")
            st.plotly_chart(fig_reversao, use_container_width=True)

            # --- ANÁLISE DA SIMULAÇÃO (FINAL) ---
            st.markdown("---")
            st.markdown("<h3 style='text-align: center;'>Análise da Sua Simulação Específica</h3>", unsafe_allow_html=True)
            
            total_arrobas_bought = initial_investment / buy_arroba_price if buy_arroba_price > 0 else 0
            arrobas_gain_total = arrobas_gain_head * num_heads_bought
            total_arrobas_sold = total_arrobas_bought + arrobas_gain_total
            revenue = total_arrobas_sold * sell_arroba_price
            operational_cost = (cost_per_head_monthly * num_heads_bought * period_months) + fixed_costs
            capital_cost_value = initial_investment * (cost_of_capital / 100) * (period_months / 12)
            total_cost_cattle = initial_investment + operational_cost + capital_cost_value
            profit_cattle = revenue - total_cost_cattle

            selic_monthly_rate = (1 + selic_rate / 100)**(1/12) - 1
            selic_final_value = initial_investment * (1 + selic_monthly_rate)**period_months
            selic_gross_profit = selic_final_value - initial_investment
            profit_selic = selic_gross_profit * (1 - ir_rate / 100)

            cash_flow_cattle = [-initial_investment] + [0] * (period_months - 1) + [revenue - operational_cost - capital_cost_value]
            irr_monthly_cattle = npf.irr(cash_flow_cattle) if sum(cash_flow_cattle) > 0 else 0
            irr_monthly_selic = selic_monthly_rate * (1 - ir_rate / 100)

            months = np.arange(period_months + 1)
            gado_values = np.linspace(initial_investment, initial_investment + profit_cattle, len(months))
            selic_values = [initial_investment * (1 + irr_monthly_selic)**m for m in months]
            
            df_comp = pd.DataFrame({'Mês': months, 'Produção de Gado': gado_values, 'Aplicação Selic': selic_values})
            
            selic_trend = np.interp(months, [7, 19, 31, 43], [10.5, 9.5, 9.0, 8.5], left=10.5, right=8.5)
            
            fig_comp, ax1 = plt.subplots(figsize=(12, 5))
            sns.set_style("darkgrid", {"axes.facecolor": "#111111", "grid.color": "#444444"})
            ax1.set_facecolor('#111111')
            fig_comp.set_facecolor('#111111')
            
            sns.lineplot(x='Mês', y='Produção de Gado', data=df_comp, ax=ax1, label='Produção de Gado (R$)', color='orange')
            sns.lineplot(x='Mês', y='Aplicação Selic', data=df_comp, ax=ax1, label='Aplicação Selic (R$)', color='cyan')
            
            ax1.set_ylabel('Valor Acumulado (R$)', color='white')
            ax1.set_xlabel('Meses', color='white')
            ax1.set_title('Comparativo de Investimentos: Produção de Gado x Selic', fontsize=16, color='white')
            ax1.tick_params(colors='white', which='both')
            
            ax2 = ax1.twinx()
            sns.lineplot(x=months, y=selic_trend, ax=ax2, color='red', linestyle='--', label='Taxa Selic Projetada (%)')
            ax2.set_ylabel('Taxa Selic (% ao ano)', color='white')
            ax2.tick_params(colors='white', which='both')
            
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            legend = ax2.legend(lines + lines2, labels + labels2, loc='upper left', frameon=True)
            legend.get_frame().set_facecolor('#333333')
            for text in legend.get_texts():
                text.set_color("white")
                
            ax1.get_legend().remove()
            
            st.pyplot(fig_comp)
            
            st.markdown("---")

            st.subheader("Resumo da Operação")
            
            detalhes_gado = f"""
            | Métrica | Valor |
            |---|---|
            | Ganho com Aumento de Arrobas (R$) | R$ {arrobas_gain_total * sell_arroba_price:,.2f} |
            | Ganho/Perda com Diferença de Preço (R$) | R$ {(sell_arroba_price - buy_arroba_price) * total_arrobas_bought:,.2f} |
            | Custo de Capital (Juros) (R$) | - R$ {capital_cost_value:,.2f} |
            | Custo Operacional Total (R$) | - R$ {operational_cost:,.2f} |
            """

            resumo_data = {
                'Métrica': [
                    'Lucro Total com Produção de Gado (R$)',
                    'Lucro Total com Selic (R$)',
                    'TIR Mensal - Gado (%)',
                    'TIR Mensal - Selic (%)',
                    '--- Detalhamento do Gado ---'
                ],
                'Valor': [
                    f"R$ {profit_cattle:,.2f}",
                    f"R$ {profit_selic:,.2f}",
                    f"{irr_monthly_cattle * 100:.2f}%" if irr_monthly_cattle is not None else "N/A",
                    f"{irr_monthly_selic * 100:.2f}%",
                    ''
                ]
            }
            
            st.table(resumo_data)
            st.markdown(detalhes_gado)

        except KeyError as e:
            st.error(f"Erro de processamento: {e}. Verifique se os nomes das colunas nos seus ficheiros CSV estão corretos.")
        except Exception as e:
            st.error(f"Ocorreu um erro inesperado ao processar os dados históricos: {e}")

# --- Coluna das Logos ---
with col_logos:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    display_linked_image("assets/oscapital.jpeg", "https://oscapitaloficial.com.br/", "<p style='text-align: center;'>VISITE NOSSO SITE</p>", 200)
    st.markdown("<br>", unsafe_allow_html=True)
    display_linked_image("assets/IB_logo_stacked1.jpg", "https://ibkr.com/referral/edgleison239", "<p style='text-align: center;'>INVISTA EM MAIS DE 160 MERCADOS</p>", 200)