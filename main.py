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
from PIL import Image

# --- Configuração da Página ---
st.set_page_config(layout="wide")

# --- Define o caminho base do projeto de forma segura ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Funções Auxiliares para Imagens e Gráficos ---

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
            if caption:
                st.markdown(caption, unsafe_allow_html=True)

def adicionar_marca_e_fonte(fig, is_matplotlib=False):
    """Adiciona uma marca d'água e um texto de fonte a um gráfico Plotly ou Matplotlib."""
    source_text = "Fonte: CEPEA, B³ / Elaborado por: OS CAPITAL."
    image_path = os.path.join(BASE_DIR, 'assets', 'oscapital.jpeg')

    if not os.path.exists(image_path):
        st.warning(f"Imagem da marca d'água não encontrada em: {image_path}")
        return fig

    if is_matplotlib:
        try:
            img = Image.open(image_path)
            width, height = fig.get_size_inches() * fig.dpi
            fig_width_px, fig_height_px = int(width), int(height)
            img_width_px, img_height_px = img.size
            x_pos = (fig_width_px - img_width_px) // 2
            y_pos = (fig_height_px - img_height_px) // 2
            fig.figimage(img, xo=x_pos, yo=y_pos, alpha=0.20, zorder=-1) 
            fig.text(0.5, 0.01, source_text, ha='center', va='bottom', fontsize=9, color='grey')
            fig.subplots_adjust(bottom=0.15)
        except Exception as e:
            st.error(f"Erro ao adicionar marca d'água no Matplotlib: {e}")
        return fig
    else: # Plotly
        try:
            fig.add_layout_image(
                dict(
                    source=Image.open(image_path),
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    sizex=0.5, sizey=0.5,
                    xanchor="center", yanchor="middle",
                    opacity=0.20,
                    layer="below"
                )
            )
            fig.add_annotation(
                text=source_text,
                xref="paper", yref="paper",
                x=0.5, y=-0.17,
                showarrow=False,
                align="center",
                font=dict(size=11, color="grey")
            )
            fig.update_layout(margin=dict(b=90))
        except Exception as e:
            st.error(f"Erro ao adicionar marca d'água no Plotly: {e}")
        return fig

# --- Carregamento de Dados ---
@st.cache_data
def load_data(file_name, sep=';', decimal=','):
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
def calcular_custo_alimentacao(arrobas_a_ganhar, conversao_alimentar, pct_concentrado_dieta, composicao, precos):
    KG_POR_ARROBA, RENDIMENTO_CARCACA = 15, 0.55
    MS_MILHO, MS_SOJA, MS_NUCLEO, MS_OUTROS = 0.88, 0.89, 0.95, 0.90
    KG_SACA_MILHO, KG_SACA_SOJA = 60, 50

    if arrobas_a_ganhar <= 0:
        is_series = any(isinstance(p, pd.Series) for p in precos.values())
        return pd.Series([0] * len(next(p for p in precos.values() if isinstance(p, pd.Series)))) if is_series else 0

    kg_carcaca_a_ganhar = arrobas_a_ganhar * KG_POR_ARROBA
    kg_peso_vivo_a_ganhar = kg_carcaca_a_ganhar / RENDIMENTO_CARCACA
    consumo_total_ms = kg_peso_vivo_a_ganhar * conversao_alimentar
    total_concentrado_ms = consumo_total_ms * (pct_concentrado_dieta / 100)
    
    custo_total = 0
    soma_composicao = sum(filter(None, composicao.values()))
    if soma_composicao == 0: return 0

    if composicao.get('milho', 0) > 0 and 'milho_saca' in precos:
        milho_ms = total_concentrado_ms * (composicao['milho'] / soma_composicao)
        consumo_total_milho_kg = milho_ms / MS_MILHO
        sacos_de_milho = consumo_total_milho_kg / KG_SACA_MILHO
        custo_total += precos['milho_saca'] * sacos_de_milho
    if composicao.get('soja', 0) > 0 and 'soja_saca' in precos:
        soja_ms = total_concentrado_ms * (composicao['soja'] / soma_composicao)
        consumo_total_soja_kg = soja_ms / MS_SOJA
        sacos_de_soja = consumo_total_soja_kg / KG_SACA_SOJA
        custo_total += precos['soja_saca'] * sacos_de_soja
    if composicao.get('nucleo', 0) > 0 and 'nucleo_kg' in precos:
        nucleo_ms = total_concentrado_ms * (composicao['nucleo'] / soma_composicao)
        consumo_total_nucleo_kg = nucleo_ms / MS_NUCLEO
        custo_total += precos['nucleo_kg'] * consumo_total_nucleo_kg
    if composicao.get('outros', 0) > 0 and 'outros_kg' in precos:
        outros_ms = total_concentrado_ms * (composicao['outros'] / soma_composicao)
        consumo_total_outros_kg = outros_ms / MS_OUTROS
        custo_total += precos['outros_kg'] * consumo_total_outros_kg
        
    return custo_total

# --- Barra Lateral com os Parâmetros ---
st.sidebar.header("Parâmetros da Simulação")

st.sidebar.markdown("##### **1. Definições de Compra e Venda**")
initial_investment = st.sidebar.number_input("Valor Total Investido (R$)", min_value=1.0, value=250000.0, step=1000.0)
num_heads_bought = st.sidebar.number_input("Quantidade de Cabeças Comprada", min_value=1, value=41, step=1)
buy_arroba_price = st.sidebar.number_input("Preço da Arroba na Compra (R$/@)", min_value=1.0, value=230.00, step=0.01)
sell_arroba_price = st.sidebar.number_input("Preço da Arroba na Venda (R$/@)", min_value=1.0, value=250.00, step=0.01)
arrobas_gain_head = st.sidebar.number_input("Meta de Ganho por Cabeça (@)", min_value=0.0, value=7.0, step=0.5, help="Quantas arrobas cada animal deve ganhar no período.")

total_arrobas_bought = initial_investment / buy_arroba_price if buy_arroba_price > 0 else 0
arroba_media_inicial = total_arrobas_bought / num_heads_bought if num_heads_bought > 0 else 0
total_arrobas_final = total_arrobas_bought + (arrobas_gain_head * num_heads_bought)

st.sidebar.markdown(f"**Quantidade Total de Arrobas Compradas:** `{total_arrobas_bought:,.2f} @`")
st.sidebar.markdown(f"**Arroba Média por Cabeça (na compra):** `{arroba_media_inicial:,.2f} @`")
st.sidebar.markdown(f"**Total de Arrobas Alcançados (venda):** `{total_arrobas_final:,.2f} @`")
st.sidebar.markdown("---")

st.sidebar.markdown("##### **2. Custos Operacionais**")
period_months = st.sidebar.slider("Período de Análise (meses)", 1, 48, 6)
fixed_costs = st.sidebar.number_input("Outros Custos Fixos no Período (R$)", min_value=0.0, value=5000.0, step=100.0, help="Use para despesas totais como aluguel, impostos, etc.")
custos_mensais_por_cabeca = st.sidebar.number_input("Custos Mensais Adicionais por Cabeça (R$)", min_value=0.0, value=0.0, step=5.0, help="Use para custos mensais por animal, como vacinas, manejo, volumoso, etc.")
cost_of_capital = st.sidebar.number_input("Custo de Oportunidade/Capital (% a.a.)", min_value=0.0, value=10.0, step=0.1, help="Taxa de juros sobre o capital investido.")
st.sidebar.markdown("---")

st.sidebar.header("Parâmetros da Aplicação Selic")
selic_rate = st.sidebar.slider("Rendimento da Aplicação Selic (% a.a.)", 0.0, 20.0, 10.25, 0.25, help="Taxa de rendimento do investimento alternativo.")
ir_rate_options = [22.5, 20.0, 17.5, 15.0]
ir_rate = st.sidebar.selectbox("Alíquota de IR na Aplicação (%)", options=ir_rate_options, index=2)

# --- CABEÇALHO ---
col_title, col_logos = st.columns([2.5, 1])
with col_title:
    st.markdown("<h1 style='text-align: left; font-size: 2.8rem; padding-top: 20px;'>Simulador de Viabilidade: Gado vs. Selic</h1>", unsafe_allow_html=True)
with col_logos:
    logo1, logo2 = st.columns(2)
    with logo1: display_linked_image("assets/oscapital.jpeg", "https://oscapitaloficial.com.br/", "", 110)
    with logo2: display_linked_image("assets/IB_logo_stacked1.jpg", "https://ibkr.com/referral/edgleison239", "", 110)

# --- CONTEÚDO PRINCIPAL ---
df_export = load_data('exportacao-kg.csv')
df_prices = load_data('arroba-sp-historico.csv')

if df_export is not None and df_prices is not None:
    try:
        # --- PREPARAÇÃO DE DADOS E CÁLCULOS BASE ---
        df_export['Data'] = pd.to_datetime(df_export['periodo'], format='%d/%m/%Y')
        df_prices['Data'] = pd.to_datetime(df_prices['data'], format='%d/%m/%Y')
        df_merged = pd.merge(df_export, df_prices, on="Data", how="inner")
        numeric_cols = ['kg_liquido', 'preco_brl_arroba', 'preco_bezerro_brl', 'preco_soja_brl', 'preco_milho_brl', 'Boi com 20@/Garrote', 'Boi com 20@/soja', 'Boi com 20@/milho']
        for col in numeric_cols:
            df_merged[col] = pd.to_numeric(df_merged[col].astype(str).str.replace(',', '.'), errors='coerce')
        df_merged.dropna(subset=numeric_cols, inplace=True)
        df_merged['kg_liquido'] = df_merged['kg_liquido'].astype(int)
        df_merged['mes'] = df_merged['Data'].dt.month
        df_merged['KG_Anterior'] = df_merged.groupby(df_merged['Data'].dt.month)['kg_liquido'].shift(1)
        
        # --- CÁLCULO DA MARGEM BASE (PARA GRÁFICOS HISTÓRICOS) ---
        composicao_base = {'milho': 80, 'soja': 20}
        precos_base_hist = {'milho_saca': df_merged['preco_milho_brl'], 'soja_saca': df_merged['preco_soja_brl']}
        custo_producao_base = calcular_custo_alimentacao(11.0, 6.5, 85.0, composicao_base, precos_base_hist) + df_merged['preco_bezerro_brl']
        df_merged['custo_producao_base'] = custo_producao_base
        df_merged['custo_producao_base_media_movel'] = df_merged['custo_producao_base'].rolling(window=12).mean()
        df_merged['receita_por_cabeca_base'] = df_merged['preco_brl_arroba'] * 18 # 7@ inicial + 11@ ganho
        df_merged['margem_bruta_base'] = df_merged['receita_por_cabeca_base'] - df_merged['custo_producao_base']

        # --- LÓGICA E GRÁFICO DO TERMÔMETRO HISTÓRICO ---
        st.markdown("---")
        st.markdown("<h3 style='text-align: center;'>Termômetro de Mercado e Histórico de Sinais</h3>", unsafe_allow_html=True)
        
        sensibilidade_sinal = st.slider("Sensibilidade do Sinal (Nº de Condições Mínimas)", min_value=3, max_value=5, value=4, help="Define o número mínimo de indicadores (de 1 a 5) que precisam apontar na mesma direção para gerar um sinal.")
        df_sazonal = df_merged.groupby('mes')[['preco_brl_arroba']].mean()
        media_sazonal_anual = df_sazonal['preco_brl_arroba'].mean()
        media_relacao_troca = df_merged['Boi com 20@/Garrote'].mean()
        periodo_media_termometro = 12
        num_desvios_termometro = 2.0
        df_merged['margem_media_movel_base'] = df_merged['margem_bruta_base'].rolling(window=periodo_media_termometro).mean()
        df_merged['margem_desvio_padrao_base'] = df_merged['margem_bruta_base'].rolling(window=periodo_media_termometro).std()
        df_merged['banda_superior_base'] = df_merged['margem_media_movel_base'] + (df_merged['margem_desvio_padrao_base'] * num_desvios_termometro)
        df_merged['banda_inferior_base'] = df_merged['margem_media_movel_base'] - (df_merged['margem_desvio_padrao_base'] * num_desvios_termometro)
        venda_1_series = df_merged['kg_liquido'] > df_merged['KG_Anterior']
        venda_2_series = df_merged['mes'].apply(lambda m: df_sazonal.loc[m, 'preco_brl_arroba'] > media_sazonal_anual)
        venda_3_series = df_merged['margem_bruta_base'] > 0
        venda_4_series = df_merged['Boi com 20@/Garrote'] < media_relacao_troca
        venda_5_series = df_merged['margem_bruta_base'] >= df_merged['banda_superior_base']
        compra_1_series = df_merged['kg_liquido'] < df_merged['KG_Anterior']
        compra_2_series = df_merged['mes'].apply(lambda m: df_sazonal.loc[m, 'preco_brl_arroba'] < media_sazonal_anual)
        compra_3_series = df_merged['custo_producao_base'] < df_merged['custo_producao_base_media_movel']
        compra_4_series = df_merged['Boi com 20@/Garrote'] > media_relacao_troca
        compra_5_series = df_merged['margem_bruta_base'] <= df_merged['banda_inferior_base']
        soma_condicoes_venda = (venda_1_series.astype(int) + venda_2_series.astype(int) + venda_3_series.astype(int) + venda_4_series.astype(int) + venda_5_series.astype(int))
        soma_condicoes_compra = (compra_1_series.astype(int) + compra_2_series.astype(int) + compra_3_series.astype(int) + compra_4_series.astype(int) + compra_5_series.astype(int))
        sinal_venda_total = soma_condicoes_venda >= sensibilidade_sinal
        sinal_compra_total = soma_condicoes_compra >= sensibilidade_sinal
        df_merged['sinal_confluencia'] = np.select([sinal_venda_total, sinal_compra_total], [-1, 1], default=0)
        latest_data = df_merged.iloc[-1]
        last_month_name = latest_data['Data'].strftime("%B de %Y")
        num_condicoes_venda_recente = int(soma_condicoes_venda.iloc[-1])
        num_condicoes_compra_recente = int(soma_condicoes_compra.iloc[-1])
        if latest_data['sinal_confluencia'] == -1:
            st.warning(f"SINAL DE VENDA ({num_condicoes_venda_recente}/5) - Confluência de Alta em {last_month_name}", icon="📈")
        elif latest_data['sinal_confluencia'] == 1:
            st.success(f"SINAL DE COMPRA ({num_condicoes_compra_recente}/5) - Confluência de Baixa em {last_month_name}", icon="📉")
        else:
            st.info(f"MERCADO MISTO OU NEUTRO - Sem Confluência de Sinais em {last_month_name} (Compra: {num_condicoes_compra_recente}/5, Venda: {num_condicoes_venda_recente}/5)", icon="📊")
        
        with st.expander("Ver explicação do Gráfico de Histórico de Sinais"):
            st.markdown(f"""
            Este gráfico é o **backtest visual** do Termômetro de Mercado. Ele plota os sinais de confluência sobre o gráfico de margem bruta para que você possa avaliar seu desempenho histórico.
            - **Sensibilidade Atual:** Um sinal é gerado quando pelo menos **{sensibilidade_sinal} de 5 indicadores** apontam na mesma direção.
            - **Sinal de Compra (▲ Verde):** Marcado abaixo da margem, aparece em meses onde o critério de sensibilidade para compra foi atingido.
            - **Sinal de Venda (▼ Vermelho):** Marcado acima da margem, aparece em meses onde o critério de sensibilidade para venda foi atingido.
            - **Barras Cinzas:** Representam a margem de lucro (receita - custo) em cada mês, fornecendo o contexto para os sinais.
            """)

        df_sinais_compra = df_merged[df_merged['sinal_confluencia'] == 1]
        df_sinais_venda = df_merged[df_merged['sinal_confluencia'] == -1]
        fig_sinais_hist = go.Figure()
        fig_sinais_hist.add_trace(go.Bar(x=df_merged['Data'], y=df_merged['margem_bruta_base'], name='Margem Bruta (R$)', marker_color='grey'))
        fig_sinais_hist.add_trace(go.Scatter(x=df_sinais_compra['Data'], y=df_sinais_compra['margem_bruta_base'] - 200, mode='markers', name='Sinal de Compra', marker=dict(symbol='triangle-up', color='green', size=12)))
        fig_sinais_hist.add_trace(go.Scatter(x=df_sinais_venda['Data'], y=df_sinais_venda['margem_bruta_base'] + 200, mode='markers', name='Sinal de Venda', marker=dict(symbol='triangle-down', color='red', size=12)))
        fig_sinais_hist.update_layout(title_text='Backtest Visual: Sinais de Confluência vs. Margem de Lucro', plot_bgcolor='rgba(17,17,17,0.9)', paper_bgcolor='rgba(17,17,17,0.9)', font_color="white", title_x=0.5, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig_sinais_hist.update_yaxes(title_text="<b>Margem por Cabeça</b> (R$)")
        fig_sinais_hist = adicionar_marca_e_fonte(fig_sinais_hist)
        st.plotly_chart(fig_sinais_hist, use_container_width=True)

        # --- SEÇÃO 1: ANÁLISE ESTRATÉGICA DE MERCADO ---
        st.markdown("---")
        st.markdown("### 1. Análise Estratégica de Mercado")
        with st.expander("Ver explicação do Gráfico de Exportação"):
            st.markdown("""
            Este gráfico contextualiza o mercado, mostrando a relação entre o **volume de carne bovina exportada pelo Brasil (barras)** e o **preço da arroba no mercado interno (linha amarela)**.
            - **Interpretação:** Geralmente, altos volumes de exportação podem aumentar a demanda total por gado, exercendo pressão de alta sobre os preços internos.
            - **Cores:** As barras verdes indicam que a exportação do mês foi **maior** que a do mesmo mês no ano anterior, enquanto as vermelhas indicam que foi **menor**.
            """)
        conditions_export = [df_merged['kg_liquido'] > df_merged['KG_Anterior'], df_merged['kg_liquido'] < df_merged['KG_Anterior']]
        choices_export = ['green', 'red']
        colors_export = np.select(conditions_export, choices_export, default='#1f77b4').tolist()
        fig_export = make_subplots(specs=[[{"secondary_y": True}]])
        fig_export.add_trace(go.Bar(x=df_merged['Data'], y=df_merged['kg_liquido'], name='KG Exportado', marker_color=colors_export), secondary_y=False)
        fig_export.add_trace(go.Scatter(x=df_merged['Data'], y=df_merged['preco_brl_arroba'], name='Preço Arroba (R$)', mode='lines', line=dict(color='orange')), secondary_y=True)
        fig_export.update_layout(title_text='Exportação Mensal (KG) vs. Preço da Arroba (R$)', plot_bgcolor='rgba(17,17,17,0.9)', paper_bgcolor='rgba(17,17,17,0.9)', font_color="white", title_x=0.5, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig_export.update_yaxes(title_text="<b>Quantidade Exportada</b> (KG)", secondary_y=False)
        fig_export.update_yaxes(title_text="<b>Preço da Arroba</b> (R$)", secondary_y=True, color="orange")
        fig_export = adicionar_marca_e_fonte(fig_export)
        st.plotly_chart(fig_export, use_container_width=True)

        st.markdown("---");
        with st.expander("Ver explicação do Gráfico de Sazonalidade"):
            st.markdown("""
            Este gráfico ajuda no **planejamento de longo prazo**, mostrando o comportamento médio dos custos e preços para cada mês do ano (de 2015 até hoje).
            - **Interpretação:** Ajuda a identificar padrões sazonais, como a **entressafra** (tipicamente no segundo semestre), onde os preços da arroba (linha amarela) historicamente tendem a ser mais altos.
            - **Estratégia:** O ideal é planejar o ciclo de engorda para que a venda dos animais coincida com os meses de preços historicamente mais altos.
            """)
        meses = {1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun', 7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'}
        df_sazonal_plot = df_merged.groupby('mes')[['preco_brl_arroba', 'custo_producao_base']].mean().reset_index()
        df_sazonal_plot['mes_nome'] = df_sazonal_plot['mes'].map(meses)
        fig_sazonal = make_subplots(specs=[[{"secondary_y": True}]])
        fig_sazonal.add_trace(go.Bar(x=df_sazonal_plot['mes_nome'], y=df_sazonal_plot['custo_producao_base'], name='Custo Médio de Produção'), secondary_y=False)
        fig_sazonal.add_trace(go.Scatter(x=df_sazonal_plot['mes_nome'], y=df_sazonal_plot['preco_brl_arroba'], name='Preço Médio da Arroba', mode='lines', line=dict(color='yellow')), secondary_y=True)
        fig_sazonal.update_layout(title_text='Análise de Sazonalidade Média (2015-Presente)', plot_bgcolor='rgba(17,17,17,0.9)', paper_bgcolor='rgba(17,17,17,0.9)', font_color="white", title_x=0.5, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig_sazonal.update_yaxes(title_text="<b>Custo Médio de Produção</b> (R$)", secondary_y=False)
        fig_sazonal.update_yaxes(title_text="<b>Preço Médio da Arroba</b> (R$)", secondary_y=True, color="yellow")
        fig_sazonal = adicionar_marca_e_fonte(fig_sazonal)
        st.plotly_chart(fig_sazonal, use_container_width=True)

        # --- SEÇÃO 3: FERRAMENTAS DE TIMING ---
        st.markdown("---")
        st.markdown("### 3. Ferramentas de Timing (Compra e Venda)")
        with st.expander("Ver explicação do Gráfico de Relação de Troca (Sinal de Compra)"):
            st.markdown("""
            Esta é uma ferramenta para identificar bons momentos de **compra** de bezerros para reposição.
            - **Interpretação:** A linha amarela mostra a relação de troca, ou seja, **quantos bezerros podem ser comprados com a venda de um Boi de 20 arrobas**.
            - **Sinal de Compra:** Quando a linha está **acima da média histórica** (tracejada), o poder de compra de bezerros está alto, indicando um momento favorável para adquirir animais de reposição. Mais bezerros por boi = melhor para o comprador.
            - **Cores das Barras:** As barras rosas indicam meses onde o preço do bezerro foi **menor que no mesmo mês do ano anterior**, sugerindo uma melhora no custo de aquisição em uma base anual.
            """)
        df_merged['Bezerro_Anterior'] = df_merged.groupby(df_merged['Data'].dt.month)['preco_bezerro_brl'].shift(1)
        conditions_bezerro = [df_merged['preco_bezerro_brl'] < df_merged['Bezerro_Anterior']]
        choices_bezerro = ['pink']
        colors_bezerro = np.select(conditions_bezerro, choices_bezerro, default='#1f77b4').tolist()
        fig_bezerro = make_subplots(specs=[[{"secondary_y": True}]])
        fig_bezerro.add_trace(go.Bar(x=df_merged['Data'], y=df_merged['preco_bezerro_brl'], name='Preço Bezerro (R$)', marker_color=colors_bezerro), secondary_y=False)
        fig_bezerro.add_trace(go.Scatter(x=df_merged['Data'], y=df_merged['Boi com 20@/Garrote'], name='Boi com 20@/Bezerro', mode='lines', line=dict(color='yellow')), secondary_y=True)
        fig_bezerro.add_hline(y=media_relacao_troca, line_dash="dash", line_color="white", annotation_text=f"Média Histórica ({media_relacao_troca:.2f})", annotation_position="bottom right", secondary_y=True)
        fig_bezerro.update_layout(title_text='Sinal de Compra: Relação de Troca (Boi Gordo vs. Bezerro)', plot_bgcolor='rgba(17,17,17,0.9)', paper_bgcolor='rgba(17,17,17,0.9)', font_color="white", title_x=0.5, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig_bezerro.update_yaxes(title_text="<b>Preço do Bezerro</b> (R$)", secondary_y=False)
        fig_bezerro.update_yaxes(title_text="<b>Relação de Troca</b>", secondary_y=True)
        fig_bezerro = adicionar_marca_e_fonte(fig_bezerro)
        st.plotly_chart(fig_bezerro, use_container_width=True)

        st.markdown("---")
        with st.expander("Ver explicação do Gráfico de Reversão da Margem (Sinais de Compra e Venda)"):
            st.markdown("""
            Esta é uma ferramenta estatística para identificar momentos de **compra (risco/oportunidade)** e **venda (picos de euforia)**.
            - **Sinal de Venda (Ciano):** Quando as barras de lucro tocam ou **ultrapassam a banda superior**, a margem está estatisticamente "esticada". Isso sugere um pico de lucratividade, representando um momento oportuno para vender.
            - **Sinal de Compra (Amarelo):** Quando as barras tocam ou **caem abaixo da banda inferior**, a margem está historicamente "comprimida" ou "barata". Para um investidor de perfil contrário, isso pode sinalizar um ponto de pessimismo máximo, que historicamente precede uma recuperação.
            - **Cores:** **Verde** para lucro normal, **Vermelho** para prejuízo normal (dentro das bandas).
            """)
        col_std1, col_std2 = st.columns(2)
        with col_std1:
            periodo_media_reversao = st.slider("Período da Média Móvel (meses)", min_value=3, max_value=24, value=12, help="Janela de cálculo para a média e desvio padrão.")
        with col_std2:
            num_desvios_reversao = st.slider("Número de Desvios Padrão", min_value=1.0, max_value=3.0, value=2.0, step=0.5, help="Define a largura das bandas. 2.0 é o padrão de mercado.")
        
        df_merged['margem_media_movel_reversao'] = df_merged['margem_bruta_base'].rolling(window=periodo_media_reversao).mean()
        df_merged['margem_desvio_padrao_reversao'] = df_merged['margem_bruta_base'].rolling(window=periodo_media_reversao).std()
        df_merged['banda_superior_reversao'] = df_merged['margem_media_movel_reversao'] + (df_merged['margem_desvio_padrao_reversao'] * num_desvios_reversao)
        df_merged['banda_inferior_reversao'] = df_merged['margem_media_movel_reversao'] - (df_merged['margem_desvio_padrao_reversao'] * num_desvios_reversao)
        
        colors_margin_reversao = []
        for i in range(len(df_merged)):
            margem = df_merged['margem_bruta_base'].iloc[i]
            superior = df_merged['banda_superior_reversao'].iloc[i]
            inferior = df_merged['banda_inferior_reversao'].iloc[i]
            if margem >= superior: colors_margin_reversao.append('cyan')
            elif margem <= inferior: colors_margin_reversao.append('yellow')
            elif margem < 0: colors_margin_reversao.append('red')
            else: colors_margin_reversao.append('green')

        fig_reversao = go.Figure()
        fig_reversao.add_trace(go.Scatter(x=df_merged['Data'], y=df_merged['banda_superior_reversao'], mode='lines', line=dict(color='rgba(255,255,255,0)'), showlegend=False))
        fig_reversao.add_trace(go.Scatter(x=df_merged['Data'], y=df_merged['banda_inferior_reversao'], mode='lines', line=dict(color='rgba(255,255,255,0)'), name='Bandas de Desvio Padrão', fill='tonexty', fillcolor='rgba(255, 255, 255, 0.1)'))
        fig_reversao.add_trace(go.Scatter(x=df_merged['Data'], y=df_merged['margem_media_movel_reversao'], name='Média Móvel da Margem', mode='lines', line=dict(color='orange', dash='dash')))
        fig_reversao.add_trace(go.Bar(x=df_merged['Data'], y=df_merged['margem_bruta_base'], name='Margem Bruta (R$)', marker_color=colors_margin_reversao))
        fig_reversao.update_layout(title_text='Sinais de Compra e Venda por Reversão à Média da Margem', plot_bgcolor='rgba(17,17,17,0.9)', paper_bgcolor='rgba(17,17,17,0.9)', font_color="white", title_x=0.5, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig_reversao.update_yaxes(title_text="<b>Margem por Cabeça</b> (R$)")
        fig_reversao = adicionar_marca_e_fonte(fig_reversao)
        st.plotly_chart(fig_reversao, use_container_width=True)

        # --- SEÇÃO 2: SIMULAÇÃO DE CUSTO E VIABILIDADE ---
        st.markdown("---")
        st.markdown("### 2. Simulação de Custo e Viabilidade")
        
        st.markdown("##### Parâmetros da Dieta")
        st.markdown("###### Composição do Concentrado (%)")
        col_comp1, col_comp2, col_comp3, col_comp4 = st.columns(4)
        with col_comp1: percent_milho = st.number_input("% Milho", min_value=0, max_value=100, value=65, step=1)
        with col_comp2: percent_soja = st.number_input("% Soja", min_value=0, max_value=100, value=30, step=1)
        with col_comp3: percent_nucleo = st.number_input("% Núcleo", min_value=0, max_value=100, value=5, step=1)
        with col_comp4: percent_outros = st.number_input("% Outros", min_value=0, max_value=100, value=0, step=1)

        soma_percentuais = percent_milho + percent_soja + percent_nucleo + percent_outros
        if soma_percentuais != 100:
            st.warning(f"A soma dos percentuais da dieta é {soma_percentuais}%. O ideal é que seja 100%.", icon="⚠️")
        
        st.info(f"**Composição atual:** {percent_milho}% Milho / {percent_soja}% Soja / {percent_nucleo}% Núcleo / {percent_outros}% Outros", icon="🌽")
        
        st.markdown("###### Parâmetros de Eficiência")
        col_eff1, col_eff2 = st.columns(2)
        with col_eff1:
            conversao_alimentar_base = st.slider("Conversão Alimentar Base (kg MS / kg PV)", min_value=5.0, max_value=12.0, value=7.0, step=0.1, help="Quanto menor o valor, mais eficiente é o animal. Valores entre 6-7 são ótimos.")
            if conversao_alimentar_base <= 7.0:
                st.success("✅ Eficiência Ótima: Cenário de alta lucratividade.")
            elif 7.0 < conversao_alimentar_base <= 8.5:
                st.info("ℹ️ Eficiência Média: Cenário realista, atenção aos custos.")
            else:
                st.warning("⚠️ Baixa Eficiência: Cenário de alto risco e provável prejuízo.")
        with col_eff2:
            pct_concentrado_input = st.number_input("% de Concentrado na Dieta", min_value=50.0, max_value=100.0, value=85.0, step=1.0, help="Percentual da dieta total que é composta por concentrado.")
        
        fator_qualidade_dieta = st.slider("Ajuste Fino por Qualidade/Tecnologia da Dieta (%)", min_value=-10, max_value=10, value=0, step=1, help="Use para simular o impacto de uma dieta mais (valor > 0) ou menos (valor < 0) energética/tecnológica na eficiência.")
        conversao_ajustada = conversao_alimentar_base * (1 - (fator_qualidade_dieta / 100))
        st.info(f"Considerando o ajuste de qualidade, a conversão final usada nos cálculos é: **{conversao_ajustada:.2f} kg MS / kg PV**")

        with st.expander("Não sabe a conversão em Matéria Seca (MS)? Clique aqui para calcular"):
            st.markdown("Use esta calculadora se você sabe a conversão da sua ração 'no cocho' (matéria natural).")
            conv_no_cocho = st.number_input("Conversão 'no cocho' (kg Ração / kg PV)", value=8.0, step=0.1)
            ms_media_dieta = st.number_input("Teor de Matéria Seca (%) da sua dieta total", value=85.0, step=0.5, help="Uma dieta de confinamento com silagem fica em torno de 60-70%. Dietas com pouca silagem, 80-90%.")
            conv_em_ms = conv_no_cocho * (ms_media_dieta / 100)
            st.success(f"O valor a ser inserido no slider 'Conversão Alimentar Base' acima é: {conv_em_ms:.2f}")
        
        with st.expander("Quer estimar o Ganho de Peso com base no Consumo? Clique aqui"):
            st.markdown("Use esta ferramenta para ter uma ideia do ganho de peso com base no consumo diário planejado.")
            consumo_diario_ms = st.number_input("Consumo Diário de Matéria Seca (kg MS/dia)", value=7.5, step=0.1)
            if conversao_ajustada > 0:
                gpv_estimado = consumo_diario_ms / conversao_ajustada
                ganho_total_pv = gpv_estimado * (period_months * 30)
                ganho_total_carcaca = ganho_total_pv * 0.55
                ganho_em_arrobas = ganho_total_carcaca / 15
                st.success(f"Com este consumo e eficiência, o ganho estimado em {period_months} meses é de **{ganho_em_arrobas:.2f} arrobas**.")
            else:
                st.warning("A conversão alimentar deve ser maior que zero.")


        st.markdown("##### Preços dos Insumos para a Simulação Final")
        latest_prices = df_merged.tail(1)
        default_milho = latest_prices['preco_milho_brl'].iloc[0] if not latest_prices.empty else 60.00
        default_soja = latest_prices['preco_soja_brl'].iloc[0] if not latest_prices.empty else 130.00

        col_preco1, col_preco2 = st.columns(2)
        with col_preco1:
            preco_milho_input = st.number_input("Preço da Saca de Milho (60kg) (R$)", min_value=1.0, value=default_milho, step=0.5)
            preco_nucleo_input = st.number_input("Preço do Núcleo (R$/kg)", min_value=0.0, value=7.0, step=0.1)
        with col_preco2:
            preco_soja_input = st.number_input("Preço da Saca de Soja (50kg) (R$)", min_value=1.0, value=default_soja, step=0.5)
            preco_outros_input = st.number_input("Preço de Outros Insumos (R$/kg)", min_value=0.0, value=1.0, step=0.1)

        KG_SACA_MILHO, KG_SACA_SOJA = 60, 50
        custo_kg_racao = 0
        if soma_percentuais > 0:
            preco_kg_milho = preco_milho_input / KG_SACA_MILHO
            preco_kg_soja = preco_soja_input / KG_SACA_SOJA
            custo_kg_racao = ((preco_kg_milho * percent_milho) + (preco_kg_soja * percent_soja) + (preco_nucleo_input * percent_nucleo) + (preco_outros_input * percent_outros)) / soma_percentuais
        st.success(f"Custo do Concentrado (matéria natural): R$ {custo_kg_racao:,.2f} por kg", icon="🧮")
        
        # --- ANÁLISE DA SIMULAÇÃO (FINAL) ---
        st.markdown("---")
        st.markdown("<h3 style='text-align: center;'>Análise da Sua Simulação Específica</h3>", unsafe_allow_html=True)
        
        custo_aquisicao = initial_investment
        composicao_interativa = {'milho': percent_milho, 'soja': percent_soja, 'nucleo': percent_nucleo, 'outros': percent_outros}
        precos_simulacao = {'milho_saca': preco_milho_input, 'soja_saca': preco_soja_input, 'nucleo_kg': preco_nucleo_input, 'outros_kg': preco_outros_input}
        custo_alimentacao_unitario = calcular_custo_alimentacao(arrobas_gain_head, conversao_ajustada, pct_concentrado_input, composicao_interativa, precos_simulacao)
        custo_alimentacao_total = custo_alimentacao_unitario * num_heads_bought
        custos_mensais_adicionais_total = custos_mensais_por_cabeca * num_heads_bought * period_months
        outros_custos_total = fixed_costs + custos_mensais_adicionais_total
        capital_cost_value = custo_aquisicao * (cost_of_capital / 100) * (period_months / 12)
        total_cost_cattle = custo_aquisicao + custo_alimentacao_total + outros_custos_total + capital_cost_value
        peso_final_por_cabeca = arroba_media_inicial + arrobas_gain_head
        revenue = num_heads_bought * peso_final_por_cabeca * sell_arroba_price
        profit_cattle = revenue - total_cost_cattle

        selic_monthly_rate = (1 + selic_rate / 100)**(1/12) - 1
        selic_final_value = custo_aquisicao * (1 + selic_monthly_rate)**period_months
        selic_gross_profit = selic_final_value - custo_aquisicao
        profit_selic = selic_gross_profit * (1 - ir_rate / 100)
        
        valor_final_operacao = revenue - custo_alimentacao_total - outros_custos_total - capital_cost_value
        cash_flow_cattle = [-custo_aquisicao] + [0] * (period_months - 1) + [valor_final_operacao]
        irr_monthly_cattle = 0
        if custo_aquisicao > 0 and valor_final_operacao > 0:
            try: irr_monthly_cattle = npf.irr(cash_flow_cattle)
            except: irr_monthly_cattle = 0
                
        irr_monthly_selic = selic_monthly_rate * (1 - ir_rate / 100)
        
        # CÁLCULOS DOS NOVOS KPIs
        total_kg_concentrado = custo_alimentacao_total / custo_kg_racao if custo_kg_racao > 0 else 0
        custo_diario_alimentacao_lote = custo_alimentacao_total / (period_months * 30) if period_months > 0 else 0
        custo_diario_alimentacao_cabeca = custo_diario_alimentacao_lote / num_heads_bought if num_heads_bought > 0 else 0
        custo_operacional_total = custo_alimentacao_total + outros_custos_total
        total_arrobas_produzidas = arrobas_gain_head * num_heads_bought
        custo_arroba_produzida = custo_operacional_total / total_arrobas_produzidas if total_arrobas_produzidas > 0 else 0
        
        months = np.arange(period_months + 1)
        gado_values = np.linspace(custo_aquisicao, custo_aquisicao + profit_cattle, len(months))
        selic_values = [custo_aquisicao * (1 + irr_monthly_selic)**m for m in months]
        df_comp = pd.DataFrame({'Mês': months, 'Produção de Gado': gado_values, 'Aplicação Selic': selic_values})
        
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
        selic_trend = np.full(len(months), selic_rate)
        sns.lineplot(x=months, y=selic_trend, ax=ax2, color='red', linestyle='--', label='Taxa Selic Projetada (%)')
        ax2.set_ylabel('Taxa Selic (% ao ano)', color='white')
        ax2.tick_params(colors='white', which='both')
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        legend = ax2.legend(lines + lines2, labels + labels2, loc='upper left', frameon=True)
        legend.get_frame().set_facecolor('#333333')
        for text in legend.get_texts(): text.set_color("white")
        ax1.get_legend().remove()
        fig_comp = adicionar_marca_e_fonte(fig_comp, is_matplotlib=True)
        st.pyplot(fig_comp)
        
        st.markdown("---")
        st.subheader("Resumo da Operação")
        
        col1_resumo, col2_detalhes = st.columns(2)
        with col1_resumo:
            resumo_data = {
                'Métrica': [
                    'Lucro Total - Gado (R$)', 'Lucro Total - Selic (R$)', 
                    'TIR Mensal - Gado (%)', 'TIR Mensal - Selic (%)'
                ],
                'Valor': [
                    f"R$ {profit_cattle:,.2f}", f"R$ {profit_selic:,.2f}", 
                    f"{irr_monthly_cattle * 100:.2f}%", f"{irr_monthly_selic * 100:.2f}%"
                ]
            }
            st.table(resumo_data)
        with col2_detalhes:
            detalhes_gado = f"""
            | Detalhamento da Operação Gado | Valor |
            |---|---|
            | (+) Receita Total da Venda | R$ {revenue:,.2f} |
            | (-) Custo de Aquisição dos Animais | - R$ {custo_aquisicao:,.2f} |
            | (-) Custo com Concentrado ({total_kg_concentrado:,.0f} kg) | - R$ {custo_alimentacao_total:,.2f} |
            | *Custo Diário por Cabeça (Alim.)* | *R$ {custo_diario_alimentacao_cabeca:,.2f}* |
            | (-) Outros Custos (Fixos + Mensais/Vol.) | - R$ {outros_custos_total:,.2f} |
            | (-) Custo de Capital (Juros) | - R$ {capital_cost_value:,.2f} |
            | **(=) Custo da Arroba Produzida** | **R$ {custo_arroba_produzida:,.2f}** |
            | **(=) Lucro/Prejuízo Total** | **R$ {profit_cattle:,.2f}** |
            """
            st.markdown(detalhes_gado)

        st.markdown("---")
        st.subheader("Parâmetros Utilizados na Simulação")
        
        param_col1, param_col2, param_col3 = st.columns(3)
        with param_col1:
            st.markdown("##### **Compra, Venda e Prazos**")
            st.markdown(f"- **Valor Total Investido:** `R$ {initial_investment:,.2f}`")
            st.markdown(f"- **Nº de Cabeças:** `{num_heads_bought}`")
            st.markdown(f"- **Preço de Compra:** `R$ {buy_arroba_price:,.2f} /@`")
            st.markdown(f"- **Preço de Venda:** `R$ {sell_arroba_price:,.2f} /@`")
            st.markdown(f"- **Meta de Ganho:** `{arrobas_gain_head:,.1f} @ /cabeça`")
            st.markdown(f"- **Período:** `{period_months} meses`")
        with param_col2:
            st.markdown("##### **Custos e Dieta**")
            st.markdown(f"- **Custos Fixos Totais:** `R$ {fixed_costs:,.2f}`")
            st.markdown(f"- **Custo Mensal Adicional:** `R$ {custos_mensais_por_cabeca:,.2f} /cabeça`")
            st.markdown(f"- **% Concentrado na Dieta:** `{pct_concentrado_input:,.1f}%`")
            st.markdown(f"- **Preço Milho (saca):** `R$ {preco_milho_input:,.2f}`")
            st.markdown(f"- **Preço Soja (saca):** `R$ {preco_soja_input:,.2f}`")
            st.markdown(f"- **Preço Núcleo (kg):** `R$ {preco_nucleo_input:,.2f}`")
            st.markdown(f"- **Preço Outros (kg):** `R$ {preco_outros_input:,.2f}`")
        with param_col3:
            st.markdown("##### **Eficiência e Financeiro**")
            st.markdown(f"- **Conversão Alimentar Base (MS):** `{conversao_alimentar_base:.2f}`")
            st.markdown(f"- **Ajuste por Qualidade:** `{fator_qualidade_dieta:+d}%`")
            st.markdown(f"- **Conversão Final (MS):** `{conversao_ajustada:.2f}`")
            st.markdown(f"- **Custo de Capital:** `{cost_of_capital:,.2f}% a.a.`")
            st.markdown(f"- **Taxa Selic (Benchmark):** `{selic_rate:,.2f}% a.a.`")
            st.markdown(f"- **Alíquota de IR (Selic):** `{ir_rate:,.1f}%`")

    except Exception as e:
        st.error(f"Ocorreu um erro inesperado: {e}")

