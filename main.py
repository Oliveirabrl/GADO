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
from PIL import Image # <-- NOVA IMPORTAÇÃO

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
            st.markdown(caption, unsafe_allow_html=True)

# --- NOVA FUNÇÃO PARA ADICIONAR MARCA D'ÁGUA E FONTE ---
def adicionar_marca_e_fonte(fig, is_matplotlib=False):
    """Adiciona uma marca d'água e um texto de fonte a um gráfico Plotly ou Matplotlib."""
    source_text = "Fonte: CEPEA, B³ / Elaborado por: OS CAPITAL."
    image_path = os.path.join(BASE_DIR, 'assets', 'oscapital.jpeg')

    # Retorna a figura original se a imagem não for encontrada
    if not os.path.exists(image_path):
        st.warning(f"Imagem da marca d'água não encontrada em: {image_path}")
        return fig

    if is_matplotlib:
        try:
            img = Image.open(image_path)
            width, height = fig.get_size_inches() * fig.dpi
            
            # Adiciona a marca d'água centralizada
            fig_width_px, fig_height_px = int(width), int(height)
            img_width_px, img_height_px = img.size
            x_pos = (fig_width_px - img_width_px) // 2
            y_pos = (fig_height_px - img_height_px) // 2
            fig.figimage(img, xo=x_pos, yo=y_pos, alpha=0.1, zorder=-1)

            # Adiciona o texto da fonte
            fig.text(0.5, 0.01, source_text, ha='center', va='bottom', fontsize=9, color='grey')
            fig.subplots_adjust(bottom=0.15) # Ajusta a margem para o texto não sobrepor
        except Exception as e:
            st.error(f"Erro ao adicionar marca d'água no Matplotlib: {e}")
        return fig
    else: # Plotly
        try:
            # Adiciona a marca d'água
            fig.add_layout_image(
                dict(
                    source=Image.open(image_path),
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    sizex=0.5, sizey=0.5,
                    xanchor="center", yanchor="middle",
                    opacity=0.15,
                    layer="below"
                )
            )
            # Adiciona o texto da fonte
            fig.add_annotation(
                text=source_text,
                xref="paper", yref="paper",
                x=0.5, y=-0.17,  # Posição abaixo do eixo X
                showarrow=False,
                align="center",
                font=dict(size=11, color="grey")
            )
            # Garante espaço para a anotação
            fig.update_layout(margin=dict(b=90))
        except Exception as e:
            st.error(f"Erro ao adicionar marca d'água no Plotly: {e}")
        return fig

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
            # --- PREPARAÇÃO DE DADOS E CÁLCULOS BASE ---
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
            df_merged['mes'] = df_merged['Data'].dt.month

            df_merged['KG_Anterior'] = df_merged.groupby(df_merged['Data'].dt.month)['kg_liquido'].shift(1)
            custo_producao_base = calcular_custo_alimentacao(df_merged, 11.0, 80, 6.5, 85.0) + df_merged['preco_bezerro_brl']
            df_merged['custo_producao_base'] = custo_producao_base
            df_merged['custo_producao_base_media_movel'] = df_merged['custo_producao_base'].rolling(window=12).mean()
            df_merged['receita_por_cabeca_base'] = df_merged['preco_brl_arroba'] * 28
            df_merged['margem_bruta_base'] = df_merged['receita_por_cabeca_base'] - df_merged['custo_producao_base']

            # --- LÓGICA E GRÁFICO DO TERMÔMETRO HISTÓRICO ---
            st.markdown("---")
            st.markdown("<h3 style='text-align: center;'>Termômetro de Mercado e Histórico de Sinais</h3>", unsafe_allow_html=True)
            
            sensibilidade_sinal = st.slider(
                "Sensibilidade do Sinal (Nº de Condições Mínimas)",
                min_value=3, max_value=5, value=4,
                help="Define o número mínimo de indicadores (de 1 a 5) que precisam apontar na mesma direção para gerar um sinal."
            )

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
            
            fig_sinais_hist = adicionar_marca_e_fonte(fig_sinais_hist) # <-- APLICAÇÃO DA FUNÇÃO
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

            fig_export = adicionar_marca_e_fonte(fig_export) # <-- APLICAÇÃO DA FUNÇÃO
            st.plotly_chart(fig_export, use_container_width=True)

            st.markdown("---")
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
            
            fig_sazonal = adicionar_marca_e_fonte(fig_sazonal) # <-- APLICAÇÃO DA FUNÇÃO
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

            with st.expander("Ver explicação do Gráfico de Custo vs. Receita"):
                st.markdown("""
                Este é o **simulador principal**. Ele compara o `Custo de Produção por Cabeça` (barras) com a `Receita Estimada por Cabeça` (linha amarela), ambos na mesma escala (R$).
                - **Interpretação:** É a visualização direta da lucratividade da operação, baseada nos parâmetros que você define acima.
                - **Cores:** As barras **azuis** indicam meses onde a receita superou o custo (lucro), enquanto as barras **vermelhas** indicam o contrário (prejuízo).
                """)
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

            fig_custo = adicionar_marca_e_fonte(fig_custo) # <-- APLICAÇÃO DA FUNÇÃO
            st.plotly_chart(fig_custo, use_container_width=True)

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

            fig_bezerro = adicionar_marca_e_fonte(fig_bezerro) # <-- APLICAÇÃO DA FUNÇÃO
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

            df_merged['margem_media_movel_reversao'] = df_merged['margem_bruta'].rolling(window=periodo_media_reversao).mean()
            df_merged['margem_desvio_padrao_reversao'] = df_merged['margem_bruta'].rolling(window=periodo_media_reversao).std()
            df_merged['banda_superior_reversao'] = df_merged['margem_media_movel_reversao'] + (df_merged['margem_desvio_padrao_reversao'] * num_desvios_reversao)
            df_merged['banda_inferior_reversao'] = df_merged['margem_media_movel_reversao'] - (df_merged['margem_desvio_padrao_reversao'] * num_desvios_reversao)

            colors_margin_reversao = []
            for i in range(len(df_merged)):
                margem = df_merged['margem_bruta'].iloc[i]
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
            fig_reversao.add_trace(go.Bar(x=df_merged['Data'], y=df_merged['margem_bruta'], name='Margem Bruta (R$)', marker_color=colors_margin_reversao))
            fig_reversao.update_layout(title_text='Sinais de Compra e Venda por Reversão à Média da Margem', plot_bgcolor='rgba(17,17,17,0.9)', paper_bgcolor='rgba(17,17,17,0.9)', font_color="white", title_x=0.5, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            fig_reversao.update_yaxes(title_text="<b>Margem por Cabeça</b> (R$)")

            fig_reversao = adicionar_marca_e_fonte(fig_reversao) # <-- APLICAÇÃO DA FUNÇÃO
            st.plotly_chart(fig_reversao, use_container_width=True)

            # --- ANÁLISE DA SIMULAÇÃO (FINAL) ---
            st.markdown("---")
            st.markdown("<h3 style='text-align: center;'>Análise da Sua Simulação Específica</h3>", unsafe_allow_html=True)
            
            with st.expander("Ver explicação do Gráfico de Comparação de Investimentos e Resumo"):
                st.markdown("""
                Esta seção final apresenta um resumo da sua simulação e compara o investimento em produção de gado com uma aplicação financeira de renda fixa atrelada à taxa Selic.
                - **Gráfico de Linhas:** Mostra a evolução do valor acumulado ao longo do período de análise para ambas as opções de investimento (Gado vs. Selic). A linha tracejada vermelha representa uma projeção simplificada da taxa Selic.
                - **Tabela de Resumo:** Detalha os lucros totais e as Taxas Internas de Retorno (TIR) mensal para cada tipo de investimento, além de um detalhamento dos custos e ganhos para a produção de gado.
                """)

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
            
            fig_comp = adicionar_marca_e_fonte(fig_comp, is_matplotlib=True) # <-- APLICAÇÃO DA FUNÇÃO
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