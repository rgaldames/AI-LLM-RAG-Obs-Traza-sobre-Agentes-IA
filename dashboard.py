import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
import numpy as np

# Configuración de página
st.set_page_config(
    page_title="Dashboard de Monitoreo - VetBot IA",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo personalizado CSS para estética premium
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    h1, h2, h3 {
        color: #00f2fe;
        font-family: 'Outfit', sans-serif;
    }
    .stMetric {
        background-color: #1f2937;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #374151;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    div[data-testid="stMetricValue"] {
        color: #00f2fe;
    }
    </style>
    """, unsafe_allow_html=True)

# Título y Descripción
st.title("🐾 VetBot IA - Dashboard de Monitoreo y Observabilidad")
st.markdown("---")

LOG_FILE = "./agent_logs.jsonl"

def load_logs(file_path):
    if not os.path.exists(file_path):
        return pd.DataFrame()
    
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue
    
    if not records:
        return pd.DataFrame()
    
    df = pd.DataFrame(records)
    # Extraer tokens del diccionario interno
    if 'tokens' in df.columns:
        df['input_tokens'] = df['tokens'].apply(lambda x: x.get('input_tokens', 0) if isinstance(x, dict) else 0)
        df['output_tokens'] = df['tokens'].apply(lambda x: x.get('output_tokens', 0) if isinstance(x, dict) else 0)
        df['total_tokens'] = df['tokens'].apply(lambda x: x.get('total_tokens', 0) if isinstance(x, dict) else 0)
    else:
        df['input_tokens'] = 0
        df['output_tokens'] = 0
        df['total_tokens'] = 0
        
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# Carga de datos
df = load_logs(LOG_FILE)

if df.empty:
    st.warning("⚠️ No se encontraron registros de logs en 'agent_logs.jsonl'. Realiza algunas interacciones con el agente para poblar la base de datos de monitoreo.")
    
    # Simulación/Ejemplo de datos para visualización si no existe el archivo
    st.info("💡 Mostrando datos de ejemplo con fines ilustrativos.")
    example_data = [
        {
            "session_id": "example-session-1",
            "timestamp": pd.Timestamp.now() - pd.Timedelta(minutes=10),
            "user_input": "Mi gato tiene fiebre",
            "agent_response": "Recomiendo ir al veterinario si tiene fiebre alta...",
            "tools_used": ["analyze_symptoms"],
            "latency": 1.25,
            "input_tokens": 120,
            "output_tokens": 80,
            "total_tokens": 200,
            "groundedness_score": 1.0,
            "errors": []
        },
        {
            "session_id": "example-session-1",
            "timestamp": pd.Timestamp.now() - pd.Timedelta(minutes=8),
            "user_input": "Buscar antecedentes de Toby",
            "agent_response": "Encontré un registro de otitis para Toby...",
            "tools_used": ["search_clinical_db"],
            "latency": 3.82,
            "input_tokens": 240,
            "output_tokens": 150,
            "total_tokens": 390,
            "groundedness_score": 0.85,
            "errors": []
        },
        {
            "session_id": "example-session-2",
            "timestamp": pd.Timestamp.now() - pd.Timedelta(minutes=5),
            "user_input": "Guardar visita de Max",
            "agent_response": "Resumen de visita guardado con éxito.",
            "tools_used": ["write_visit_summary"],
            "latency": 2.10,
            "input_tokens": 150,
            "output_tokens": 90,
            "total_tokens": 240,
            "groundedness_score": 1.0,
            "errors": []
        },
        {
            "session_id": "example-session-2",
            "timestamp": pd.Timestamp.now() - pd.Timedelta(minutes=2),
            "user_input": "Mi perro sangra mucho",
            "agent_response": "[CRITICO] Llevar al veterinario inmediatamente...",
            "tools_used": ["analyze_symptoms"],
            "latency": 5.45,  # Anomalía simulada
            "input_tokens": 310,
            "output_tokens": 180,
            "total_tokens": 490,
            "groundedness_score": 0.95,
            "errors": []
        }
    ]
    df = pd.DataFrame(example_data)

# Métricas Principales (KPIs)
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_queries = len(df)
    st.metric(label="📊 Consultas Totales", value=total_queries)

with col2:
    avg_latency = df['latency'].mean()
    st.metric(label="⏱️ Latencia Promedio (seg)", value=f"{avg_latency:.2f} s")

with col3:
    total_tokens = df['total_tokens'].sum()
    st.metric(label="🪙 Tokens Consumidos", value=f"{total_tokens:,}")

with col4:
    # Contar errores
    error_count = df['errors'].apply(lambda x: len(x) if isinstance(x, list) else 0).sum()
    error_rate = (error_count / total_queries) * 100 if total_queries > 0 else 0
    st.metric(label="🚨 Errores Detectados", value=error_count, delta=f"{error_rate:.1f}% Tasa de Error", delta_color="inverse")

st.markdown("---")

# Fila 1: Latencia y Anomalías
st.subheader("⏱️ Análisis de Latencia y Detección de Anomalías")
col_lat1, col_lat2 = st.columns([2, 1])

# Cálculo de anomalías (Media + 1.5 * Desviación Estándar)
mean_latency = df['latency'].mean()
std_latency = df['latency'].std()
# Evitar NaN si hay solo 1 registro
std_latency = std_latency if not pd.isna(std_latency) else 0
anomaly_threshold = mean_latency + 1.5 * std_latency
df['is_anomaly'] = df['latency'] > anomaly_threshold
anomalies_df = df[df['is_anomaly']]

with col_lat1:
    fig_latency = px.line(
        df, 
        x='timestamp', 
        y='latency', 
        title="Evolución de Latencia por Consulta",
        labels={'latency': 'Latencia (s)', 'timestamp': 'Fecha y Hora'},
        markers=True,
        color_discrete_sequence=['#00f2fe']
    )
    # Línea de umbral de anomalías
    fig_latency.add_hline(y=anomaly_threshold, line_dash="dash", line_color="#ff073a", annotation_text="Umbral Anomalías")
    st.plotly_chart(fig_latency, use_container_width=True)

with col_lat2:
    st.write(f"**Umbral de Anomalía:** `{anomaly_threshold:.2f} segundos`")
    st.write(f"**Consultas Anómalas:** `{len(anomalies_df)}` de `{len(df)}`")
    
    if not anomalies_df.empty:
        st.dataframe(
            anomalies_df[['timestamp', 'user_input', 'latency']],
            column_config={
                "timestamp": "Fecha/Hora",
                "user_input": "Consulta",
                "latency": "Latencia (s)"
            },
            hide_index=True
        )
    else:
        st.success("✅ No se detectaron anomalías de latencia en los registros.")

st.markdown("---")

# Fila 2: Herramientas y Tokens
st.subheader("🛠️ Uso de Herramientas y Consumo de Tokens")
col_tool, col_tok = st.columns(2)

with col_tool:
    # Contar frecuencia de uso de cada herramienta
    tools_list = []
    for tools in df['tools_used']:
        if isinstance(tools, list):
            tools_list.extend(tools)
        else:
            tools_list.append(tools)
            
    if tools_list:
        tools_series = pd.Series(tools_list).value_counts().reset_index()
        tools_series.columns = ['Herramienta', 'Frecuencia']
        fig_tools = px.bar(
            tools_series, 
            x='Frecuencia', 
            y='Herramienta', 
            orientation='h',
            title="Frecuencia de Uso de Herramientas",
            color='Frecuencia',
            color_continuous_scale=px.colors.sequential.Teal
        )
        st.plotly_chart(fig_tools, use_container_width=True)
    else:
        st.info("ℹ️ No se registraron llamadas a herramientas.")

with col_tok:
    fig_tokens = px.bar(
        df,
        x='timestamp',
        y=['input_tokens', 'output_tokens'],
        title="Distribución de Consumo de Tokens (Entrada / Salida)",
        labels={'value': 'Cantidad de Tokens', 'timestamp': 'Fecha y Hora', 'variable': 'Tipo Token'},
        color_discrete_map={'input_tokens': '#00f2fe', 'output_tokens': '#4facfe'}
    )
    st.plotly_chart(fig_tokens, use_container_width=True)

st.markdown("---")

# Fila 3: Groundedness y Errores
st.subheader("🎯 Calidad de Respuesta y Errores")
col_qual1, col_qual2 = st.columns(2)

with col_qual1:
    avg_groundedness = df['groundedness_score'].mean() if 'groundedness_score' in df.columns else 1.0
    st.write(f"**Groundedness Promedio (Fidelidad RAG):** `{avg_groundedness * 100:.1f}%`")
    
    if 'groundedness_score' in df.columns:
        fig_groundedness = px.histogram(
            df,
            x='groundedness_score',
            nbins=10,
            title="Distribución de Groundedness Score",
            labels={'groundedness_score': 'Groundedness'},
            color_discrete_sequence=['#4facfe']
        )
        st.plotly_chart(fig_groundedness, use_container_width=True)

with col_qual2:
    st.write("**Detalle de Errores Registrados:**")
    errors_list = []
    for idx, row in df.iterrows():
        if isinstance(row.get('errors'), list) and row['errors']:
            for err in row['errors']:
                errors_list.append({
                    "timestamp": row['timestamp'],
                    "consulta": row['user_input'],
                    "error": err
                })
                
    if errors_list:
        errors_df = pd.DataFrame(errors_list)
        st.dataframe(errors_df, use_container_width=True)
    else:
        st.success("✅ Todos los llamados a APIs y herramientas se ejecutaron sin errores registrados.")

# Tabla de logs crudos al final
st.subheader("📋 Registro de Logs Completo (Sanitizado)")
st.dataframe(df[['session_id', 'timestamp', 'user_input', 'agent_response', 'tools_used', 'latency', 'total_tokens', 'groundedness_score']])
