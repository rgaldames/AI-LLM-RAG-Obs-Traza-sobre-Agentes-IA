# Informe Técnico EP3: Monitoreo, Observabilidad, Seguridad y Optimización en Agentes de IA

Este documento detalla el análisis de la arquitectura implementada para la **Evaluación Parcial N°3** sobre observabilidad, trazabilidad, seguridad y ética en el agente RAG veterinario (**VetBot**).

---

## 1. Arquitectura de Observabilidad y Trazabilidad

Se implementó un sistema de logs estructurados en formato JSON Lines (`agent_logs.jsonl`) que registra automáticamente cada interacción del agente con los siguientes metadatos clave:
- **`session_id`**: UUID único generado al inicio de la sesión para correlación de hilos de chat.
- **`timestamp`**: Fecha y hora en formato ISO 8601 de la transacción.
- **`user_input`**: Entrada del usuario sanitizada de información sensible (PII).
- **`agent_response`**: Respuesta final generada por el agente, también sanitizada.
- **`tools_used`**: Lista de herramientas utilizadas durante la planificación adaptativa de ese turno (`search_clinical_db`, `analyze_symptoms`, `write_visit_summary`).
- **`latency`**: Tiempo de procesamiento total en segundos.
- **`tokens`**: Consumo exacto o estimado de tokens de entrada, salida y totales.
- **`groundedness_score`**: Índice de calidad y fidelidad de la respuesta frente a las fuentes de contexto inyectadas (RAG).
- **`errors`**: Registro detallado de excepciones y fallas encontradas en APIs y herramientas.

---

## 2. Seguridad y Privacidad: Anonimización de Datos (PII)

Para cumplir con los estándares éticos y legales (como la Ley N° 19.628 de Protección de la Vida Privada en Chile), se desarrolló un wrapper de sanitización basado en expresiones regulares antes de persistir los logs o enviar los datos al LLM:

### Expresiones Regulares Utilizadas:
1. **Emails**:
   ```regex
   [a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+
   ```
   *Reemplazo*: `[REDACTED_EMAIL]`
2. **RUT Chileno**:
   ```regex
   \b\d{1,2}(?:\.?\d{3}){2}-?[0-9kK]\b
   ```
   *Reemplazo*: `[REDACTED_RUT]`

---

## 3. Análisis de Métricas y Cuellos de Botella Detectados

Tras el análisis de las métricas recopiladas en los logs:
- **Latencia de Herramientas**: Las llamadas a la herramienta RAG `search_clinical_db` (búsqueda de similitud en ChromaDB) representan el principal cuello de botella debido a la carga local del modelo de embeddings `all-MiniLM-L6-v2`.
- **Anomalías de Latencia**: Se consideran anomalías todas las consultas que sobrepasan el umbral de $\mu + 1.5 \times \sigma$. En la mayoría de los casos, están correlacionadas con loops de llamadas consecutivas a herramientas (tool-calling loops) cuando el agente evalúa síntomas complejos o realiza múltiples consultas secuenciales de base de datos.
- **Eficiencia de Tokens**: El historial de la memoria de corto plazo de ventana deslizante ($k=5$) incrementa el tamaño del prompt acumulativamente, requiriendo optimización en consultas largas.

---

## 4. Capturas de Pantalla del Dashboard de Monitoreo

*(Inserta aquí las capturas de pantalla de tu dashboard levantado localmente mediante Streamlit)*

- **Métricas KPI**: `[Captura_KPIs.png]`
- **Gráfico de Latencia y Anomalías**: `[Captura_Latencia_Anomalias.png]`
- **Consumo de Tokens e Histograma de Groundedness**: `[Captura_Tokens_Groundedness.png]`

---

## 5. Recomendaciones de Optimización y Escalabilidad

1. **Embedding Caching & Vector DB en la Nube**: Migrar de ChromaDB local con embeddings HuggingFace locales a un servicio administrado (como Google Cloud Vertex AI Vector Search) para reducir la latencia de RAG de ~3s a <100ms.
2. **Optimización del Contexto de Memoria**: Implementar una estrategia de resumen de memoria en lugar de un buffer de ventana deslizante crudo para reducir el consumo redundante de tokens en conversaciones extensas.
3. **Mecanismo de Reintento con Exponencial Backoff**: Para solventar fallas temporales de red y Timeouts en el llamado a herramientas, configurar políticas de resiliencia directamente en los wrappers de LangChain.
