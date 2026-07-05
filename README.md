# Proyecto Académico: Agente RAG Veterinario

> **Asignatura:** Ingeniería de Soluciones con IA  
> **Evaluación:** Desarrollo de un Agente Funcional  
> **Framework:** LangChain 1.2 + LangGraph  
> **Dominio:** Asistente virtual de orientación para clínica veterinaria

---

## Descripción

Sistema de agente funcional que integra herramientas de **consulta, escritura y razonamiento** en un flujo de trabajo veterinario simulado. El agente usa el patrón **Tool-Calling Loop** (equivalente moderno al ReAct) de LangChain 1.2, combinando recuperación de información (RAG), memoria de corto y largo plazo, y toma de decisiones adaptativa.

---

## Requisitos del Sistema

| Requisito | Versión mínima |
|---|---|
| Python | 3.10+ (probado en 3.13.9) |
| Sistema Operativo | Windows / macOS / Linux |
| Conexión a internet | Necesaria (para GitHub Models API) |
| Token de GitHub | Con acceso a GitHub Models Beta |

---

## Instalación Paso a Paso

### 1. Clona el repositorio

```bash
git clone https://github.com/rgaldames/5sem-proyecto-llm-rag.git
cd 5sem-proyecto-llm-rag
```

### 2. Crear y activar un entorno virtual

**Windows:**
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar las dependencias del proyecto

Las librerías del agente RAG junto con los nuevos módulos de observabilidad y el dashboard (`streamlit`, `pandas`, `plotly`) están empaquetadas en `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Configurar las Credenciales

Crea un archivo llamado `.env` en la raíz del proyecto para alojar de forma segura tus variables de entorno locales:

```ini
# Token de GitHub Models (Azure Inference Endpoint)
GITHUB_PAT_TOKEN=ghp_tu_github_personal_access_token
```

> [!IMPORTANT]
> **Cómo obtener el token de GitHub Models:**
> 1. Inicia sesión en [GitHub](https://github.com).
> 2. Haz clic en tu avatar en la esquina superior derecha y selecciona **Settings**.
> 3. En el menú de la izquierda, selecciona **Developer settings** -> **Personal access tokens** -> **Tokens (classic)**.
> 4. Haz clic en **Generate new token (classic)**, asígnale un nombre (ej: `VetBot-LLM`) y guárdalo **sin marcar ningún permiso especial** (alcance mínimo de lectura pública).
> 5. Copia el token y pégalo en tu archivo `.env`.

---

## Estructura del Proyecto

```
5sem-proyecto-llm-rag/
|
|-- agent.py                     <- Agente principal (Tool-Calling Loop + sanitización + logs)
|-- tools.py                     <- 3 herramientas del agente: consulta, escritura, razonamiento
|-- short_term_memory.py         <- Memoria de corto plazo (buffer deslizante de mensajes k=5)
|-- memory_store.py              <- Memoria de largo plazo (persistencia en JSON + embeddings)
|-- dashboard.py                 <- Panel interactivo de monitoreo en Streamlit
|
|-- requirements.txt             <- Listado completo de dependencias (incluye Streamlit y Plotly)
|-- agent_logs.jsonl             <- Archivo de logs estructurados (se genera dinámicamente)
|-- visit_summaries.jsonl        <- Registro de visitas guardadas por la herramienta (se genera solo)
|-- memories.json                <- Almacenamiento persistente de recuerdos (se genera solo)
|-- chroma_db_local/             <- Carpeta de base de datos vectorial ChromaDB (se genera sola)
|
|-- Informe_Tecnico_EP3.md       <- Plantilla del informe de observabilidad y seguridad para evaluación
|-- .env                         <- Credenciales locales (excluido en .gitignore por seguridad)
|-- .gitignore
+-- README.md
```

---

## Evidencia de Trazabilidad y Observabilidad en Acción

Para que cualquier estudiante, profesor o evaluador pueda constatar el funcionamiento de la observabilidad y trazabilidad, se describe a continuación la estructura del archivo `agent_logs.jsonl` generado con cada interacción del agente:

### Ejemplo de Registro JSONL Estructurado (Traza de una consulta con PII)

Si un usuario ingresa una consulta que contiene datos sensibles como un RUT chileno o correo electrónico:
> *"Hola, mi email es **contacto@cliente.com** y mi RUT es **18.765.432-1**, busquemos si Toby tiene vacunas vigentes."*

El sistema automáticamente la intercepta, la sanitiza en memoria y escribe la traza en `agent_logs.jsonl` con el siguiente formato:

```json
{
  "session_id": "84b46b2a-f125-463f-a697-fc12282af85e",
  "timestamp": "2026-07-04T20:58:12-04:00",
  "user_input": "Hola, mi email es [REDACTED_EMAIL] y mi RUT es [REDACTED_RUT], busquemos si Toby tiene vacunas vigentes.",
  "agent_response": "De acuerdo al registro clínico de Toby en nuestra base de datos, tiene su vacuna antirrábica vigente desde el 12 de marzo. Te recomiendo agendar su próximo chequeo preventivo.",
  "tools_used": [
    "search_clinical_db"
  ],
  "latency": 3.4215,
  "tokens": {
    "input_tokens": 185,
    "output_tokens": 74,
    "total_tokens": 259
  },
  "groundedness_score": 0.8125,
  "errors": []
}
```

### ¿Cómo interpretar y auditar estas trazas?
- **Trazabilidad de Hilo (Session ID)**: El `session_id` agrupa todas las interacciones realizadas en una misma sesión de chat, permitiendo al evaluador auditar el historial de la conversación.
- **Seguridad Comprobable**: El RUT y el correo se han anonimizado en caliente antes de guardarse en el log y antes de enviarse al LLM (cumpliendo con la ética de privacidad).
- **Métricas de Rendimiento**: `latency` calcula exactamente cuántos segundos tomó el ciclo completo (RAG + LLM), y `tokens` detalla el coste computacional de la interacción.
- **Métrica de Calidad**: `groundedness_score` (entre `0.0` y `1.0`) mide el porcentaje de palabras clave de la respuesta que coinciden con el contexto de ChromaDB, validando si el modelo está inventando respuestas o ciñéndose a la base de datos veterinaria.


---

## Ejecución y Monitoreo

### 1. Ejecutar el Agente Veterinario (Consola Interactiva)

```bash
python agent.py
```

- **Comandos especiales en el chat**:
  - `estado`: Muestra estadísticas en tiempo real del buffer de memoria.
  - `historial`: Imprime el historial del buffer conversacional de corto plazo.
  - `salir`: Apaga el agente de forma segura.

### 2. Ejecutar el Dashboard de Monitoreo y Observabilidad (Streamlit)

```bash
streamlit run dashboard.py
```

Esto desplegará una interfaz web premium en tu navegador local (por defecto en `http://localhost:8501`) donde podrás monitorear el comportamiento del agente mediante gráficos interactivos en tiempo real.

---

## Nuevas Funcionalidades e Integración (EP3)

El sistema ahora integra cuatro pilares fundamentales para la producción de sistemas de IA:

1. **Seguridad y Privacidad (Anonimización de PII)**:
   - Toda entrada del usuario es analizada por la función `sanitize_pii` antes de ser guardada en logs o enviada al modelo de lenguaje (LLM).
   - Utiliza expresiones regulares optimizadas para interceptar correos electrónicos (`[REDACTED_EMAIL]`) y RUTs chilenos (`[REDACTED_RUT]`).

2. **Trazabilidad y Logs Estructurados (`agent_logs.jsonl`)**:
   - Cada interacción genera un registro único JSON conteniendo: `session_id` (UUID), `timestamp`, `user_input` (sanitizado), `agent_response` (sanitizada), `tools_used` (lista de herramientas llamadas), `latency` (en segundos), `tokens` (entrada, salida y total), `groundedness_score` y `errors` (fallas capturadas).

3. **Métricas de Observabilidad**:
   - **Latencia**: Registrada por turno y agregada en el panel.
   - **Rendimiento y Costos**: Registro detallado de tokens usando callbacks nativos de LangChain.
   - **Calidad y Estabilidad**: Cálculo heurístico de Groundedness (solapamiento contextual RAG) y conteo automático de excepciones en APIs y herramientas.

4. **Dashboard Interactivo**:
   - Construido con Streamlit, Pandas y Plotly. Permite analizar el uso de herramientas, consumo de tokens, evolución de la latencia e identificar anomalías dinámicamente mediante el criterio estadístico de $\mu + 1.5 \times \sigma$.

---

## Arquitectura Detallada del Sistema

El siguiente diagrama ilustra cómo colaboran las funcionalidades originales con los nuevos módulos de seguridad y observabilidad en cada turno de conversación:

```
                  Usuario (Ingresa consulta con RUT / Email)
                                   |
                                   v
             +-------------------------------------------+
             |      Wrapper de Sanitización de PII       | <-- Quita datos sensibles
             +-------------------------------------------+
                                   |
                         (Consulta Sanitizada)
                                   |
                                   v
             +-------------------------------------------+
             |         Sistema de Memoria Dual           |
             |  - Corto Plazo (Buffer de mensajes)       |
             |  - Largo Plazo (Recuperación semántica)   |
             +-------------------------------------------+
                                   |
                       (Prompt Contextualizado)
                                   |
                                   v
             +-------------------------------------------+
             |     Planificación Adaptativa (LLM)        |
             |  - Evalúa y decide uso de herramientas    |
             |  - [search_clinical_db] -> RAG            |
             |  - [analyze_symptoms] -> Reglas           |
             |  - [write_visit_summary] -> Escritura     |
             +-------------------------------------------+
                                   |
                                   v
             +-------------------------------------------+
             |     Cálculo de Métricas y Logging         |
             |  - Latencia | Tokens | Groundedness       |
             |  - Registra en 'agent_logs.jsonl'         |
             +-------------------------------------------+
                                   |
                                   +--------> Dashboard de Streamlit (Monitoreo)
                                   |
                                   v
                   Respuesta Final Sanitizada al Usuario
```

---

## Cómo Validar la Observabilidad y la Trazabilidad

Para verificar el cumplimiento técnico de los módulos implementados, realiza las siguientes pruebas:

### 1. Validación de Seguridad (Sanitización PII)
- Ejecuta `python agent.py`.
- Introduce una consulta conteniendo datos personales simulados, por ejemplo:
  `"Hola, mi correo es test@correo.cl y mi RUT es 19.345.678-9, ¿cómo está Toby?"`
- Revisa el archivo `agent_logs.jsonl`. Deberás comprobar que los datos personales aparecen reemplazados por sus etiquetas redactadas tanto en `user_input` como en la respuesta, garantizando que ninguna PII se guarde en disco o se envíe al LLM.

### 2. Validación de Métricas en el Dashboard
- Ejecuta consultas rápidas y otras complejas que utilicen RAG.
- Abre el dashboard con `streamlit run dashboard.py`.
- Valida que:
  - Las consultas totales aumenten y se calculen la latencia promedio y los tokens.
  - Se grafique la frecuencia correcta de las herramientas llamadas (ej: `search_clinical_db` o `analyze_symptoms`).
  - Las llamadas lentas (por ejemplo, el primer arranque del embedding o fallos de red) aparezcan destacadas en la tabla de **Consultas Anómalas** al exceder el umbral dinámico de anomalía.

### 3. Validación de Errores y Calidad (Groundedness)
- Desconecta la red local temporalmente o deshabilita la clave del `.env` y realiza una consulta. El dashboard de Streamlit mostrará inmediatamente la tasa de error incrementada y los detalles del fallo de API capturados en la sección **🎯 Calidad de Respuesta y Errores**.
- Compara cómo varía la puntuación de `groundedness` entre respuestas directas del LLM (cercanas a 1.0 por no requerir validación de contexto) frente a respuestas basadas en la base de datos de historiales clínicos.


---

## Descripción de Componentes

### `agent.py`  Orquestador Principal

Punto de entrada del sistema. Inicializa todos los módulos y ejecuta el bucle de interacción.

- Usa `create_agent()` de LangChain 1.2 (basado en LangGraph internamente)
- El LLM llama herramientas en loop automático hasta tener suficiente información
- Integra memoria dual en cada invocación mediante `run_agent()`
- Inyecta el contexto de largo plazo directamente en el mensaje de entrada

### `tools.py`  Herramientas del Agente

Define las tres herramientas que el agente puede invocar de forma autónoma:

| Herramienta | Tipo | Descripción |
|---|---|---|
| `search_clinical_db` | **CONSULTA** | Búsqueda semántica en ChromaDB sobre 50 historiales clínicos |
| `write_visit_summary` | **ESCRITURA** | Persiste consultas en `visit_summaries.jsonl` para auditoría |
| `analyze_symptoms` | **RAZONAMIENTO** | Clasifica síntomas como CRÍTICOS / MODERADOS / LEVES por reglas |

### `short_term_memory.py`  Memoria de Corto Plazo (Activa y Funcional)

Buffer de ventana deslizante implementado con `HumanMessage` / `AIMessage` de `langchain_core`.

- **Historial Completo:** Mantiene activas las últimas `k=5` interacciones (10 mensajes) y se inyecta directamente al `agent.invoke()` para asegurar la coherencia conversacional del hilo actual.
- **Volatilidad:** Almacenado en RAM; se limpia al cerrar la sesión o reiniciar el proceso.
- **API Nativa:** Compatible 100% con los tipos de mensajes de LangChain 1.2+.

### `memory_store.py`  Memoria de Largo Plazo (Optimizado a 800 Caracteres)

Almacenamiento persistente entre sesiones usando embeddings semánticos locales y similitud coseno.

- **Persistencia Física:** Guarda las interacciones en el archivo local `memories.json`.
- **Amplitud Extendida (800 Caracteres):** Guarda hasta 800 caracteres de preguntas/respuestas combinadas (`[:800]`) para retener nombres de medicamentos, dosis, mascotas y alérgenos específicos, alineándose perfectamente con la ventana semántica de 256 tokens del modelo `all-MiniLM-L6-v2`.
- **Recuperación Inteligente:** Busca y extrae los 3 recuerdos semánticamente más similares para inyectarlos en el prompt antes de cada consulta.

---

## Planificación Adaptativa

El agente **no sigue un flujo rígido**. El LLM decide en tiempo real qué herramienta usar según el tipo de consulta:

```
Input del usuario
      |
      v
  LLM analiza el contenido
      |
      +-------> Sintomas mencionados?        -> analyze_symptoms    -> nivel de urgencia
      |
      +-------> Pregunta clinica especifica? -> search_clinical_db  -> contexto clinico
      |
      +-------> Consulta concluida?          -> write_visit_summary -> registro persistente
      |
      +-------> Puede responder directamente? -> respuesta sin llamar herramientas
      |
      v
  Respuesta final al usuario
```

**Ejemplos de comportamiento adaptativo:**

| Consulta del usuario | Herramienta elegida | Razón |
|---|---|---|
| "Mi perro tuvo una convulsión" | `analyze_symptoms` | Detecta síntoma crítico  alerta urgencia |
| "¿Qué tratamientos usan para otitis?" | `search_clinical_db` | Busca en historiales clínicos |
| "Gracias, mi gato se llama TOM" | `write_visit_summary` | Registra el cierre de la consulta |
| "¿Cuántas veces al día debo alimentar a un gatito?" | *(sin herramienta)* | El LLM responde directamente |

---

## Preguntas de Prueba Sugeridas

Puedes probar con las siguientes preguntas o crear las tuyas al ejecutar `agent.py`:

```
1. Mi perro lleva dos dias con tos y sin apetito. ¿Que hago?

2. ¿Que tipo de enfermedades han tenido los gatos en la clinica?

3. Mi gato tuvo una convulsion hace 10 minutos.

4. ¿Cada cuanto debo llevar a mi perro al veterinario para chequeo preventivo?

5. Mi mascota se llama Max, es un labrador de 4 anios y consulte por perdida de pelo.
   Guarda un resumen de esta consulta.
```

---

## Solución de Problemas Frecuentes

### Error: `cannot import name 'AgentExecutor' from 'langchain.agents'`

Este error ocurre si tienes LangChain 0.x instalado. Este proyecto usa **LangChain 1.2+** que reemplazó `AgentExecutor` / `create_react_agent` por `create_agent`. Solución:

```bash
pip install --upgrade langchain==1.2.15 langchain-core==1.3.0
```

### Error: `No module named 'langchain.memory'`

El módulo `langchain.memory` fue eliminado en LangChain 1.x. El proyecto ya implementa su propia memoria en `short_term_memory.py` sin depender de ese módulo. Verifica que estés ejecutando `agent.py` (no `rag_chatbot.py`).

### Error: `No se encontro GITHUB_PAT_TOKEN`

Verifica que el archivo `.env` existe en la misma carpeta que `agent.py` con el formato exacto:
```
GITHUB_PAT_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
```
Sin comillas, sin espacios alrededor del `=`.

### Error: `looping content` del modelo

El proyecto ya tiene este comportamiento corregido en el system prompt de `agent.py`. El LLM decide autónomamente si llamar o no una herramienta, evitando el loop.

### El modelo de embeddings tarda mucho en cargar

Normal en la primera ejecución  descarga `all-MiniLM-L6-v2` (~90 MB) desde HuggingFace. En ejecuciones siguientes se carga desde caché local en segundos.

### Warning: `HuggingFaceEmbeddings was deprecated`

Es solo un aviso, no un error. El proyecto funciona correctamente.

---

## Versiones Exactas Utilizadas

| Paquete | Versión |
|---|---|
| Python | 3.13.9 |
| langchain | 1.2.15 |
| langchain-core | 1.3.0 |
| langchain-community | 0.4.1 |
| langchain-openai | 1.1.14 |
| langchain-chroma | 1.1.0 |
| sentence-transformers | 5.4.1 |
| chromadb | 1.5.8 |
| openai | 2.32.0 |
| numpy | 2.4.4 |
| python-dotenv | 1.2.2 |
| huggingface-hub | 1.11.0 |

## Glosario de Términos

A continuación se describen los conceptos técnicos clave utilizados en este proyecto para facilitar la comprensión y evaluación del sistema:

*   **PII (Personally Identifiable Information - Información de Identificación Personal)**: Cualquier dato que permita identificar de forma única a un individuo, como correos electrónicos, RUTs chilenos, nombres o números telefónicos. En este proyecto se anonimizan automáticamente para resguardar la privacidad y cumplir con la ética de protección de datos.
*   **RAG (Retrieval-Augmented Generation - Generación Recuperada por Búsqueda)**: Técnica que complementa al modelo de lenguaje (LLM) inyectándole información relevante recuperada en tiempo real desde una base de datos externa (ChromaDB) para fundamentar sus respuestas.
*   **ChromaDB**: Base de datos vectorial utilizada de forma local para indexar y buscar de manera eficiente similitudes semánticas en historiales clínicos veterinarios.
*   **Embeddings**: Representaciones vectoriales numéricas que capturan el significado semántico de las palabras y textos. Este proyecto usa el modelo local `all-MiniLM-L6-v2`.
*   **Observabilidad**: La capacidad de inferir los estados internos y comportamientos de un sistema a partir de las métricas externas recopiladas en tiempo real (como latencia, conteo de errores y groundedness).
*   **Trazabilidad**: El registro cronológico detallado de las decisiones, llamadas a APIs y ejecuciones de herramientas que sigue el agente para dar respuesta a un requerimiento de usuario.
*   **JSONL (JSON Lines)**: Formato de persistencia de archivos de texto estructurado donde cada línea representa un objeto JSON independiente. Ideal para la escritura rápida de logs y lectura con Pandas.
*   **LLM (Large Language Model)**: El modelo fundacional de inteligencia artificial (`gpt-4o-mini` a través de GitHub Models API) encargado del procesamiento del lenguaje, la toma de decisiones y la generación de respuestas.
*   **ReAct (Reasoning and Acting)**: Paradigma de agentes en el cual el LLM combina pasos de razonamiento lógico con el llamado a herramientas externas en bucle (Tool-Calling Loop) para resolver la tarea del usuario.
*   **Groundedness**: Métrica de calidad que evalúa el nivel de consistencia y veracidad de la respuesta del agente respecto al contexto recuperado, previniendo alucinaciones en el diagnóstico y la orientación clínica.

---

## Indicadores de Logro

| IL | Indicador | Implementación |
|---|---|---|
| **IL2.1** | Herramientas de consulta, escritura y razonamiento | `search_clinical_db`, `write_visit_summary`, `analyze_symptoms` en `tools.py` |
| **IL2.2** | Memoria de corto y largo plazo | `ShortTermMemory` (buffer HumanMessage/AIMessage k=5) + `MemoryStore` (embeddings JSON persistentes) |
| **IL2.3** | Planificación y toma de decisiones adaptativas | LangGraph tool-calling loop: el LLM decide autónomamente qué herramienta invocar y cuándo |
| **IL2.4** | Documentación técnica y orquestación | Diagramas ASCII en código fuente + este README con arquitectura, flujos y ejemplos |

---

## Referencias Bibliográficas (APA)

Chase, H. (2022). *LangChain* [Software]. GitHub. https://github.com/langchain-ai/langchain

Chroma (Trychroma, Inc.). (2023). *Chroma: The open-source embedding database*. Chroma. https://docs.trychroma.com/

Hugging Face. (2023). *Sentence Transformers: all-MiniLM-L6-v2*. Hugging Face. https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

LangChain AI. (2024). *LangChain 1.2 documentation: Agents*. LangChain. https://python.langchain.com/docs/modules/agents/

LangChain AI. (2024). *LangGraph: Build stateful, multi-actor applications with LLMs*. LangChain AI. https://langchain-ai.github.io/langgraph/

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems, 33*, 9459–9474. https://doi.org/10.48550/arXiv.2005.11401

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing*. https://doi.org/10.48550/arXiv.1908.10084

Streamlit Inc. (2024). *Streamlit: The fastest way to build and share data apps* [Software]. https://docs.streamlit.io/

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2023). ReAct: Synergizing reasoning and acting in language models. *International Conference on Learning Representations (ICLR 2023)*. https://doi.org/10.48550/arXiv.2210.03629

Sipior, J. C., Ward, B. T., & Rainone, S. M. (2023). Ethical considerations of AI-driven systems and data protection regulation. *Information Systems Management, 40*(3), 234–249. https://doi.org/10.1080/10580530.2023.2189876

OpenAI. (2024). *Best practices for safety and privacy in LLM deployments*. OpenAI. https://platform.openai.com/docs/guides/safety-best-practices

---

*Desplegado con proposito academico  Ingenieria de Soluciones con IA, 2026.*

*por Ricardo A. Galdames Soto*
