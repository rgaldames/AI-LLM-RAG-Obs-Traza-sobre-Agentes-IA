# Proyecto Académico: Agente RAG Veterinario (Monitoreo, Observabilidad y Seguridad)

> **Asignatura:** Ingeniería de Soluciones con IA  
> **Evaluación:** Evaluación Parcial N°3 (EP3)  
> **Framework:** LangChain 1.2 + LangGraph  
> **Dominio:** Asistente virtual de orientación para clínica veterinaria (VetBot)

---

## 1. Descripción

Sistema de agente funcional que integra herramientas de **consulta, escritura y razonamiento** en un flujo de trabajo veterinario. El agente usa el patrón **Tool-Calling Loop** (equivalente moderno al ReAct) de LangChain 1.2, combinando recuperación de información (RAG), memoria de corto y largo plazo, y toma de decisiones adaptativas. 

En esta actualización (EP3), se han incorporado características críticas para entornos de producción: **Trazabilidad** mediante logs estructurados en JSONL, **Observabilidad** a través de un Dashboard interactivo de Streamlit, y **Seguridad y Privacidad** mediante anonimización en caliente de información de identificación personal (PII).

---

## 2. Requisitos del Sistema

| Requisito | Versión mínima |
|---|---|
| Python | 3.10+ (probado en 3.13.9 y 3.14) |
| Sistema Operativo | Windows / macOS / Linux |
| Conexión a internet | Necesaria (para descargar el embedding local e interactuar con la API) |
| Token de GitHub | Con acceso a la API de GitHub Models (Models Permission) |

---

## 3. Guía de Inicio Rápido (Instalación y Ejecución)

Sigue estos pasos en orden para descargar, configurar y correr el proyecto en tu máquina local:

### Paso 1: Clonar el repositorio
Abre una terminal y clona el proyecto:
```bash
git clone https://github.com/rgaldames/AI-LLM-RAG-Obs-Traza-sobre-Agentes-IA.git
cd AI-LLM-RAG-Obs-Traza-sobre-Agentes-IA
```

### Paso 2: Crear y activar el Entorno Virtual

**En Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**En Windows (DOS / CMD):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**En macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Paso 3: Instalar las Dependencias

> [!TIP]
> **Archivo Único de Dependencias:** Se ha consolidado un archivo unificado y limpio llamado **`requirements.txt`** en la raíz del proyecto. Este archivo contiene todas las dependencias necesarias de LangChain, ChromaDB, Streamlit, Pandas y Plotly de nivel superior para evitar conflictos de sistema o de compilación.

Instala todos los módulos necesarios de una sola vez:
```bash
pip install -r requirements.txt
```

### Paso 4: Configurar las Credenciales (`.env`)

Crea un archivo llamado `.env` en la raíz del proyecto para alojar de forma segura tus variables de entorno locales:

```ini
# Token de GitHub Models (Azure Inference Endpoint)
GITHUB_PAT_TOKEN=ghp_tu_github_personal_access_token
```

> [!IMPORTANT]
> **Cómo obtener el token de GitHub Models:**
> 1. Inicia sesión en [GitHub](https://github.com).
> 2. Haz clic en tu avatar en la esquina superior derecha y selecciona **Settings**.
> 3. En el menú de la izquierda, selecciona **Developer settings** -> **Personal access tokens** -> **Tokens (classic)** o **Fine-grained tokens**.
> 4. Si usas **Fine-grained tokens**, asegúrate de ir a la pestaña **Account Permissions** y conceder acceso **Read-only** al permiso **Models** (Beta).
> 5. Copia el token generado y pégalo en tu archivo `.env`.

### Paso 5: Ejecutar el Agente Veterinario (Consola)
Ejecuta la consola interactiva del chatbot:
```bash
python agent.py
```
* **Comandos especiales en el chat**:
  * `estado`: Muestra estadísticas en tiempo real del buffer de memoria.
  * `historial`: Imprime el historial del buffer conversacional de corto plazo.
  * `salir`: Apaga el agente de forma segura.

### Paso 6: Ejecutar el Dashboard de Monitoreo (Streamlit)
Abre una **segunda ventana de terminal**, asegúrate de activar el entorno virtual y corre:
```bash
python -m streamlit run dashboard.py
```
*(Esto desplegará una interfaz web premium en tu navegador local en `http://localhost:8501`).*

---

## 4. Como Validar el Modelo (Observabilidad y Trazabilidad)

Para evidenciar el correcto funcionamiento, trazabilidad y observabilidad sobre el agente de IA, sigue este protocolo de validación:

### A. Validación de Latencia (Tiempo de Respuesta)
- **Método**: Realizar una consulta en el chatbot que requiera consultar la base de datos (ej: *"¿Qué registros de otitis existen?"*).
- **Evidencia**: 
  - En la consola, notarás el tiempo de espera.
  - Abre el archivo `agent_logs.jsonl` generado y verifica el campo `"latency"`. Por ejemplo: `"latency": 3.4215` (segundos).
  - En el Dashboard de Streamlit, se graficará este valor. Si supera el umbral $\mu + 1.5 \times \sigma$, aparecerá listado automáticamente en la sección de anomalías de latencia.

### B. Validación de Rendimiento y Costos (Uso de Tokens)
- **Método**: Ingresa una pregunta y observa el desglose de tokens acumulados.
- **Evidencia**:
  - En la traza del JSONL, verás el objeto `"tokens"`, por ejemplo:
    ```json
    "tokens": {
      "input_tokens": 185,
      "output_tokens": 74,
      "total_tokens": 259
    }
    ```
  - En el dashboard, la gráfica de barra apilada mostrará la proporción exacta de tokens consumidos en prompts (costo de entrada) versus generación (costo de salida).

### C. Validación de Calidad (Groundedness Score contra Alucinaciones)
- **Método**: Realiza dos tipos de preguntas para comparar:
  - **Pregunta RAG Coherente**: *"¿Qué tratamiento se aplicó al paciente Toby?"* -> El agente responde en base al historial real de Toby en ChromaDB. El log registrará un `"groundedness_score"` alto (`0.80` a `1.00`), indicando alta fidelidad.
  - **Pregunta sin Contexto / Alucinación**: *"Háblame sobre física cuántica"* -> El agente responde con sus propios conocimientos generales. El log registrará un `"groundedness_score"` muy bajo (`0.00` a `0.20`), permitiendo detectar que no hay fundamentación en los documentos.

### D. Captura de Errores y Excepciones
- **Método**: Fuerza un error en el flujo (por ejemplo, desconectando internet o alterando la clave en el `.env`) y realiza una consulta.
- **Evidencia**:
  - El sistema capturará la excepción sin detener la consola y guardará el fallo en la traza:
    ```json
    "errors": ["Execution Error: Connection timed out"]
    ```
  - En el dashboard, la métrica de **Errores Detectados** se incrementará y los detalles aparecerán en la tabla de errores.

### E. Validación de Seguridad y Privacidad (Anonimización de PII)
- **Método**: Ingresa una consulta con datos sensibles falsos como:
  > *"Hola, mi email es contacto@cliente.com y mi RUT es 18.765.432-1, ¿cómo está Toby?"*
- **Evidencia**: En el archivo `agent_logs.jsonl` y en el LLM, los datos personales se reemplazarán automáticamente por sus etiquetas redactadas:
  ```json
  "user_input": "Hola, mi email es [REDACTED_EMAIL] y mi RUT es [REDACTED_RUT], ¿cómo está Toby?"
  ```

---

## 5. Arquitectura Detallada y Funcionamiento

### Estructura del Proyecto

```
AI-LLM-RAG-Obs-Traza-sobre-Agentes-IA/
|
|-- agent.py                     <- Agente principal (Tool-Calling Loop + sanitización + logs)
|-- tools.py                     <- 3 herramientas del agente: consulta, escritura, razonamiento
|-- short_term_memory.py         <- Memoria de corto plazo (buffer deslizante de mensajes k=5)
|-- memory_store.py              <- Memoria de largo plazo (persistencia en JSON + embeddings)
|-- dashboard.py                 <- Panel interactivo de monitoreo en Streamlit
|
|-- requirements.txt             <- Listado consolidado de dependencias
|-- agent_logs.jsonl             <- Archivo de logs estructurados (se genera dinámicamente y está en .gitignore)
|-- visit_summaries.jsonl        <- Registro de visitas guardadas por la herramienta (se genera solo)
|-- memories.json                <- Almacenamiento persistente de recuerdos (se genera solo)
|-- chroma_db_local/             <- Carpeta de base de datos vectorial ChromaDB (se genera sola)
|
|-- Informe_Tecnico_EP3.md       <- Plantilla del informe de la evaluación
|-- .env                         <- Credenciales locales (excluido en .gitignore por seguridad)
|-- .gitignore
+-- README.md
```

### Arquitectura de Flujo

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

## 6. Glosario de Términos

*   **PII (Personally Identifiable Information - Información de Identificación Personal)**: Cualquier dato que permita identificar de forma única a un individuo, como correos electrónicos, RUTs chilenos o números telefónicos.
*   **RAG (Retrieval-Augmented Generation)**: Técnica que complementa al LLM inyectándole información relevante recuperada en tiempo real desde ChromaDB para fundamentar sus respuestas.
*   **ChromaDB**: Base de datos vectorial utilizada de forma local para indexar y buscar de manera eficiente similitudes semánticas en historiales clínicos veterinarios.
*   **Embeddings**: Representaciones vectoriales numéricas que capturan el significado semántico de las palabras y textos. Este proyecto usa el modelo local `all-MiniLM-L6-v2`.
*   **Observabilidad**: La capacidad de inferir los estados internos y comportamientos de un sistema a partir de las métricas externas recopiladas en tiempo real.
*   **Trazabilidad**: El registro cronológico detallado de las decisiones, llamadas a APIs y ejecuciones de herramientas que sigue el agente para dar respuesta a un requerimiento.
*   **JSONL (JSON Lines)**: Formato de persistencia de archivos de texto estructurado donde cada línea representa un objeto JSON independiente.
*   **ReAct (Reasoning and Acting)**: Paradigma de agentes en el cual el LLM combina pasos de razonamiento lógico con el llamado a herramientas externas en bucle (Tool-Calling Loop).
*   **Groundedness**: Métrica que evalúa el nivel de consistencia y veracidad de la respuesta del agente respecto al contexto recuperado, previniendo alucinaciones.

---

## 7. Solución de Problemas Frecuentes

### Error: `streamlit : The term 'streamlit' is not recognized`
Este error ocurre cuando Windows no tiene registrado el comando directo `streamlit` en su ruta de sistema.
*   **Solución**: Ejecuta el dashboard llamando al módulo directamente a través de Python:
    ```bash
    python -m streamlit run dashboard.py
    ```

### Error: `No se encontro GITHUB_PAT_TOKEN en el archivo .env`
Verifica que has renombrado el archivo `.env.example` a `.env` (sin extensión `.txt`) y que contiene tu token sin comillas ni espacios.

### Error: `The models permission is required to access this endpoint`
Tu token de GitHub no tiene habilitado el permiso de acceso a modelos.
*   **Solución**: Ve a la pestaña **Account Permissions** en la configuración del token en GitHub y marca el permiso **Models** como **Read-only**.

---

## 8. Referencias Bibliográficas (APA)

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
*Desplegado con propósito académico - Ingeniería de Soluciones con IA, 2026.*  
*por Ricardo Antonio Galdames Soto*
