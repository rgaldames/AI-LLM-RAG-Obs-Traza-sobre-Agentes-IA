"""
agent.py  Agente Funcional Veterinario (Evaluación Parcial 2)
=============================================================
Implementa un agente funcional usando la API moderna de LangChain 1.2+
basada en LangGraph (create_agent con tool-calling loop).

Arquitectura:
  - LangChain 1.2.15 / LangGraph 1.1.8
  - Patrón: Tool-Calling Loop (equivalente moderno al ReAct)
  - Herramientas: consulta (RAG), escritura (JSONL), razonamiento (reglas)
  - Memoria: corto plazo (buffer de mensajes) + largo plazo (embeddings JSON)

IL2.1  Herramientas de consulta, escritura y razonamiento
IL2.2  Memoria de corto y largo plazo
IL2.3  Planificación y toma de decisiones adaptativas
IL2.4  Documentación técnica y orquestación de componentes

Flujo de Orquestación:
  Usuario
     Recuperar contexto largo plazo (MemoryStore)
     Construir historial + system prompt contextual
     Agente (LLM + herramientas en loop)
         LLM decide si llamar herramienta
         [search_clinical_db | write_visit_summary | analyze_symptoms]
         Observación  LLM decide continuar o responder
     Respuesta final
     Guardar en corto plazo (buffer) + largo plazo (embeddings)
"""

import os
import sys
from dotenv import load_dotenv

# -- 1. CONFIGURACIÓN DE ENTORNO ---------------------------------------------
load_dotenv()
GITHUB_PAT_TOKEN = os.getenv("GITHUB_PAT_TOKEN")
if not GITHUB_PAT_TOKEN:
    raise ValueError("Error: No se encontro GITHUB_PAT_TOKEN en el archivo .env")

os.environ["OPENAI_API_KEY"] = GITHUB_PAT_TOKEN

# -- 2. IMPORTACIONES --------------------------------------------------------
import re
import json
import time
import uuid
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_chroma import Chroma
from langchain.agents import create_agent          # API nueva LangChain 1.2+
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.callbacks import get_openai_callback

from tools import make_search_tool, write_visit_summary, analyze_symptoms
from memory_store import MemoryStore
from short_term_memory import ShortTermMemory

# -- 2.1. SANITIZACIÓN Y MÉTRICAS (Seguridad y Privacidad - EP3) ---------------
def sanitize_pii(text: str) -> str:
    """
    Detecta y oculta información personal (PII) usando Expresiones Regulares.
    Reemplaza correos electrónicos por [REDACTED_EMAIL] y RUTs chilenos por [REDACTED_RUT].
    """
    if not isinstance(text, str):
        return text
    
    # Expresión regular para correos electrónicos
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    # Expresión regular para RUT chileno (con o sin puntos, con guion opcional y dígito verificador)
    rut_pattern = r'\b\d{1,2}(?:\.?\d{3}){2}-?[0-9kK]\b'
    
    text = re.sub(email_pattern, "[REDACTED_EMAIL]", text)
    text = re.sub(rut_pattern, "[REDACTED_RUT]", text)
    return text

def calculate_groundedness(response: str, context: str) -> float:
    """
    Calcula una métrica heurística de groundedness (coherencia de RAG)
    basada en la coincidencia de palabras significativas entre el contexto y la respuesta.
    """
    if not context or not response:
        return 1.0  # Si no hay contexto de RAG para validar, asumimos que no aplica/coherente.
    
    def get_clean_words(text):
        # Elimina caracteres especiales y se queda con palabras de longitud > 3
        text_clean = re.sub(r'[^\w\s]', '', text.lower())
        return set(word for word in text_clean.split() if len(word) > 3)
    
    resp_words = get_clean_words(response)
    ctx_words = get_clean_words(context)
    
    if not resp_words:
        return 0.0
    
    intersection = resp_words.intersection(ctx_words)
    return float(len(intersection) / len(resp_words))


# -- 3. RUTAS DE ARCHIVOS ----------------------------------------------------
CSV_PATH      = "./veterinary_clinical_data.csv"
CHROMA_DB_DIR = "./chroma_db_local"
MEMORY_PATH   = "./memories.json"

# -- 4. SYSTEM PROMPT DEL AGENTE ---------------------------------------------
# Prompt simplificado para evitar que el LLM entre en loops de tool-calling.
# El modelo decide POR SÍ MISMO si necesita o no llamar una herramienta.
SYSTEM_PROMPT = """Eres VetBot, un asistente de inteligencia artificial para una clinica veterinaria.
Ayudas a duenos de mascotas con consejos de cuidado, informacion clinica y orientacion.

REGLAS:
- Jamas recetes medicamentos ni emitas diagnosticos medicos definitivos.
- Si hay sintomas graves, recomienda ir a urgencias veterinarias.
- Usa las herramientas disponibles SOLO cuando sea necesario para responder mejor.
- Si puedes responder directamente sin herramientas, hazlo sin llamarlas.
- Responde siempre en espanol, de forma clara y amigable.

Herramientas disponibles (usarlas solo si aportan valor real a la respuesta):
- search_clinical_db: busca en historiales clinicos de la BD interna
- analyze_symptoms: evalua nivel de urgencia de sintomas descritos
- write_visit_summary: guarda un resumen al FINAL de una consulta importante"""


def initialize_components():
    """
    Inicializa todos los componentes del agente de forma modular.

    Returns:
        Tupla con (agente_compilado, memoria_corto_plazo, memoria_largo_plazo)
    """
    print("[1/5] Configurando modelo de lenguaje (gpt-4o-mini)...")
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        base_url="https://models.inference.ai.azure.com"
    )

    print("[2/5] Cargando modelo de embeddings local (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("[3/5] Indexando base de conocimiento clinico en ChromaDB...")
    if not os.path.exists(CSV_PATH):
        print(f"Error: No se encontro el archivo CSV: {CSV_PATH}")
        sys.exit(1)

    loader = CSVLoader(file_path=CSV_PATH, encoding="utf-8")
    documents = loader.load()
    documents = documents[:50]
    print(f"     {len(documents)} registros clinicos indexados.")

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )

    print("[4/5] Inicializando sistema de memoria dual...")
    # Corto plazo: buffer de ventana deslizante (últimas 5 interacciones)
    short_term = ShortTermMemory(window_size=5)
    # Largo plazo: almacenamiento semántico persistente entre sesiones
    long_term = MemoryStore(embeddings, path=MEMORY_PATH)
    print("     Corto plazo: ConversationBufferWindowMemory (k=5)")
    print("     Largo plazo: MemoryStore JSON + embeddings cosine similarity")

    print("[5/5] Construyendo agente con herramientas (LangChain 1.2 create_agent)...")
    # Herramienta de consulta vinculada al vectorstore activo
    search_tool = make_search_tool(vectorstore)

    # Las 3 herramientas del agente (IL2.1)
    tools = [
        search_tool,          # CONSULTA: RAG sobre historiales clinicos
        analyze_symptoms,     # RAZONAMIENTO: deteccion de urgencia por reglas
        write_visit_summary,  # ESCRITURA: persistencia de consultas
    ]

    # Agente moderno LangChain 1.2+ (equivalente a ReAct, basado en LangGraph)
    # El LLM llama herramientas en loop hasta que no necesita mas informacion
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )
    print("     Agente compilado con 3 herramientas.")

    return agent, short_term, long_term


def run_agent(agent, short_term: ShortTermMemory, long_term: MemoryStore,
              user_input: str, session_id: str = "default-session") -> str:
    """
    Ejecuta el agente integrando memoria dual en cada invocación,
    aplicando sanitización de PII, medición de latencia, rastreo de tokens y logging estructurado.
    """
    # Paso 0: Sanitización de PII en la entrada del usuario (Seguridad - EP3)
    user_input_sanitized = sanitize_pii(user_input)
    
    start_time = time.time()
    errors = []
    tools_used = []
    tokens = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    context_extracted = ""

    # Paso 1: Recuperar memorias relevantes de largo plazo
    past_memories = long_term.retrieve(user_input_sanitized, top_k=3)
    long_term_context = ""
    if past_memories:
        long_term_context = "\n".join(f"- {m['text']}" for m in past_memories)
        context_extracted += long_term_context + "\n"

    # Paso 2: Construir el input para el agente.
    if long_term_context:
        enriched_input = (
            f"{user_input_sanitized}\n\n"
            f"[Contexto de sesiones anteriores relevante:]\n{long_term_context}"
        )
    else:
        enriched_input = user_input_sanitized

    # Paso 3: Invocar el agente incluyendo la memoria de corto plazo y el callback de tokens
    response = ""
    try:
        messages_history = short_term.get_messages()
        messages_history.append(HumanMessage(content=enriched_input))
        
        with get_openai_callback() as cb:
            result = agent.invoke({"messages": messages_history})
            
            # Obtener tokens consumidos
            tokens["input_tokens"] = cb.prompt_tokens
            tokens["output_tokens"] = cb.completion_tokens
            tokens["total_tokens"] = cb.total_tokens

        # Fallback de estimación de tokens si el callback de OpenAI no los registra (por endpoint personalizado)
        if tokens["total_tokens"] == 0:
            prompt_est = len(enriched_input) // 4
            tokens["input_tokens"] = prompt_est
            # estimación simple para la respuesta más adelante

        output_messages = result.get("messages", [])
        
        # Extraer herramientas utilizadas y contenidos de contexto / errores
        for msg in output_messages:
            msg_type = msg.__class__.__name__
            
            # Captura de llamadas a herramientas
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    t_name = tc.get("name")
                    if t_name and t_name not in tools_used:
                        tools_used.append(t_name)
            
            # Captura de ejecuciones y sus retornos (contextos y errores de herramientas)
            if msg_type == "ToolMessage":
                t_name = getattr(msg, "name", "")
                if t_name and t_name not in tools_used:
                    tools_used.append(t_name)
                
                content_str = str(getattr(msg, "content", ""))
                # Si fue RAG, extraemos el contenido como contexto de RAG
                if t_name == "search_clinical_db":
                    context_extracted += content_str + "\n"
                
                # Rastrear fallos en herramientas
                if "[ERROR]" in content_str or "Error" in content_str:
                    errors.append(f"Tool Error ({t_name}): {content_str}")

        # Extraer respuesta final del agente
        for msg in reversed(output_messages):
            msg_type = msg.__class__.__name__
            content = getattr(msg, "content", "")
            tool_calls = getattr(msg, "tool_calls", [])
            if msg_type == "AIMessage" and content and not tool_calls:
                response = content
                break
        if not response:
            for msg in reversed(output_messages):
                if getattr(msg, "content", ""):
                    response = str(msg.content)
                    break
        if not response:
            response = "No pude generar una respuesta. Por favor intenta de nuevo."

        # Sanitizar respuesta final por seguridad
        response = sanitize_pii(response)

    except Exception as e:
        error_msg = str(e)
        errors.append(f"Execution Error: {error_msg}")
        response = (
            f"Ocurrio un error al procesar tu consulta: {error_msg}\n"
            "Por favor, intenta reformular tu pregunta."
        )

    # Si se usó estimación de tokens
    if tokens["total_tokens"] == 0:
        response_est = len(response) // 4
        tokens["output_tokens"] = response_est
        tokens["total_tokens"] = tokens["input_tokens"] + response_est

    # Calcular latencia (Observabilidad - EP3)
    latency = time.time() - start_time

    # Calcular groundedness (Calidad - EP3)
    groundedness = calculate_groundedness(response, context_extracted.strip())

    # Paso 4: Guardar en memoria de corto plazo
    short_term.save_turn(user_input_sanitized, response)

    # Paso 5: Guardar resumen en memoria de largo plazo
    try:
        combined = f"Consulta: {user_input_sanitized[:800]} | Respuesta: {response[:800]}"
        long_term.add_memory(
            combined,
            metadata={"role": "interaction", "turn": short_term.turn_count}
        )
    except Exception:
        pass

    # Paso 6: Guardar logs estructurados en agent_logs.jsonl (Trazabilidad - EP3)
    log_record = {
        "session_id": session_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "user_input": user_input_sanitized,
        "agent_response": response,
        "tools_used": tools_used,
        "latency": round(latency, 4),
        "tokens": tokens,
        "groundedness_score": round(groundedness, 4),
        "errors": errors
    }

    try:
        with open("./agent_logs.jsonl", "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(log_record, ensure_ascii=False) + "\n")
    except Exception as log_err:
        print(f"Error escribiendo en agent_logs.jsonl: {log_err}")

    return response



def print_status(short_term: ShortTermMemory):
    """Imprime el estado actual de la memoria del agente."""
    stats = short_term.get_summary_stats()
    print(
        f"\n>> Estado de Memoria: "
        f"Turno #{stats['total_turns_session']} | "
        f"Buffer: {stats['messages_in_buffer']}/{stats['window_size']*2} msgs | "
        f"{'[LLENO]' if stats['buffer_full'] else '[ACTIVO]'}"
    )


def main():
    """Punto de entrada principal del agente funcional veterinario."""
    print("=" * 65)
    print("  AGENTE VETERINARIO FUNCIONAL - Evaluacion Parcial 2")
    print("=" * 65)
    print("  Framework  : LangChain 1.2 + LangGraph (create_agent)")
    print("  Herramientas: Consulta | Escritura | Razonamiento")
    print("  Memoria    : Corto Plazo (buffer) + Largo Plazo (semantico)")
    print("=" * 65)
    print()

    agent, short_term, long_term = initialize_components()
    
    # Generar ID de sesión único al inicio de la sesión
    session_id = str(uuid.uuid4())

    print()
    print("Agente listo! (Escribe 'salir' para terminar | 'estado' para ver memoria)\n")
    print("-" * 65)

    # -- Bucle principal de interaccion --------------------------------------
    while True:
        try:
            user_input = input("\nUsuario: ").strip()

            # Comandos especiales
            if not user_input:
                continue
            if user_input.lower() in ["salir", "exit", "quit"]:
                print("\nAgente apagandose. Hasta pronto!")
                break
            if user_input.lower() == "estado":
                print_status(short_term)
                continue
            if user_input.lower() == "historial":
                hist = short_term.get_history_as_text()
                print(f"\nHistorial reciente:\n{hist if hist else '(vacio)'}")
                continue

            # Ejecutar el agente
            print("\n[Agente procesando...]\n")
            response = run_agent(agent, short_term, long_term, user_input, session_id=session_id)

            print("\n" + "=" * 65)
            print("VetBot responde:")
            print("-" * 65)
            print(response)
            print("=" * 65)

        except KeyboardInterrupt:
            print("\n\nSesion interrumpida. Hasta luego!")
            break
        except Exception as e:
            print(f"\nError inesperado: {e}")
            print("Intenta nuevamente o escribe 'salir' para terminar.")


if __name__ == "__main__":
    main()
