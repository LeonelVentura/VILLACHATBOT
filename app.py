import streamlit as st
import pandas as pd
import os
import re
from dotenv import load_dotenv
import fitz  # PyMuPDF
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import unicodedata
from openai import OpenAI
import tiktoken

# --- Configuración de la Aplicación ---
MAX_PDF_PAGES = 50         # Límite de páginas por PDF
CHUNK_SIZE = 500           # Tamaño de los fragmentos de texto
CHUNK_OVERLAP = 50         # Superposición para no perder contexto
MAX_CHUNKS_RETRIEVED = 10  # Máximo de fragmentos a recuperar con TF-IDF
MAX_CONTEXT_TOKENS = 12000 # Límite de tokens para el contexto enviado a GPT-4o

# --- Carga de la API KEY de OpenAI y Cliente ---
api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("❌ API key de OpenAI no encontrada. Por favor, configúrala en los secretos de Streamlit (OPENAI_API_KEY) o en un archivo .env.")
    st.stop()

client = OpenAI(api_key=api_key)
ENCODER = tiktoken.encoding_for_model("gpt-4o")

# --- Funciones de Utilidad y Normalización ---
def normalize_text(text):
    if not text:
        return ""
    text = text.lower()
    nfkd_form = unicodedata.normalize('NFKD', text)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

# --- Clasificador de Intención Mejorado ---
def get_question_intent(question):
    normalized_q = normalize_text(question)
    if any(keyword in normalized_q for keyword in ["profesor", "instructor", "docente"]) and any(intent_word in normalized_q for intent_word in ["quien", "nombre", "curso", "imparte", "da"]):
        professor_name = st.session_state.professor_name
        if professor_name != 'Desconocido':
            return "direct_answer", f"El profesor del curso es {professor_name}."
        else:
            return "direct_answer", "No tengo información específica sobre el profesor del curso. Te recomiendo revisar el programa del curso o contactar a la administración para obtener detalles."
    if any(keyword in normalized_q for keyword in ["que dia es hoy", "fecha actual", "cual es la fecha"]):
        hoy = datetime.now().strftime("%A, %d de %B de %Y")
        hora = datetime.now().strftime("%I:%M %p")
        return "direct_answer", f"Hoy es {hoy} y la hora es {hora}."
    if normalized_q in ["hola", "buenos dias", "buenas tardes", "buenas noches", "como estas"]:
        return "direct_answer", "¡Hola! Soy tu tutor inteligente. ¿En qué puedo ayudarte hoy?"
    irrelevant_keywords = [
        "partido de futbol", "goles", "reggaeton", "farandula", "chisme",
        "memes", "broma", "cuentame un chiste"
    ]
    if any(keyword in normalized_q for keyword in irrelevant_keywords):
        return "irrelevant", "Mi propósito es ayudarte con temas académicos de Ingeniería de Sistemas. No puedo responder sobre otros temas."
    general_keywords = [
        "quien es", "quien fue", "que es", "define", "cual es la capital de",
        "presidente de", "historia de", "resume la pelicula", "como funciona"
    ]
    academic_keywords = ["sistema", "software", "base de datos", "redes", "crm", "erp", "ciberseguridad"]
    if any(normalized_q.startswith(keyword) for keyword in general_keywords) and not any(acad_keyword in normalized_q for acad_keyword in academic_keywords):
        return "general_knowledge", None
    return "document_question", None

# --- Lógica de Procesamiento de Documentos (RAG) ---
@st.cache_data(show_spinner="Cargando y procesando materiales académicos...")
def load_and_chunk_documents():
    all_chunks = []
    pdf_files = sorted([f for f in os.listdir() if f.endswith(".pdf")])
    if not pdf_files:
        st.warning("⚠️ No se encontraron archivos PDF en el directorio.")
        return []
    for filename in pdf_files:
        try:
            with fitz.open(filename) as doc:
                full_text = " ".join(page.get_text("text") for i, page in enumerate(doc) if i < MAX_PDF_PAGES)
                full_text = re.sub(r'\s+', ' ', full_text).strip()
                words = full_text.split()
                for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
                    chunk_text = " ".join(words[i:i + CHUNK_SIZE])
                    all_chunks.append({"text": chunk_text, "source": filename})
        except Exception as e:
            st.error(f"❌ Error procesando el archivo {filename}: {e}")
    return all_chunks

def get_relevant_chunks(chunks, question):
    if not chunks: return []
    chunk_texts = [chunk["text"] for chunk in chunks]
    all_texts = [question] + chunk_texts
    try:
        vectorizer = TfidfVectorizer().fit_transform(all_texts)
        similarities = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()
        relevant_indices = np.argsort(similarities)[::-1][:MAX_CHUNKS_RETRIEVED]
        return [chunks[i] for i in relevant_indices if similarities[i] > 0.1]
    except ValueError:
        return []

# --- Interacción con la API de OpenAI ---
def get_openai_response(system_prompt, user_prompt, chat_history):
    messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_prompt})
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.3,
            max_tokens=1024,
            top_p=0.9
        )
        return response.choices[0].message.content
    except Exception as e:
        if "rate_limit_exceeded" in str(e):
            return "⚠️ Límite de uso de la API excedido. Por favor, espera un momento."
        return f"❌ Ocurrió un error con la API de OpenAI: {e}"

# --- Lógica de Autenticación ---
def validar_estudiante(codigo):
    try:
        codigo = str(codigo).strip()
        excel_path = "Estudiante.xlsx"
        if not os.path.exists(excel_path):
            st.error(f"❌ Archivo '{excel_path}' no encontrado.")
            return False, None, 'Desconocido'
        df = pd.read_excel(excel_path)
        match = df[df['codigo'].astype(str).str.strip() == codigo]
        if not match.empty:
            nombre = f"{match.iloc[0]['nombre']} {match.iloc[0]['apellido']}"
            professor = match.iloc[0]['profesor'] if 'profesor' in df.columns else 'Desconocido'
            return True, nombre, professor
        return False, None, 'Desconocido'
    except Exception as e:
        st.error(f"❌ Error de validación: {e}")
        return False, None, 'Desconocido'

# --- Interfaz Principal ---
def main():
    st.set_page_config(page_title="Tutor Inteligente", page_icon="🧠", layout="centered")
    st.markdown("""
    <style>
        :root {
            --primary-color: #4361ee;
            --light-bg: #f8f9fa;
            --user-bubble-bg: #e3f2fd;
            --assistant-bubble-bg: #ffffff;
        }
        .stApp { background-color: var(--light-bg); }
        .stChatMessage { border-radius: 18px; padding: 1rem; margin-bottom: 1rem; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
        .stChatMessage[data-testid="stChatMessage-user"] { background-color: var(--user-bubble-bg); }
        .stChatMessage[data-testid="stChatMessage-assistant"] { background-color: var(--assistant-bubble-bg); }
        .stButton>button { border-radius: 12px; font-weight: bold; }
        h1 { color: var(--primary-color); text-align: center; font-weight: 700; }
    </style>
    """, unsafe_allow_html=True)
    st.title("🧠 Tutor Inteligente con GPT-4o")

    # Inicialización del estado de la sesión
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.student_name = ""
        st.session_state.professor_name = "Desconocido"
        st.session_state.messages = [{"role": "assistant", "content": "¡Hola! Por favor, valida tu código de estudiante para comenzar."}]
        st.session_state.chunks = []

    # Autenticación
    if not st.session_state.authenticated:
        st.subheader("🔒 Acceso de Estudiante")
        with st.form("login_form"):
            codigo = st.text_input("Código de estudiante:", type="password")
            submitted = st.form_submit_button("Validar Identidad", use_container_width=True)
            if submitted:
                is_valid, student_name, professor_name = validar_estudiante(codigo)
                if is_valid:
                    st.session_state.authenticated = True
                    st.session_state.student_name = student_name
                    st.session_state.professor_name = professor_name
                    st.session_state.chunks = load_and_chunk_documents()
                    st.session_state.messages = [{"role": "assistant", "content": f"¡Bienvenido, {student_name}! Los materiales están listos. ¿En qué puedo ayudarte?"}]
                    st.success(f"✅ Acceso concedido a {student_name}.")
                    st.rerun()
                else:
                    st.error("❌ Código no válido. Inténtalo de nuevo.")
        return

    # Interfaz de chat
    st.subheader(f"Estudiante: {st.session_state.student_name}")
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="👨‍🎓" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"])

    if prompt := st.chat_input(f"Pregúntame algo, {st.session_state.student_name.split()[0]}..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👨‍🎓"):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Pensando..."):
                intent, direct_answer = get_question_intent(prompt)
                response_content = ""
                if intent == "direct_answer" or intent == "irrelevant":
                    response_content = direct_answer
                elif intent == "general_knowledge":
                    system_prompt = "Eres un asistente experto y amigable. Responde a la pregunta del usuario de forma clara, concisa y educativa, utilizando tu conocimiento general. Actúa como un profesor universitario."
                    response_content = get_openai_response(system_prompt, prompt, st.session_state.messages[:-1])
                elif intent == "document_question":
                    relevant_chunks = get_relevant_chunks(st.session_state.chunks, prompt)
                    if not relevant_chunks:
                        system_prompt = "Eres un profesor de Ingeniería de Sistemas. Informa al usuario que no encontraste información en los documentos sobre su pregunta, pero ofrécele una breve explicación general sobre el tema usando tu propio conocimiento. Sé amable y servicial."
                        context_text = ""
                    else:
                        context_text = ""
                        prompt_tokens = len(ENCODER.encode(prompt))
                        tokens_available = MAX_CONTEXT_TOKENS - prompt_tokens - 1500
                        for chunk in relevant_chunks:
                            chunk_content = f"Fuente: {chunk['source']}\nContenido: {chunk['text']}\n\n---\n\n"
                            chunk_tokens = len(ENCODER.encode(chunk_content))
                            if tokens_available - chunk_tokens >= 0:
                                context_text += chunk_content
                                tokens_available -= chunk_tokens
                            else:
                                break
                    system_prompt = f"""
                    Eres un profesor experto en Ingeniería de Sistemas. Tu tarea es responder la pregunta del usuario basándote ESTRICTAMENTE en los siguientes fragmentos de texto.

                    Instrucciones:
                    1. Sintetiza la información de los fragmentos para dar una respuesta coherente y precisa.
                    2. Si los fragmentos no contienen la respuesta, di: "No encontré información específica sobre eso en los materiales, pero te puedo dar una explicación general...".
                    3. Al final de tu respuesta, CITA TUS FUENTES de forma clara. Ejemplo: (Fuente: NombreDelArchivo.pdf).
                    4. No inventes información que no esté en los textos.

                    Fragmentos de contexto:
                    {context_text if context_text else "No se encontró información relevante en los documentos."}
                    """
                    response_content = get_openai_response(system_prompt, prompt, st.session_state.messages[:-1])
                st.markdown(response_content)
                st.session_state.messages.append({"role": "assistant", "content": response_content})

if __name__ == "__main__":
    main()
