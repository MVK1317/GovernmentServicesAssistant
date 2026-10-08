import os
import glob
import time
import numpy as np
import streamlit as st

from dotenv import load_dotenv
from pypdf import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
from google import genai


# =========================================================
# 1. CONFIGURATION
# =========================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("❌ Gemini API key not found in .env file.")
    st.stop()

client = genai.Client(api_key=API_KEY)

st.set_page_config(
    page_title="Government Services Assistant",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. PROFESSIONAL UI
# =========================================================

st.markdown("""
<style>

/* =====================================================
   MAIN BACKGROUND
===================================================== */

.stApp {
    background:
        linear-gradient(
            135deg,
            #eef5ff 0%,
            #f7faff 45%,
            #edf6f4 100%
        ) !important;

    color: #1e293b !important;
}


/* =====================================================
   HIDE STREAMLIT DEFAULT UI
===================================================== */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}


/* =====================================================
   GENERAL TEXT
===================================================== */

.stMarkdown,
.stMarkdown p,
.stMarkdown li,
.stMarkdown span {
    color: #1e293b !important;
}

h1,
h2,
h3,
h4,
h5,
h6 {
    color: #0f172a !important;
}


/* =====================================================
   GOVERNMENT HEADER
===================================================== */

.gov-header {

    background: linear-gradient(
        135deg,
        #0b1f3a,
        #123d6a,
        #164e63
    );

    padding: 28px 32px;

    border-radius: 18px;

    margin-bottom: 25px;

    box-shadow:
        0 8px 25px rgba(15, 23, 42, 0.15);
}

.gov-title {

    color: white !important;

    font-size: 30px;

    font-weight: 700;

    margin-bottom: 6px;
}

.gov-subtitle {

    color: #dbeafe !important;

    font-size: 15px;
}


/* =====================================================
   WELCOME CARD
===================================================== */

.welcome-card {

    background: rgba(255, 255, 255, 0.92);

    padding: 28px;

    border-radius: 18px;

    border: 1px solid #dbeafe;

    box-shadow:
        0 6px 20px rgba(30, 64, 175, 0.07);

    margin-bottom: 25px;
}

.welcome-title {

    font-size: 24px;

    font-weight: 700;

    color: #172554 !important;

    margin-bottom: 8px;
}

.welcome-text {

    color: #475569 !important;

    font-size: 15px;

    line-height: 1.6;
}


/* =====================================================
   SERVICE CARDS
===================================================== */

.service-card {

    background: rgba(255, 255, 255, 0.94);

    padding: 20px;

    border-radius: 16px;

    border: 1px solid #dbeafe;

    min-height: 125px;

    box-shadow:
        0 5px 16px rgba(30, 64, 175, 0.06);

    margin-bottom: 10px;

    transition: 0.2s;
}

.service-card:hover {

    transform: translateY(-2px);

    box-shadow:
        0 8px 20px rgba(30, 64, 175, 0.10);
}

.service-icon {

    font-size: 28px;

    margin-bottom: 8px;
}

.service-title {

    font-size: 17px;

    font-weight: 600;

    color: #1e293b !important;

    margin-bottom: 5px;
}

.service-description {

    font-size: 13px;

    color: #64748b !important;

    line-height: 1.4;
}


/* =====================================================
   CHAT MESSAGE
===================================================== */

[data-testid="stChatMessage"] {

    border-radius: 16px;

    color: #1e293b !important;
}


/* Assistant/user message text */

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span {

    color: #1e293b !important;
}


/* =====================================================
   CHAT INPUT
===================================================== */

[data-testid="stChatInput"] {

    background-color: rgba(255,255,255,0.96) !important;

    border-radius: 16px;

    border: 1px solid #cbd5e1;
}

[data-testid="stChatInput"] textarea {

    color: #1e293b !important;

    background-color: white !important;
}

[data-testid="stChatInput"] textarea::placeholder {

    color: #64748b !important;
}


/* =====================================================
   SIDEBAR
===================================================== */

section[data-testid="stSidebar"] {

    background:
        linear-gradient(
            180deg,
            #f8fbff,
            #eef5fb
        ) !important;
}

section[data-testid="stSidebar"] * {

    color: #1e293b !important;
}

.sidebar-title {

    font-size: 21px;

    font-weight: 700;

    color: #0f172a !important;
}

.sidebar-info {

    font-size: 13px;

    color: #64748b !important;

    line-height: 1.5;
}


/* =====================================================
   BUTTONS
===================================================== */

.stButton > button {

    border-radius: 11px;

    border: 1px solid #cbd5e1;

    min-height: 42px;

    font-weight: 500;

    background-color: rgba(255,255,255,0.92);

    color: #1e293b !important;

    transition: 0.2s;
}

.stButton > button:hover {

    border-color: #2563eb;

    color: #1d4ed8 !important;

    background-color: #eff6ff;
}


/* =====================================================
   EXPANDER
===================================================== */

[data-testid="stExpander"] {

    background-color: rgba(255,255,255,0.94) !important;

    border: 1px solid #dbeafe !important;

    border-radius: 12px !important;
}

[data-testid="stExpander"] * {

    color: #1e293b !important;
}


/* =====================================================
   INFO BOX
===================================================== */

[data-testid="stAlert"] {

    color: #1e293b !important;

    background-color: rgba(255,255,255,0.80);
}

[data-testid="stAlert"] p {

    color: #1e293b !important;
}


/* =====================================================
   LOADING / SPINNER
===================================================== */

[data-testid="stSpinner"] {

    color: #1d4ed8 !important;
}


/* =====================================================
   STATUS AREA
===================================================== */

.search-status {

    background: rgba(239, 246, 255, 0.95);

    border: 1px solid #bfdbfe;

    padding: 12px 16px;

    border-radius: 12px;

    color: #1e40af !important;

    font-size: 14px;

    margin: 8px 0 15px 0;
}


/* =====================================================
   FOOTER
===================================================== */

.gov-footer {

    text-align: center;

    color: #64748b !important;

    font-size: 12px;

    padding: 25px;

    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. PDF TEXT EXTRACTION
# =========================================================

def extract_pdf_text(pdf_path):

    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        if text and text.strip():

            pages.append({

                "page": page_number,

                "text": text

            })

    return pages


# =========================================================
# 4. LOAD GOVERNMENT DOCUMENTS
# =========================================================

documents = []

pdf_files = glob.glob(
    "government_documents/*.pdf"
)


for pdf_file in pdf_files:

    try:

        pages = extract_pdf_text(
            pdf_file
        )

        for page in pages:

            documents.append({

                "source": os.path.basename(
                    pdf_file
                ),

                "page": page["page"],

                "text": page["text"]

            })

    except Exception as e:

        st.warning(
            f"Could not read "
            f"{os.path.basename(pdf_file)}: {e}"
        )


if len(documents) == 0:

    st.error(
        "❌ No government PDF documents found."
    )

    st.info(
        "Put your government PDF files inside "
        "the government_documents folder."
    )

    st.stop()


# =========================================================
# 5. CREATE TEXT CHUNKS
# =========================================================

def create_chunks(
    text,
    chunk_size=500,
    overlap=50
):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(
            words[start:end]
        )

        if chunk.strip():

            chunks.append(
                chunk
            )

        start += (
            chunk_size - overlap
        )

    return chunks


chunk_database = []


for document in documents:

    chunks = create_chunks(
        document["text"]
    )

    for chunk_id, chunk in enumerate(
        chunks
    ):

        chunk_database.append({

            "source": document["source"],

            "page": document["page"],

            "chunk_id": chunk_id,

            "text": chunk

        })


# =========================================================
# 6. GEMINI EMBEDDINGS
# =========================================================

def get_embedding(text):

    result = client.models.embed_content(

        model="gemini-embedding-001",

        contents=text

    )

    return np.array(
        result.embeddings[0].values
    )


@st.cache_data
def create_embeddings(texts):

    embeddings = []

    for text in texts:

        embedding = get_embedding(
            text
        )

        embeddings.append(
            embedding
        )

    return np.array(
        embeddings
    )


texts = [
    item["text"]
    for item in chunk_database
]


# =========================================================
# 7. CREATE EMBEDDINGS
# =========================================================

with st.spinner(
    "🔄 Preparing government knowledge base..."
):

    embedding_matrix = create_embeddings(
        texts
    )


# =========================================================
# 8. DOCUMENT RETRIEVAL
# =========================================================

def retrieve_documents(
    question,
    top_k=5
):

    query_embedding = get_embedding(
        question
    )

    scores = cosine_similarity(
        [query_embedding],
        embedding_matrix
    )[0]

    top_indices = np.argsort(
        scores
    )[::-1][:top_k]


    results = []


    for index in top_indices:

        item = chunk_database[
            index
        ].copy()

        item["score"] = float(
            scores[index]
        )

        results.append(
            item
        )


    # Remove very weak matches

    filtered_results = [

        result

        for result in results

        if result["score"] >= 0.25

    ]


    return filtered_results


# =========================================================
# 9. BUILD CONTEXT
# =========================================================

def build_context(results):

    if not results:

        return (
            "NO RELEVANT INFORMATION WAS FOUND "
            "IN THE PROVIDED GOVERNMENT DOCUMENTS."
        )


    context = ""


    for number, result in enumerate(
        results,
        start=1
    ):

        context += f"""

SOURCE {number}

Document:
{result["source"]}

Page:
{result["page"]}

Relevance Score:
{result["score"]:.4f}

Content:
{result["text"]}

--------------------------------
"""


    return context


# =========================================================
# 10. GOVERNMENT RAG + GEMINI
# =========================================================

def government_rag(
    question,
    chat_history,
    language
):

    results = retrieve_documents(
        question,
        top_k=5
    )


    context = build_context(
        results
    )


    # Conversation history

    history_text = ""


    for message in chat_history[-6:]:

        history_text += (

            f'{message["role"].upper()}: '

            f'{message["content"]}\n'

        )


    # Language

    if language == "English":

        language_instruction = """
Answer in English.
"""

    elif language == "Telugu":

        language_instruction = """
Answer in Telugu.
Use natural and easy-to-understand Telugu.
Keep important official terms in English
inside brackets when useful.
"""

    else:

        language_instruction = """
Answer in Hindi.
Use natural and easy-to-understand Hindi.
Keep important official terms in English
inside brackets when useful.
"""


    # Prompt

    prompt = f"""

You are an intelligent Government Services Assistant.

You help citizens understand government services using
ONLY the government documents provided to you.

Your responses should feel natural, friendly,
professional and human.

==================================================
LANGUAGE
==================================================

{language_instruction}

==================================================
IMPORTANT KNOWLEDGE RULES
==================================================

1. Use ONLY the provided government documents
   as the factual source.

2. Do NOT use outside knowledge.

3. Do NOT invent government rules.

4. Do NOT invent eligibility requirements.

5. Do NOT invent required documents.

6. Do NOT invent fees.

7. Do NOT invent processing times.

8. Do NOT invent application procedures.

9. Do NOT assume information that is not present.

10. If the requested information is not available,
    clearly say:

"The requested information was not found in the
provided government documents."

11. Ignore retrieved content that is clearly unrelated
    to the user's question.

12. Do not combine information from different
    government services unless the documents clearly
    connect them.

==================================================
ANSWER STYLE
==================================================

- Be conversational.
- Be helpful.
- Be concise but informative.
- Use bullet points when useful.
- Use numbered steps for procedures.
- Avoid unnecessary repetition.
- Do not sound robotic.
- Explain difficult terms simply.
- Mention source document and page when appropriate.

==================================================
FOLLOW-UP QUESTIONS
==================================================

The conversation history is ONLY for understanding
the user's follow-up question.

Previous AI answers are NOT factual sources.

The government documents remain the source of truth.

==================================================
CONVERSATION HISTORY
==================================================

{history_text}

==================================================
CURRENT USER QUESTION
==================================================

{question}

==================================================
GOVERNMENT DOCUMENT INFORMATION
==================================================

{context}

==================================================

Now answer the user naturally and accurately.
"""


    response = client.models.generate_content(

        model="gemini-3.1-flash-lite",

        contents=prompt

    )


    return (
        response.text,
        results
    )


# =========================================================
# 11. SESSION STATE
# =========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if "pending_question" not in st.session_state:

    st.session_state.pending_question = None


# =========================================================
# 12. SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">'
        '🏛️ Citizen Services'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="sidebar-info">'
        'AI-powered government information assistant'
        '</div>',
        unsafe_allow_html=True
    )


    st.divider()


    # Language

    st.markdown(
        "### 🌐 Language"
    )


    language = st.selectbox(

        "Choose response language",

        [
            "English",
            "Telugu",
            "Hindi"
        ],

        index=0

    )


    st.divider()


    # New Chat

    if st.button(
        "➕ New Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.session_state.pending_question = None

        st.rerun()


    st.divider()


    # Knowledge Base

    st.markdown(
        "### 📊 Knowledge Base"
    )


    st.write(
        f"📄 Government PDFs: "
        f"**{len(pdf_files)}**"
    )


    st.write(
        f"📑 Pages indexed: "
        f"**{len(documents)}**"
    )


    st.write(
        f"🧩 Knowledge chunks: "
        f"**{len(chunk_database)}**"
    )


    st.divider()


    # AI System

    st.markdown(
        "### ⚙️ AI System"
    )


    st.write(
        "🔎 Semantic Retrieval"
    )

    st.write(
        "🧠 Gemini Embeddings"
    )

    st.write(
        "🤖 Gemini LLM"
    )

    st.write(
        "📚 Source-grounded Answers"
    )


    st.divider()


    # Clear Chat

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.session_state.pending_question = None

        st.rerun()


    st.divider()


    st.caption(
        "Answers are generated from the government "
        "documents available to this assistant."
    )


    st.caption(
        "For important applications, verify final "
        "details with the relevant official department."
    )


# =========================================================
# 13. HEADER
# =========================================================

st.markdown(

    '<div class="gov-header">'

    '<div class="gov-title">'
    '🏛️ Government Services Assistant'
    '</div>'

    '<div class="gov-subtitle">'
    'Digital Citizen Services • '
    'Document-Grounded AI Assistance'
    '</div>'

    '</div>',

    unsafe_allow_html=True
)


# =========================================================
# 14. WELCOME SCREEN
# =========================================================

if len(
    st.session_state.messages
) == 0:


    st.markdown(

        '<div class="welcome-card">'

        '<div class="welcome-title">'
        '👋 Welcome to the Citizen Services Assistant'
        '</div>'

        '<div class="welcome-text">'
        'Ask questions about government services, '
        'eligibility, required documents, application '
        'procedures, fees and other information available '
        'in the uploaded government documents.'
        '</div>'

        '</div>',

        unsafe_allow_html=True
    )


    # Popular Services

    st.markdown(
        "### 🔎 Popular Services"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.markdown(

            '<div class="service-card">'

            '<div class="service-icon">'
            '🗳️'
            '</div>'

            '<div class="service-title">'
            'Voter Services'
            '</div>'

            '<div class="service-description">'
            'Registration and required documents'
            '</div>'

            '</div>',

            unsafe_allow_html=True
        )


    with col2:

        st.markdown(

            '<div class="service-card">'

            '<div class="service-icon">'
            '🛂'
            '</div>'

            '<div class="service-title">'
            'Passport Services'
            '</div>'

            '<div class="service-description">'
            'Guidelines and application information'
            '</div>'

            '</div>',

            unsafe_allow_html=True
        )


    with col3:

        st.markdown(

            '<div class="service-card">'

            '<div class="service-icon">'
            '📄'
            '</div>'

            '<div class="service-title">'
            'Citizen Documents'
            '</div>'

            '<div class="service-description">'
            'Government document information'
            '</div>'

            '</div>',

            unsafe_allow_html=True
        )


    # Suggested Questions

    st.markdown(
        "### 💡 Try asking"
    )


    question_col1, question_col2 = (
        st.columns(2)
    )


    with question_col1:

        if st.button(
            "💬 What is the eligibility for voter registration?",
            use_container_width=True
        ):

            st.session_state.pending_question = (
                "What is the eligibility for voter registration?"
            )

            st.rerun()


        if st.button(
            "💬 What documents are required?",
            use_container_width=True
        ):

            st.session_state.pending_question = (
                "What documents are required?"
            )

            st.rerun()


    with question_col2:

        if st.button(
            "💬 What is the application procedure?",
            use_container_width=True
        ):

            st.session_state.pending_question = (
                "What is the application procedure?"
            )

            st.rerun()


        if st.button(
            "💬 What are the fees?",
            use_container_width=True
        ):

            st.session_state.pending_question = (
                "What are the fees?"
            )

            st.rerun()


# =========================================================
# 15. DISPLAY CHAT HISTORY
# =========================================================

for message in (
    st.session_state.messages
):


    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


        if (
            message["role"] == "assistant"
            and "sources" in message
        ):

            sources = message["sources"]


            if sources:

                with st.expander(
                    "📚 View Retrieved Sources"
                ):

                    for source in sources:

                        st.write(
                            f"📄 {source['source']}"
                        )

                        st.caption(

                            f"Page {source['page']} "
                            f"• Relevance: "
                            f"{source['score']:.4f}"

                        )


# =========================================================
# 16. CHAT INPUT
# =========================================================

typed_question = st.chat_input(
    "Ask about government services..."
)


# =========================================================
# 17. GET QUESTION
# =========================================================

question = typed_question


if question is None:

    question = (
        st.session_state.pending_question
    )

    st.session_state.pending_question = None


# =========================================================
# 18. PROCESS QUESTION
# =========================================================

if question:


    # -----------------------------------------
    # USER MESSAGE
    # -----------------------------------------

    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )


    st.session_state.messages.append({

        "role": "user",

        "content": question

    })


    # -----------------------------------------
    # ASSISTANT
    # -----------------------------------------

    with st.chat_message(
        "assistant"
    ):


        # Visible searching message

        st.markdown(
            '<div class="search-status">'
            '🔎 Searching government documents...'
            '</div>',
            unsafe_allow_html=True
        )


        # Retrieval + Gemini

        with st.spinner(
            "🤖 Analyzing the relevant information..."
        ):

            answer, results = (
                government_rag(

                    question,

                    st.session_state.messages,

                    language

                )
            )


        # Answer

        st.markdown(
            answer
        )


        # Sources

        if results:

            with st.expander(
                "📚 View Retrieved Sources"
            ):

                for result in results:

                    st.write(
                        f"📄 {result['source']}"
                    )

                    st.caption(

                        f"Page {result['page']} "
                        f"• Relevance: "
                        f"{result['score']:.4f}"

                    )


    # -----------------------------------------
    # SAVE RESPONSE
    # -----------------------------------------

    st.session_state.messages.append({

        "role": "assistant",

        "content": answer,

        "sources": results

    })


# =========================================================
# 19. FOOTER
# =========================================================

st.markdown(

    '<div class="gov-footer">'

    '🏛️ Government Services Assistant'

    '&nbsp; • &nbsp;'

    'Document-Grounded AI'

    '&nbsp; • &nbsp;'

    'RAG + Gemini'

    '</div>',

    unsafe_allow_html=True
)