import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CodeBot — Coding Assistant",
    page_icon="💻",
    layout="centered"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400&display=swap');
:root {
    --blue: #3A7CA5;
    --blue-light: #6EC1E4;
    --blue-dark: #1B3B5F;
    --panel: #101B2A;
    --panel-light: #1A2538;
    --border: rgba(58,124,165,0.25);
    --text: #E3EAF2;
    --text-muted: #6B7280;
    --glow: rgba(58,124,165,0.15);
}
html, body, [data-testid="stAppViewContainer"] {
    background: var(--panel) !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { display: none; }
.block-container {
    max-width: 860px !important;
    padding: 0 2rem 6rem !important;
}
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(58,124,165,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(58,124,165,0.03) 1px, transparent 1px);
    background-size: 48px 48px;
    pointer-events: none;
    z-index: 0;
}
.code-header {
    text-align: center;
    padding: 3.5rem 0 2rem;
    position: relative;
}
.code-emblem {
    font-size: 3.2rem;
    display: block;
    margin-bottom: 0.4rem;
    filter: drop-shadow(0 0 18px rgba(58,124,165,0.6));
    animation: pulse-glow 3s ease-in-out infinite;
}
@keyframes pulse-glow {
    0%, 100% { filter: drop-shadow(0 0 12px rgba(58,124,165,0.5)); }
    50%       { filter: drop-shadow(0 0 28px rgba(58,124,165,0.9)); }
}
.code-title {
    font-size: 2.8rem !important;
    font-weight: 900 !important;
    letter-spacing: 0.06em;
    background: linear-gradient(135deg, var(--blue-light) 0%, var(--blue) 50%, var(--blue-dark) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 !important;
    line-height: 1.1 !important;
}
.code-subtitle {
    font-size: 0.82rem;
    letter-spacing: 0.25em;
    color: var(--blue-dark);
    text-transform: uppercase;
    margin-top: 0.5rem;
}
.code-divider {
    width: 280px;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--blue), transparent);
    margin: 1.5rem auto;
}
.code-tagline {
    font-size: 1rem;
    font-style: italic;
    color: var(--text-muted);
    letter-spacing: 0.04em;
}
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.25rem 0 !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) .stMarkdown {
    background: linear-gradient(135deg, #1C2A2F 0%, #16243A 100%) !important;
    border: 1px solid rgba(58,124,165,0.2) !important;
    border-left: 3px solid #6EC1E4 !important;
    border-radius: 2px 8px 8px 2px !important;
    padding: 1rem 1.2rem !important;
    font-size: 1.05rem !important;
    color: #B8D8F2 !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) .stMarkdown {
    background: linear-gradient(135deg, var(--panel-light) 0%, #15243A 100%) !important;
    border: 1px solid var(--border) !important;
    border-left: 3px solid var(--blue) !important;
    border-radius: 2px 8px 8px 2px !important;
    padding: 1.2rem 1.4rem !important;
    font-size: 1.08rem !important;
    line-height: 1.75 !important;
    color: var(--text) !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4), inset 0 0 40px rgba(58,124,165,0.03) !important;
}
[data-testid="chatAvatarIcon-user"] {
    background: #1C2A2F !important;
    border: 1px solid rgba(58,124,165,0.3) !important;
}
[data-testid="chatAvatarIcon-assistant"] {
    background: var(--panel-light) !important;
    border: 1px solid var(--border) !important;
}
[data-testid="stChatInputContainer"] {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    padding: 0.2rem 0.5rem !important;
    box-shadow: 0 0 30px rgba(58,124,165,0.08) !important;
}
[data-testid="stChatInputContainer"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--blue), transparent);
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.05rem !important;
    border: none !important;
    caret-color: var(--blue) !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: var(--text-muted) !important;
    font-style: italic;
}
[data-testid="stChatInputSubmitButton"] {
    color: var(--blue) !important;
}
[data-testid="stSpinner"] {
    color: var(--blue) !important;
}
[data-testid="stSpinner"] > div {
    border-top-color: var(--blue) !important;
}
.code-footer {
    text-align: center;
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(58,124,165,0.1);
}
.code-footer-text {
    font-size: 0.65rem;
    letter-spacing: 0.22em;
    color: var(--blue-dark);
    text-transform: uppercase;
}
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--panel); }
::-webkit-scrollbar-thumb { background: var(--blue-dark); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="code-header">
    <span class="code-emblem">💻</span>
    <h1 class="code-title">CodeBot</h1>
    <p class="code-subtitle">Coding Assistant &nbsp;·&nbsp; v1.0.0</p>
    <div class="code-divider"></div>
    <p class="code-tagline">\"Your AI pair programmer for code generation and debugging.\"</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LLM & CHAIN
# ─────────────────────────────────────────────
@st.cache_resource
def get_chain():
    model = OllamaLLM(model="hf.co/ehrrh/codegemma-2b-Q8_0-GGUF:Q8_0",
                      model_path="models/codegemma-2b-q8_0.gguf")

    template = """
    You are CodeBot, an expert coding assistant.
    You help users with code generation, explanation, and debugging.
    - If the user asks for code, generate clear, well-commented code.
    - If the user asks for debugging, analyze the code and provide fixes and explanations.
    - If the user asks a question, answer concisely and with code examples if relevant.
    - Always format code in markdown blocks.
    - If you need more information, ask clarifying questions.

    User message:
    {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    return prompt | model

chain = get_chain()

# ─────────────────────────────────────────────
# CHAT HISTORY
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ─────────────────────────────────────────────
# USER INPUT
# ─────────────────────────────────────────────
if question := st.chat_input("Ask CodeBot to generate or debug code..."):
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chain.invoke({"question": question})
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="code-footer">
    <p class="code-footer-text">CodeBot &nbsp;·&nbsp; Powered by Ollama + LangChain &nbsp;·&nbsp; All conversations are private</p>
</div>
""", unsafe_allow_html=True)
