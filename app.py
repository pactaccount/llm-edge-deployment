"""
app.py — Streamlit UI for Quantized Mistral-7B
================================================
Single-command demo of 4-bit NF4 quantized inference.
Shows real-time hardware stats (VRAM, latency, tokens/sec)
alongside model output.

Run with:
    streamlit run app.py
"""

import streamlit as st
from inference import load_quantized_model, build_prompt, generate_response, get_hardware_info

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Quantized LLM Edge Deployment",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #1a1a3e, #24243e);
    }
    /* Header style */
    .main-title {
        text-align: center;
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a78bfa, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 8px 0;
    }
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
    /* Hardware badge */
    .hw-badge {
        background: linear-gradient(135deg, #1e1b4b, #312e81);
        border-radius: 10px;
        padding: 12px 18px;
        color: #c7d2fe;
        font-family: monospace;
        font-size: 0.82rem;
        margin-bottom: 1rem;
        border: 1px solid #4338ca;
    }
    /* Metric cards */
    [data-testid="stMetric"] {
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 8px;
        padding: 8px 12px;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Model (cached — only runs once per session) ──────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    model, tokenizer = load_quantized_model()
    return model, tokenizer

# ── Sidebar Controls ──────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Controls")

    system_prompt = st.text_area(
        "System Prompt",
        value="You are a concise, technically precise AI assistant. Answer clearly and directly.",
        height=100,
    )

    max_tokens = st.slider("Max New Tokens", min_value=64, max_value=1024, value=256, step=64)
    temperature = st.slider("Temperature", min_value=0.1, max_value=1.2, value=0.7, step=0.05)

    st.divider()
    st.markdown("## 💡 Example Prompts")
    examples = [
        "Explain 4-bit NF4 quantization and why it outperforms INT4.",
        "What are the trade-offs between LoRA and full fine-tuning for 7B models?",
        "Compare GPTQ and bitsandbytes quantization approaches.",
        "How does double quantization save 0.4 bits per parameter?",
        "What is perplexity and how does quantization affect it?",
    ]
    for ex in examples:
        if st.button(ex[:55] + "...", use_container_width=True, key=ex):
            st.session_state["prefill"] = ex

    st.divider()
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pop("last_stats", None)
        st.rerun()

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown('<div class="main-title">🧠 Quantized LLM Edge Deployment</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Mistral-7B-Instruct running in <strong>4-bit NF4 quantization</strong> '
    'via bitsandbytes — 3× VRAM reduction (~14 GB → ~4.5 GB) with &lt;5% accuracy drop on MMLU.</div>',
    unsafe_allow_html=True
)

# ── Load Model with Spinner ───────────────────────────────────────────────────

with st.spinner("Loading 4-bit quantized Mistral-7B... (~20 seconds on first run)"):
    model, tokenizer = load_model()

hw = get_hardware_info()
st.markdown(
    f'<div class="hw-badge">'
    f'🖥️ <strong>Device:</strong> {hw["device"]} &nbsp;|&nbsp; '
    f'💾 <strong>VRAM:</strong> {hw["vram_used_gb"]} / {hw["vram_total_gb"]} GB &nbsp;|&nbsp; '
    f'🔢 <strong>Precision:</strong> 4-bit NF4 + Double Quantization &nbsp;|&nbsp; '
    f'⚡ <strong>Compute dtype:</strong> bfloat16'
    f'</div>',
    unsafe_allow_html=True
)

# ── Inference Metrics Row ─────────────────────────────────────────────────────

if "last_stats" in st.session_state:
    s = st.session_state.last_stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("⏱️ Latency", f"{s['latency_ms']} ms")
    col2.metric("🚀 Throughput", f"{s['tokens_per_second']} tok/s")
    col3.metric("📝 Tokens Generated", s["tokens_generated"])
    col4.metric("💾 VRAM Used", f"{s['vram_used_gb']} GB")
    st.divider()

# ── Chat History ──────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat Input ────────────────────────────────────────────────────────────────

# Handle example button prefill
default_input = st.session_state.pop("prefill", "")

user_input = st.chat_input("Ask the quantized model anything...", )

# Use prefilled text if a button was clicked, otherwise use chat input
query = user_input or default_input

if query:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Run inference
    with st.chat_message("assistant"):
        with st.spinner("Generating..."):
            prompt = build_prompt(query, system_prompt if system_prompt.strip() else None)
            result = generate_response(
                model,
                tokenizer,
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                stream=False,
            )
        st.markdown(result["response"])

    # Save assistant message and stats
    st.session_state.messages.append({"role": "assistant", "content": result["response"]})
    st.session_state.last_stats = {
        "latency_ms": result["latency_ms"],
        "tokens_per_second": result["tokens_per_second"],
        "tokens_generated": result["tokens_generated"],
        "vram_used_gb": result["vram_used_gb"],
    }

    st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown(
    "<div style='text-align:center; color:#475569; font-size:0.78rem; padding-top:2rem;'>"
    "Abhinav Vummidichetty — Large Model Edge Deployment Architecture | "
    "Mistral-7B-Instruct-v0.3 | 4-bit NF4 via bitsandbytes"
    "</div>",
    unsafe_allow_html=True
)
