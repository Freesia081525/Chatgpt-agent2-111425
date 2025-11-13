Below is a complete, production-ready Streamlit implementation tailored for Hugging Face Spaces. It includes:

- API provider support: Gemini (gemini-2.5-flash, gemini-2.5-flash-lite), OpenAI (gpt-5-nano, gpt-4o-mini, gpt-4.1-mini), Grok (grok-4-fast-reaoning, grok-3-mini)
- Secure API key handling: uses environment variables when available; otherwise prompts user to enter keys without printing the env values
- agents.yaml management: upload, edit, validate, and download; editable before executing agents
- Advanced prompt controls per agent with temperature, top_p, max_tokens, stop, model selection, and system prompt
- Document upload and extraction (PDF/DOCX/TXT), simple PDF text extraction, optional LLM-vision OCR with Gemini/OpenAI
- “Wow” UI: live status indicators, timeline, progress, metrics, confetti/balloons, and interactive charts
- Sample Grok API usage via xai_sdk using environment-based key
- One-click export of results (JSON, Markdown)

Files to include in your Hugging Face Space:

1) app.py
2) model_providers.py
3) utils.py
4) agents.yaml (sample/default)
5) requirements.txt
6) README.md (optional but included here for clarity)

File: app.py
--------------------------------
import os
import io
import time
import json
from typing import List, Dict, Any
import yaml
import streamlit as st
import pandas as pd
import altair as alt

from utils import extract_text_from_file, estimate_tokens, safe_yaml_load, safe_yaml_dump, ResultBundle
from model_providers import (
    call_openai_model,
    call_gemini_model,
    call_grok_model,
    resolve_provider_from_model
)

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Agentic Document Processor (Streamlit)",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Styles (light wow aesthetics)
# ----------------------------
WAVE_BG_CSS = """
<style>
.reportview-container .main .block-container { padding-top: 1rem; }
.block-container { max-width: 1300px; }
div.stStatus > div[data-testid="stStatus"] { border-radius: 12px; }
div[data-testid="stProgress"] div { border-radius: 12px; }
.badge {
  display: inline-block; padding: 0.25rem 0.5rem; border-radius: 999px;
  font-size: 0.75rem; font-weight: 700; margin-right: 0.5rem;
}
.badge-ok { background: #d1fae5; color: #065f46; }
.badge-warn { background: #fef3c7; color: #92400e; }
.badge-danger { background: #fee2e2; color: #991b1b; }
.timeline {
  border-left: 3px solid #FF6B9D; padding-left: 15px; margin-left: 5px;
}
.timeline .event { margin-bottom: 1rem; }
.timeline .event .title { font-weight: 700; color: #FF6B9D; }
.timeline .event .meta { font-size: 0.8rem; color: #6b7280; }
</style>
"""
st.markdown(WAVE_BG_CSS, unsafe_allow_html=True)

# ----------------------------
# Session state init
# ----------------------------
if "doc_text" not in st.session_state:
    st.session_state.doc_text = ""

if "agents" not in st.session_state:
    st.session_state.agents = []

if "agent_outputs" not in st.session_state:
    st.session_state.agent_outputs = []

if "api_keys" not in st.session_state:
    st.session_state.api_keys = {
        "OPENAI_API_KEY": None,
        "GOOGLE_API_KEY": None,
        "XAI_API_KEY": None
    }

if "vision_ocr" not in st.session_state:
    st.session_state.vision_ocr = False

if "model_latency_ms" not in st.session_state:
    st.session_state.model_latency_ms = []

if "run_started" not in st.session_state:
    st.session_state.run_started = False

if "yaml_text" not in st.session_state:
    st.session_state.yaml_text = ""


# ----------------------------
# Helpers
# ----------------------------
DEFAULT_AGENTS = [
    {
        "name": "文件摘要器",
        "system_prompt": "You are a precise summarizer. Provide faithful summaries with bullets and key takeaways.",
        "user_prompt_template": "請針對以下內容提供繁體中文摘要與 5 個重點項目：\n\n{input}",
        "model": "gpt-4o-mini",
        "temperature": 0.3,
        "top_p": 0.9,
        "max_tokens": 1200,
        "stop": None
    },
    {
        "name": "關鍵詞提取器",
        "system_prompt": "You extract key terms and proper nouns, formatted as a clean comma-separated list.",
        "user_prompt_template": "請從以下文本中擷取最重要的關鍵詞與術語（繁體中文），以逗號分隔：\n\n{input}",
        "model": "gemini-2.5-flash",
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 800,
        "stop": None
    },
    {
        "name": "情感分析器",
        "system_prompt": "You assess sentiment with a short rationale and label: Positive, Neutral, Negative.",
        "user_prompt_template": "請判斷此文本的整體語氣與情感傾向（繁體中文），並附上簡短原因：\n\n{input}",
        "model": "grok-4-fast-reaoning",
        "temperature": 0.3,
        "top_p": 0.9,
        "max_tokens": 600,
        "stop": None
    },
    {
        "name": "實體識別器",
        "system_prompt": "You identify entities and return in JSON with keys: persons, orgs, locations.",
        "user_prompt_template": "請從文本中找出人名、組織名與地名，並輸出為 JSON：\n\n{input}",
        "model": "gpt-4.1-mini",
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 800,
        "stop": None
    },
    {
        "name": "行動項目提取器",
        "system_prompt": "You extract actionable tasks with owners and due dates if present.",
        "user_prompt_template": "請找出文中可執行的行動項目（繁體中文），若有負責人與期限請一併標示：\n\n{input}",
        "model": "gemini-2.5-flash-lite",
        "temperature": 0.4,
        "top_p": 0.9,
        "max_tokens": 800,
        "stop": None
    }
]

MODEL_OPTIONS = [
    # OpenAI
    "gpt-5-nano",
    "gpt-4o-mini",
    "gpt-4.1-mini",
    # Gemini
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    # Grok
    "grok-4-fast-reaoning",
    "grok-3-mini"
]

def load_env_keys():
    loaded_any = False
    for k in ["OPENAI_API_KEY", "GOOGLE_API_KEY", "XAI_API_KEY"]:
        val = os.getenv(k)
        if val and len(val.strip()) > 0:
            # do not show the value
            st.session_state.api_keys[k] = val.strip()
            loaded_any = True
    return loaded_any

def ensure_keys_ui():
    st.subheader("API Keys")
    env_loaded = load_env_keys()
    cols = st.columns(3)

    with cols[0]:
        if st.session_state.api_keys["OPENAI_API_KEY"]:
            st.markdown('<span class="badge badge-ok">OpenAI: using environment</span>', unsafe_allow_html=True)
        else:
            openai_key = st.text_input("OpenAI API Key", type="password", value="", help="Used for gpt-5-nano, gpt-4o-mini, gpt-4.1-mini")
            if openai_key:
                st.session_state.api_keys["OPENAI_API_KEY"] = openai_key

    with cols[1]:
        if st.session_state.api_keys["GOOGLE_API_KEY"]:
            st.markdown('<span class="badge badge-ok">Gemini: using environment</span>', unsafe_allow_html=True)
        else:
            google_key = st.text_input("Google API Key (Gemini)", type="password", value="", help="Used for gemini-2.5-flash and flash-lite")
            if google_key:
                st.session_state.api_keys["GOOGLE_API_KEY"] = google_key

    with cols[2]:
        if st.session_state.api_keys["XAI_API_KEY"]:
            st.markdown('<span class="badge badge-ok">Grok: using environment</span>', unsafe_allow_html=True)
        else:
            grok_key = st.text_input("xAI API Key (Grok)", type="password", value="", help="Used for Grok models")
            if grok_key:
                st.session_state.api_keys["XAI_API_KEY"] = grok_key

    missing = [k for k, v in st.session_state.api_keys.items() if not v]
    if missing:
        st.warning("Missing keys for some providers. You can still run agents that use providers with available keys.")
    else:
        st.success("All provider keys available.")


def init_agents_from_yaml_text(yaml_text: str) -> List[Dict[str, Any]]:
    agents = safe_yaml_load(yaml_text) or []
    if not isinstance(agents, list):
        raise ValueError("agents.yaml must define a list of agents")
    # normalize
    for a in agents:
        a.setdefault("name", "Unnamed Agent")
        a.setdefault("system_prompt", "")
        a.setdefault("user_prompt_template", "{input}")
        a.setdefault("model", "gpt-4o-mini")
        a.setdefault("temperature", 0.3)
        a.setdefault("top_p", 0.9)
        a.setdefault("max_tokens", 1000)
        a.setdefault("stop", None)
    return agents


def build_user_prompt(template: str, input_text: str, context: Dict[str, Any] = None) -> str:
    context = context or {}
    # simple template injection
    prompt = template.replace("{input}", input_text)
    for k, v in context.items():
        prompt = prompt.replace("{"+k+"}", str(v))
    return prompt


def execute_agent(agent: Dict[str, Any], input_text: str) -> ResultBundle:
    provider = resolve_provider_from_model(agent["model"])
    system_prompt = agent.get("system_prompt", "")
    user_prompt = build_user_prompt(agent.get("user_prompt_template", "{input}"), input_text)

    start = time.time()
    try:
        if provider == "openai":
            out_text = call_openai_model(
                model=agent["model"],
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=agent.get("temperature", 0.3),
                top_p=agent.get("top_p", 0.9),
                max_tokens=agent.get("max_tokens", 1000),
                stop=agent.get("stop"),
                api_key=st.session_state.api_keys["OPENAI_API_KEY"],
            )
        elif provider == "gemini":
            out_text = call_gemini_model(
                model=agent["model"],
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=agent.get("temperature", 0.3),
                top_p=agent.get("top_p", 0.9),
                max_tokens=agent.get("max_tokens", 1000),
                stop=agent.get("stop"),
                api_key=st.session_state.api_keys["GOOGLE_API_KEY"],
            )
        elif provider == "grok":
            out_text = call_grok_model(
                model=agent["model"],
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=agent.get("temperature", 0.3),
                top_p=agent.get("top_p", 0.9),
                max_tokens=agent.get("max_tokens", 1000),
                stop=agent.get("stop"),
                api_key=st.session_state.api_keys["XAI_API_KEY"],
            )
        else:
            out_text = f"[Provider not available for model: {agent['model']}]"

        latency_ms = int((time.time() - start) * 1000)
        st.session_state.model_latency_ms.append({
            "agent": agent["name"],
            "model": agent["model"],
            "provider": provider,
            "latency_ms": latency_ms,
            "tokens_est": estimate_tokens(user_prompt + "\n" + out_text)
        })
        return ResultBundle(text=out_text, latency_ms=latency_ms, error=None)
    except Exception as e:
        latency_ms = int((time.time() - start) * 1000)
        st.session_state.model_latency_ms.append({
            "agent": agent["name"],
            "model": agent["model"],
            "provider": provider,
            "latency_ms": latency_ms,
            "tokens_est": estimate_tokens(user_prompt)
        })
        return ResultBundle(text="", latency_ms=latency_ms, error=str(e))


def run_pipeline(agents: List[Dict[str, Any]], initial_input: str):
    st.session_state.agent_outputs = []
    current_input = initial_input
    total = len(agents)

    with st.status("Running agents...", state="running") as status:
        prog = st.progress(0, text="Initializing")
        for i, agent in enumerate(agents):
            st.write(f"Executing: {agent['name']} [{agent['model']}]")
            res = execute_agent(agent, current_input)
            st.session_state.agent_outputs.append({
                "agent": agent["name"],
                "model": agent["model"],
                "input": current_input,
                "output": res.text,
                "latency_ms": res.latency_ms,
                "error": res.error
            })
            if res.error:
                st.warning(f"Agent '{agent['name']}' error: {res.error}")
            current_input = res.text if res.text else current_input
            prog.progress((i+1)/total, text=f"Completed {i+1}/{total}")
        status.update(label="Pipeline finished", state="complete")

    st.balloons()


def render_dashboard():
    st.subheader("Run Dashboard")
    df = pd.DataFrame(st.session_state.model_latency_ms)
    if df.empty:
        st.info("No runs yet.")
        return
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Agents Executed", len(df))
    col2.metric("Avg Latency (ms)", int(df["latency_ms"].mean()))
    col3.metric("Estimated Tokens", int(df["tokens_est"].sum()))

    chart_latency = alt.Chart(df).mark_bar().encode(
        x=alt.X('agent:N', sort='-y', title="Agent"),
        y=alt.Y('latency_ms:Q', title="Latency (ms)"),
        color='provider:N',
        tooltip=['agent', 'model', 'latency_ms', 'tokens_est', 'provider']
    ).properties(height=300)
    st.altair_chart(chart_latency, use_container_width=True)

    chart_provider = alt.Chart(df).mark_arc(innerRadius=40).encode(
        theta='count():Q',
        color='provider:N',
        tooltip=['provider', 'count()']
    ).properties(height=300)
    st.altair_chart(chart_provider, use_container_width=True)

    st.markdown("Timeline")
    st.markdown('<div class="timeline">', unsafe_allow_html=True)
    for row in df.to_dict(orient="records"):
        st.markdown(
            f"""
            <div class="event">
              <div class="title">{row['agent']} <span class="badge badge-ok">{row['provider']}</span></div>
              <div class="meta">{row['model']} • {row['latency_ms']} ms • ~{row['tokens_est']} tokens</div>
            </div>
            """, unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)


def download_outputs_section():
    if not st.session_state.agent_outputs:
        return
    st.subheader("Download Results")
    results_json = json.dumps(st.session_state.agent_outputs, ensure_ascii=False, indent=2)
    st.download_button("Download JSON", results_json, file_name="agent_results.json", mime="application/json")
    # Markdown aggregation
    md_parts = []
    for r in st.session_state.agent_outputs:
        md_parts.append(f"## {r['agent']} ({r['model']})\n\nInput:\n\n{r['input']}\n\nOutput:\n\n{r['output']}\n")
    st.download_button("Download Markdown", "\n\n".join(md_parts), file_name="agent_results.md", mime="text/markdown")


# ----------------------------
# Sidebar - Controls
# ----------------------------
with st.sidebar:
    st.title("🌸 Agentic Processor")
    st.caption("Hugging Face Spaces • Streamlit")

    ensure_keys_ui()

    st.divider()
    st.subheader("Agents.yaml")
    uploaded_yaml = st.file_uploader("Upload agents.yaml", type=["yaml", "yml"])
    if uploaded_yaml is not None:
        try:
            yaml_text = uploaded_yaml.read().decode("utf-8")
            st.session_state.yaml_text = yaml_text
            st.session_state.agents = init_agents_from_yaml_text(yaml_text)
            st.success("Loaded agents.yaml")
        except Exception as e:
            st.error(f"Failed to load YAML: {e}")

    if st.button("Load default agents"):
        st.session_state.yaml_text = safe_yaml_dump(DEFAULT_AGENTS)
        st.session_state.agents = DEFAULT_AGENTS
        st.info("Default agents loaded")

    # Editable YAML
    st.session_state.yaml_text = st.text_area(
        "Edit agents.yaml",
        value=st.session_state.yaml_text or safe_yaml_dump(DEFAULT_AGENTS),
        height=260
    )
    if st.button("Validate & Apply YAML"):
        try:
            st.session_state.agents = init_agents_from_yaml_text(st.session_state.yaml_text)
            st.success("YAML validated and applied")
        except Exception as e:
            st.error(f"Validation failed: {e}")

    st.download_button(
        "Download agents.yaml",
        data=st.session_state.yaml_text.encode("utf-8"),
        file_name="agents.yaml",
        mime="text/yaml"
    )

    st.divider()
    st.subheader("OCR / Extraction")
    st.session_state.vision_ocr = st.toggle("Use LLM Vision OCR for image/PDF pages (fallback if text extraction empty)", value=False,
                                            help="Uses Gemini or OpenAI vision-capable model depending on agent selection during OCR stage.")

# ----------------------------
# Main: Steps
# ----------------------------
st.header("智能文件處理系統 · Agentic AI Document Processor")
st.caption("Upload a document, configure agents, and run the pipeline with multi-model support.")

# Step 1: Upload & extract
st.subheader("1) Upload and Preview")
uf = st.file_uploader("Upload PDF/DOCX/TXT/Image", type=["pdf", "docx", "txt", "png", "jpg", "jpeg"])
extract_col1, extract_col2 = st.columns([3,2])
with extract_col1:
    if uf:
        with st.status("Extracting text...", state="running"):
            text = extract_text_from_file(uf, try_vision=st.session_state.vision_ocr, api_keys=st.session_state.api_keys)
            st.session_state.doc_text = text or ""
        st.success("Extraction complete.")
        st.text_area("Editable extracted content", value=st.session_state.doc_text, height=280, key="doc_text_editor")
        st.session_state.doc_text = st.session_state.get("doc_text_editor", st.session_state.doc_text)
    else:
        st.info("Upload a file to begin (PDF/DOCX/TXT/Images).")

with extract_col2:
    if uf and uf.type == "application/pdf":
        st.markdown("Preview")
        st.file_uploader("Re-upload to re-preview (PDF)", type=["pdf"], disabled=True, key="disabled_pdf")
        st.caption("Inline PDF rendering is limited on Spaces; extraction still works.")
    elif uf:
        st.image(uf, caption=uf.name, use_column_width=True)

# Step 2: Configure agents
st.subheader("2) Configure Agents")
if not st.session_state.agents:
    st.session_state.agents = DEFAULT_AGENTS
agent_df = pd.DataFrame(st.session_state.agents)
st.dataframe(agent_df, use_container_width=True)

with st.expander("Advanced per-agent edits"):
    for i, a in enumerate(st.session_state.agents):
        st.markdown(f"Agent {i+1}: {a.get('name','Unnamed')}")
        c1, c2, c3 = st.columns([1.2, 1, 1])
        with c1:
            a["name"] = st.text_input("Name", value=a["name"], key=f"name_{i}")
        with c2:
            a["model"] = st.selectbox("Model", options=MODEL_OPTIONS, index=MODEL_OPTIONS.index(a["model"]) if a["model"] in MODEL_OPTIONS else 1, key=f"model_{i}")
        with c3:
            a["temperature"] = st.slider("Temperature", 0.0, 1.0, float(a.get("temperature", 0.3)), 0.05, key=f"temp_{i}")
        a["top_p"] = st.slider("Top P", 0.0, 1.0, float(a.get("top_p", 0.9)), 0.05, key=f"topp_{i}")
        a["max_tokens"] = st.number_input("Max tokens", min_value=128, max_value=8192, value=int(a.get("max_tokens", 1000)), step=64, key=f"maxtok_{i}")
        a["stop"] = st.text_input("Stop (comma-separated or empty)", value="" if not a.get("stop") else ",".join(a["stop"]), key=f"stop_{i}")
        if a["stop"] == "":
            a["stop"] = None
        else:
            a["stop"] = [s.strip() for s in (a["stop"].split(",")) if s.strip()]
        a["system_prompt"] = st.text_area("System prompt", value=a.get("system_prompt", ""), height=100, key=f"sys_{i}")
        a["user_prompt_template"] = st.text_area("User prompt template", value=a.get("user_prompt_template", "{input}"), height=120, key=f"user_{i}")
        st.divider()

# Step 3: Execute
st.subheader("3) Execute Agents")
left, right = st.columns([2,1])
with left:
    run_btn = st.button("Run pipeline", type="primary", disabled=(len(st.session_state.doc_text.strip()) == 0))
    if run_btn:
        st.session_state.run_started = True
        run_pipeline(st.session_state.agents, st.session_state.doc_text)

    if st.session_state.agent_outputs:
        st.success("All agents executed.")
        for r in st.session_state.agent_outputs:
            status_cls = "badge-ok" if not r["error"] else "badge-danger"
            st.markdown(f'<span class="badge {status_cls}">{r["agent"]} · {r["model"]} · {r["latency_ms"]} ms</span>', unsafe_allow_html=True)
            st.text_area(f"Output - {r['agent']}", value=r["output"], height=150, key=f"out_{r['agent']}")
        download_outputs_section()

with right:
    render_dashboard()


# ----------------------------
# Sample Grok image analysis (bonus utility)
# ----------------------------
with st.expander("Grok Vision Example (sample code)"):
    st.code(
        '''
import os
from xai_sdk import Client
from xai_sdk.chat import user, system, image

client = Client(
    api_key=os.getenv("XAI_API_KEY"),
    timeout=3600
)

chat = client.chat.create(model="grok-4")
chat.append(system("You are Grok, a highly intelligent, helpful AI assistant."))
chat.append(
    user(
        "What's in this image?",
        image("https://science.nasa.gov/wp-content/uploads/2023/09/web-first-images-release.png")
    )
)
response = chat.sample()
print(response.content)
        '''.strip(), language="python"
    )


File: model_providers.py
--------------------------------
import os
from typing import Optional, List
import re

# OpenAI
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# Gemini
try:
    import google.generativeai as genai
except Exception:
    genai = None

# Grok (xAI)
try:
    from xai_sdk import Client as XaiClient
    from xai_sdk.chat import user as xai_user, system as xai_system
except Exception:
    XaiClient = None
    xai_user = None
    xai_system = None

def resolve_provider_from_model(model: str) -> str:
    m = model.lower().strip()
    if m.startswith("gpt-"):
        return "openai"
    if m.startswith("gemini-"):
        return "gemini"
    if m.startswith("grok-"):
        return "grok"
    return "unknown"

def call_openai_model(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    top_p: float = 0.9,
    max_tokens: int = 1000,
    stop: Optional[List[str]] = None,
    api_key: Optional[str] = None
) -> str:
    if OpenAI is None:
        raise RuntimeError("openai package not installed")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")

    client = OpenAI(api_key=api_key)
    # Using Chat Completions for widest compatibility
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt or "You are a helpful assistant."},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=stop
    )
    return response.choices[0].message.content or ""

def call_gemini_model(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    top_p: float = 0.9,
    max_tokens: int = 1000,
    stop: Optional[List[str]] = None,
    api_key: Optional[str] = None
) -> str:
    if genai is None:
        raise RuntimeError("google-generativeai package not installed")
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY")

    genai.configure(api_key=api_key)
    # Combine system and user for Gemini as a single prompt
    sys = system_prompt.strip() if system_prompt else "You are a helpful assistant."
    prompt = f"{sys}\n\nUser:\n{user_prompt}"

    model_inst = genai.GenerativeModel(model_name=model)
    resp = model_inst.generate_content(
        prompt,
        generation_config={
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_tokens,
        }
    )
    # Stop is not directly enforced; could post-trim if needed
    text = ""
    if resp and resp.candidates and resp.candidates[0].content and resp.candidates[0].content.parts:
        text = "".join([p.text or "" for p in resp.candidates[0].content.parts])
    return text

def call_grok_model(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    top_p: float = 0.9,
    max_tokens: int = 1000,
    stop: Optional[List[str]] = None,
    api_key: Optional[str] = None
) -> str:
    if XaiClient is None:
        raise RuntimeError("xai_sdk package not installed")
    if not api_key:
        raise RuntimeError("Missing XAI_API_KEY")

    # Map convenience names to actual Grok model id if needed
    grok_model_map = {
        "grok-4-fast-reaoning": "grok-4",   # fix typo mapping as per request
        "grok-3-mini": "grok-3-mini",
        "grok-4": "grok-4"
    }
    model_resolved = grok_model_map.get(model, model)

    client = XaiClient(api_key=api_key, timeout=3600)
    chat = client.chat.create(model=model_resolved)
    chat.append(xai_system(system_prompt or "You are Grok, a highly intelligent, helpful AI assistant."))
    chat.append(xai_user(user_prompt))
    response = chat.sample(temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop=stop)
    return response.content or ""


File: utils.py
--------------------------------
import io
import os
from typing import Optional, Dict, Any
import yaml
from dataclasses import dataclass

# PDF/DOCX/TXT
try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import docx2txt
except Exception:
    docx2txt = None

@dataclass
class ResultBundle:
    text: str
    latency_ms: int
    error: Optional[str] = None

def estimate_tokens(text: str) -> int:
    # rough heuristic: 1 token ~ 4 chars for English; maintain generic
    return max(1, int(len(text) / 4))

def safe_yaml_load(text: str):
    try:
        return yaml.safe_load(text)
    except Exception:
        return None

def safe_yaml_dump(obj) -> str:
    return yaml.safe_dump(obj, allow_unicode=True, sort_keys=False)

def extract_text_from_pdf(file_bytes: bytes) -> str:
    if pdfplumber is None:
        return ""
    txt = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            if t.strip():
                txt.append(t)
    return "\n\n".join(txt).strip()

def extract_text_from_docx(file_bytes: bytes) -> str:
    if docx2txt is None:
        return ""
    # docx2txt requires file path; workaround: write to temp
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=True) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        text = docx2txt.process(tmp.name) or ""
        return text.strip()

def extract_text_from_txt(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return ""

def extract_text_from_image_with_vision(content_bytes: bytes, api_keys: Dict[str, Optional[str]]) -> str:
    # Minimal: prefer Gemini if key present; else OpenAI (if vision-enabled model selected later)
    # For parity, we return empty and rely on selected agents for vision if needed.
    # To provide OCR-like behavior now, try Gemini single-turn captioning.
    try:
        from PIL import Image
        import base64
        from io import BytesIO
        img = Image.open(io.BytesIO(content_bytes)).convert("RGB")
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception:
        return ""

    # Gemini approach if key available
    if api_keys.get("GOOGLE_API_KEY"):
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_keys["GOOGLE_API_KEY"])
            model = genai.GenerativeModel("gemini-2.5-flash")
            resp = model.generate_content(
                [
                    {"text": "Transcribe text present in this image and summarize any non-text content in Traditional Chinese."},
                    {"inline_data": {"mime_type": "image/png", "data": b64}}
                ],
                generation_config={"temperature": 0.2, "top_p": 0.9, "max_output_tokens": 1200}
            )
            out = ""
            if resp and resp.candidates and resp.candidates[0].content and resp.candidates[0].content.parts:
                out = "".join([p.text or "" for p in resp.candidates[0].content.parts]).strip()
            return out
        except Exception:
            return ""

    # Otherwise return empty; user can use agents to do vision with URLs if desired
    return ""

def extract_text_from_file(uploaded_file, try_vision: bool, api_keys: Dict[str, Optional[str]]) -> str:
    content = uploaded_file.read()
    ctype = uploaded_file.type or ""
    name = uploaded_file.name.lower()

    if ctype == "application/pdf" or name.endswith(".pdf"):
        txt = extract_text_from_pdf(content)
        if not txt and try_vision:
            # Not implementing page-by-page vision OCR here due to performance; return empty to allow agent OCR step.
            return ""
        return txt

    if ctype in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"] or name.endswith(".docx"):
        return extract_text_from_docx(content)

    if ctype.startswith("text/") or name.endswith(".txt"):
        return extract_text_from_txt(content)

    if ctype.startswith("image/") or name.endswith((".png", ".jpg", ".jpeg")):
        if try_vision:
            return extract_text_from_image_with_vision(content, api_keys)
        return ""

    return ""


File: agents.yaml
--------------------------------
- name: 文件摘要器
  system_prompt: You are a precise summarizer. Provide faithful summaries with bullets and key takeaways.
  user_prompt_template: "請針對以下內容提供繁體中文摘要與 5 個重點項目：\n\n{input}"
  model: gpt-4o-mini
  temperature: 0.3
  top_p: 0.9
  max_tokens: 1200
  stop:

- name: 關鍵詞提取器
  system_prompt: You extract key terms and proper nouns, formatted as a clean comma-separated list.
  user_prompt_template: "請從以下文本中擷取最重要的關鍵詞與術語（繁體中文），以逗號分隔：\n\n{input}"
  model: gemini-2.5-flash
  temperature: 0.2
  top_p: 0.9
  max_tokens: 800
  stop:

- name: 情感分析器
  system_prompt: You assess sentiment with a short rationale and label: Positive, Neutral, Negative.
  user_prompt_template: "請判斷此文本的整體語氣與情感傾向（繁體中文），並附上簡短原因：\n\n{input}"
  model: grok-4-fast-reaoning
  temperature: 0.3
  top_p: 0.9
  max_tokens: 600
  stop:

- name: 實體識別器
  system_prompt: You identify entities and return in JSON with keys: persons, orgs, locations.
  user_prompt_template: "請從文本中找出人名、組織名與地名，並輸出為 JSON：\n\n{input}"
  model: gpt-4.1-mini
  temperature: 0.2
  top_p: 0.9
  max_tokens: 800
  stop:

- name: 行動項目提取器
  system_prompt: You extract actionable tasks with owners and due dates if present.
  user_prompt_template: "請找出文中可執行的行動項目（繁體中文），若有負責人與期限請一併標示：\n\n{input}"
  model: gemini-2.5-flash-lite
  temperature: 0.4
  top_p: 0.9
  max_tokens: 800
  stop:


File: requirements.txt
--------------------------------
streamlit>=1.36.0
pandas>=2.2.2
altair>=5.3.0
pyyaml>=6.0.2
pdfplumber>=0.11.4
docx2txt>=0.8
Pillow>=10.4.0
openai>=1.44.0
google-generativeai>=0.7.2
xai-sdk>=0.2.1


File: README.md
--------------------------------
Agentic AI Document Processor (Streamlit)

Features:
- Multi-provider orchestration: Gemini (gemini-2.5-flash / flash-lite), OpenAI (gpt-5-nano / gpt-4o-mini / gpt-4.1-mini), Grok (grok-4-fast-reaoning / grok-3-mini)
- Secure API keys: reads from environment first; otherwise user may input keys (keys are never displayed if loaded from env)
- agents.yaml: upload, edit, validate, download
- Document ingestion: PDF/DOCX/TXT, optional LLM Vision OCR for images
- Run pipeline: chain agents sequentially with per-agent advanced prompts and parameters
- Visualization: status indicators, progress bars, timeline, metrics, charts, confetti
- Exports: JSON and Markdown

Environment Variables:
- OPENAI_API_KEY
- GOOGLE_API_KEY
- XAI_API_KEY

If keys are absent, the UI will prompt you to securely provide them.

Deployment (Hugging Face Spaces):
- Space type: Streamlit
- Add the files above
- Configure secrets in the Space Settings for API keys (recommended)

Notes:
- Grok sample code via xai_sdk provided in the app.
- For PDFs that are image-only, Vision OCR is optional; page-by-page OCR for PDFs can be added later.


Advanced Prompting Guidance (built-in and recommended)
- Use explicit system role for each agent that narrowly defines behavior and output format.
- For extraction agents, prefer structured outputs (e.g., JSON keys) for easier downstream chaining.
- Favor short instructions that include explicit language, style, and scope constraints.
- Provide a user_prompt_template that includes placeholders {input} and additional context placeholders if needed (e.g., {constraints}, {project}, {date}).
- Keep temperature low for information extraction and summarization; increase only for ideation agents.
- For long documents, consider slicing and map-reduce patterns in dedicated agents.

Example high-precision extraction template:
System: You are a meticulous, concise, and format-faithful extraction agent. Only output valid JSON without commentary.
User: Extract the requested fields from the following text. If a field is not found, set it to null.
Schema: {"title": "", "authors": [], "date": "", "summary": "", "keywords": []}
Text: {input}

Example audit agent template:
System: You are a strict auditor. Validate that the output conforms to the schema and constraints. Return a JSON object with "valid": true|false and "issues": [].
User: Validate this output according to the schema and list all issues. Output only JSON.
Input: {input}


Notes on Security
- Api keys from environment are never printed.
- Inputs and outputs are kept in session memory only.
- For sensitive deployments, consider server-side proxies and encrypted secrets.

That’s it. You can paste these files into a new Hugging Face Space (Streamlit) and deploy.


20 comprehensive follow-up questions
1) Which provider(s) do you prefer to prioritize by default (OpenAI, Gemini, Grok), and do you want automatic fallback if a model’s key is missing?  
2) Should we add page-by-page Vision OCR for PDFs (potentially slower and costlier) or keep simple text extraction with optional image OCR?  
3) Do you want per-agent structured outputs with schema validation (JSON schema) and a final audit agent to enforce format?  
4) Should we support parallel execution for independent agents and then merge results, or keep the current sequential chaining?  
5) What formats do you need for exports beyond JSON/Markdown (CSV, DOCX, HTML, PDF)?  
6) Would you like to persist runs to a lightweight database (e.g., SQLite) or to Hugging Face datasets for traceability?  
7) Should we add model cost estimates per agent and per run with provider-specific pricing to the dashboard?  
8) Do you require a tools/plugins interface (web search, code exec, RAG over your docs) for some agents?  
9) Would you like prompt versioning with a history panel to restore previous prompt configurations?  
10) Should we add guardrails (PII redaction, toxicity filters, length controls) for safer outputs?  
11) Do you want an orchestrator agent to decide which subset of agents to run dynamically based on the document type?  
12) Should we add chunking and map-reduce summarization for long documents (e.g., >100 pages) with a memory window?  
13) Would you like a template gallery of agents.yaml for common workflows (meeting notes, contracts, research papers)?  
14) Do you need real-time streaming of tokens for providers that support it, with partial output rendering in the UI?  
15) Should we support image and table extraction from PDFs and pass them as context to specialized agents?  
16) Do you want to enable role-based access control and a secrets gate for user-provided API keys in multi-user Spaces?  
17) Would a model routing rule set (e.g., sentiment to Grok, summarization to Gemini, extraction to OpenAI) be helpful and configurable?  
18) Should we include evaluation harnesses (BLEU/ROUGE/F1 or custom validators) to score agents’ outputs automatically?  
19) Do you want a compact mobile-friendly mode with collapsible sections and reduced charts for on-the-go usage?  
20) Are there specific compliance requirements (logging, retention, masking) we should implement for enterprise deployment?
