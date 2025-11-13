import os
import io
import time
import json
from typing import List, Dict, Any, Optional
import yaml
import streamlit as st
import pandas as pd
import altair as alt
import re
from dataclasses import dataclass
import tempfile

# -----------------------------------------------------------------------------
# File: model_providers.py (Integrated)
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# File: utils.py (Integrated)
# -----------------------------------------------------------------------------
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

    if api_keys.get("GOOGLE_API_KEY"):
        try:
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

    return ""

def extract_text_from_file(uploaded_file, try_vision: bool, api_keys: Dict[str, Optional[str]]) -> str:
    content = uploaded_file.read()
    ctype = uploaded_file.type or ""
    name = uploaded_file.name.lower()

    if ctype == "application/pdf" or name.endswith(".pdf"):
        txt = extract_text_from_pdf(content)
        if not txt and try_vision:
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

# -----------------------------------------------------------------------------
# File: app.py (Main Logic)
# -----------------------------------------------------------------------------

# Page config
st.set_page_config(
    page_title="Agentic Document Processor (Streamlit)",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles
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

# Session state init
if "doc_text" not in st.session_state:
    st.session_state.doc_text = ""
if "agents" not in st.session_state:
    st.session_state.agents = []
if "agent_outputs" not in st.session_state:
    st.session_state.agent_outputs = []
if "api_keys" not in st.session_state:
    st.session_state.api_keys = {"OPENAI_API_KEY": None, "GOOGLE_API_KEY": None, "XAI_API_KEY": None}
if "vision_ocr" not in st.session_state:
    st.session_state.vision_ocr = False
if "model_latency_ms" not in st.session_state:
    st.session_state.model_latency_ms = []
if "run_started" not in st.session_state:
    st.session_state.run_started = False
if "yaml_text" not in st.session_state:
    st.session_state.yaml_text = ""

# Helpers
DEFAULT_AGENTS = [
    {"name": "文件摘要器", "system_prompt": "You are a precise summarizer. Provide faithful summaries with bullets and key takeaways.", "user_prompt_template": "請針對以下內容提供繁體中文摘要與 5 個重點項目：\n\n{input}", "model": "gpt-4o-mini", "temperature": 0.3, "top_p": 0.9, "max_tokens": 1200, "stop": None},
    {"name": "關鍵詞提取器", "system_prompt": "You extract key terms and proper nouns, formatted as a clean comma-separated list.", "user_prompt_template": "請從以下文本中擷取最重要的關鍵詞與術語（繁體中文），以逗號分隔：\n\n{input}", "model": "gemini-2.5-flash", "temperature": 0.2, "top_p": 0.9, "max_tokens": 800, "stop": None},
    {"name": "情感分析器", "system_prompt": "You assess sentiment with a short rationale and label: Positive, Neutral, Negative.", "user_prompt_template": "請判斷此文本的整體語氣與情感傾向（繁體中文），並附上簡短原因：\n\n{input}", "model": "grok-4-fast-reaoning", "temperature": 0.3, "top_p": 0.9, "max_tokens": 600, "stop": None},
    {"name": "實體識別器", "system_prompt": "You identify entities and return in JSON with keys: persons, orgs, locations.", "user_prompt_template": "請從文本中找出人名、組織名與地名，並輸出為 JSON：\n\n{input}", "model": "gpt-4.1-mini", "temperature": 0.2, "top_p": 0.9, "max_tokens": 800, "stop": None},
    {"name": "行動項目提取器", "system_prompt": "You extract actionable tasks with owners and due dates if present.", "user_prompt_template": "請找出文中可執行的行動項目（繁體中文），若有負責人與期限請一併標示：\n\n{input}", "model": "gemini-2.5-flash-lite", "temperature": 0.4, "top_p": 0.9, "max_tokens": 800, "stop": None}
]
MODEL_OPTIONS = ["gpt-5-nano", "gpt-4o-mini", "gpt-4.1-mini", "gemini-2.5-flash", "gemini-2.5-flash-lite", "grok-4-fast-reaoning", "grok-3-mini"]

def load_env_keys():
    for k in ["OPENAI_API_KEY", "GOOGLE_API_KEY", "XAI_API_KEY"]:
        val = os.getenv(k)
        if val and len(val.strip()) > 0:
            st.session_state.api_keys[k] = val.strip()

def ensure_keys_ui():
    st.subheader("API Keys")
    load_env_keys()
    cols = st.columns(3)
    with cols[0]:
        if st.session_state.api_keys["OPENAI_API_KEY"]:
            st.markdown('<span class="badge badge-ok">OpenAI: using environment</span>', unsafe_allow_html=True)
        else:
            openai_key = st.text_input("OpenAI API Key", type="password", value="", help="Used for gpt-5-nano, gpt-4o-mini, gpt-4.1-mini")
            if openai_key: st.session_state.api_keys["OPENAI_API_KEY"] = openai_key
    with cols[1]:
        if st.session_state.api_keys["GOOGLE_API_KEY"]:
            st.markdown('<span class="badge badge-ok">Gemini: using environment</span>', unsafe_allow_html=True)
        else:
            google_key = st.text_input("Google API Key (Gemini)", type="password", value="", help="Used for gemini-2.5-flash and flash-lite")
            if google_key: st.session_state.api_keys["GOOGLE_API_KEY"] = google_key
    with cols[2]:
        if st.session_state.api_keys["XAI_API_KEY"]:
            st.markdown('<span class="badge badge-ok">Grok: using environment</span>', unsafe_allow_html=True)
        else:
            grok_key = st.text_input("xAI API Key (Grok)", type="password", value="", help="Used for Grok models")
            if grok_key: st.session_state.api_keys["XAI_API_KEY"] = grok_key
    if any(not v for v in st.session_state.api_keys.values()):
        st.warning("Missing keys for some providers. Agents using those providers will fail.")
    else:
        st.success("All provider keys available.")

def init_agents_from_yaml_text(yaml_text: str) -> List[Dict[str, Any]]:
    agents = safe_yaml_load(yaml_text) or []
    if not isinstance(agents, list): raise ValueError("agents.yaml must define a list of agents")
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
    prompt = template.replace("{input}", input_text)
    for k, v in context.items(): prompt = prompt.replace("{" + k + "}", str(v))
    return prompt

def execute_agent(agent: Dict[str, Any], input_text: str) -> ResultBundle:
    provider = resolve_provider_from_model(agent["model"])
    system_prompt = agent.get("system_prompt", "")
    user_prompt = build_user_prompt(agent.get("user_prompt_template", "{input}"), input_text)
    start = time.time()
    try:
        out_text = ""
        if provider == "openai":
            out_text = call_openai_model(model=agent["model"], system_prompt=system_prompt, user_prompt=user_prompt, temperature=agent.get("temperature", 0.3), top_p=agent.get("top_p", 0.9), max_tokens=agent.get("max_tokens", 1000), stop=agent.get("stop"), api_key=st.session_state.api_keys["OPENAI_API_KEY"])
        elif provider == "gemini":
            out_text = call_gemini_model(model=agent["model"], system_prompt=system_prompt, user_prompt=user_prompt, temperature=agent.get("temperature", 0.3), top_p=agent.get("top_p", 0.9), max_tokens=agent.get("max_tokens", 1000), stop=agent.get("stop"), api_key=st.session_state.api_keys["GOOGLE_API_KEY"])
        elif provider == "grok":
            out_text = call_grok_model(model=agent["model"], system_prompt=system_prompt, user_prompt=user_prompt, temperature=agent.get("temperature", 0.3), top_p=agent.get("top_p", 0.9), max_tokens=agent.get("max_tokens", 1000), stop=agent.get("stop"), api_key=st.session_state.api_keys["XAI_API_KEY"])
        else:
            out_text = f"[Provider not available for model: {agent['model']}]"
        latency_ms = int((time.time() - start) * 1000)
        st.session_state.model_latency_ms.append({"agent": agent["name"], "model": agent["model"], "provider": provider, "latency_ms": latency_ms, "tokens_est": estimate_tokens(user_prompt + "\n" + out_text)})
        return ResultBundle(text=out_text, latency_ms=latency_ms, error=None)
    except Exception as e:
        latency_ms = int((time.time() - start) * 1000)
        st.session_state.model_latency_ms.append({"agent": agent["name"], "model": agent["model"], "provider": provider, "latency_ms": latency_ms, "tokens_est": estimate_tokens(user_prompt)})
        return ResultBundle(text="", latency_ms=latency_ms, error=str(e))

def run_pipeline(agents: List[Dict[str, Any]], initial_input: str):
    st.session_state.agent_outputs = []
    st.session_state.model_latency_ms = []
    current_input = initial_input
    total = len(agents)
    with st.status("Running agents...", state="running") as status:
        prog = st.progress(0, text="Initializing")
        for i, agent in enumerate(agents):
            st.write(f"Executing: {agent['name']} [{agent['model']}]")
            res = execute_agent(agent, current_input)
            st.session_state.agent_outputs.append({"agent": agent["name"], "model": agent["model"], "input": current_input, "output": res.text, "latency_ms": res.latency_ms, "error": res.error})
            if res.error: st.warning(f"Agent '{agent['name']}' error: {res.error}")
            current_input = res.text if res.text else current_input
            prog.progress((i + 1) / total, text=f"Completed {i + 1}/{total}")
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
    chart_latency = alt.Chart(df).mark_bar().encode(x=alt.X('agent:N', sort='-y', title="Agent"), y=alt.Y('latency_ms:Q', title="Latency (ms)"), color='provider:N', tooltip=['agent', 'model', 'latency_ms', 'tokens_est', 'provider']).properties(height=300)
    st.altair_chart(chart_latency, use_container_width=True)
    chart_provider = alt.Chart(df).mark_arc(innerRadius=40).encode(theta='count():Q', color='provider:N', tooltip=['provider', 'count()']).properties(height=300)
    st.altair_chart(chart_provider, use_container_width=True)
    st.markdown("Timeline")
    st.markdown('<div class="timeline">', unsafe_allow_html=True)
    for row in df.to_dict(orient="records"):
        st.markdown(f'<div class="event"><div class="title">{row["agent"]} <span class="badge badge-ok">{row["provider"]}</span></div><div class="meta">{row["model"]} • {row["latency_ms"]} ms • ~{row["tokens_est"]} tokens</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def download_outputs_section():
    if not st.session_state.agent_outputs: return
    st.subheader("Download Results")
    results_json = json.dumps(st.session_state.agent_outputs, ensure_ascii=False, indent=2)
    st.download_button("Download JSON", results_json, file_name="agent_results.json", mime="application/json")
    md_parts = [f"## {r['agent']} ({r['model']})\n\nOutput:\n\n{r['output']}\n" for r in st.session_state.agent_outputs]
    st.download_button("Download Markdown", "\n\n".join(md_parts), file_name="agent_results.md", mime="text/markdown")

# Sidebar
with st.sidebar:
    st.title("🌸 Agentic Processor")
    st.caption("Hugging Face Spaces • Streamlit")
    ensure_keys_ui()
    st.divider()
    st.subheader("Agents.yaml")
    uploaded_yaml = st.file_uploader("Upload agents.yaml", type=["yaml", "yml"])
    if uploaded_yaml:
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
    st.session_state.yaml_text = st.text_area("Edit agents.yaml", value=st.session_state.yaml_text or safe_yaml_dump(DEFAULT_AGENTS), height=260)
    if st.button("Validate & Apply YAML"):
        try:
            st.session_state.agents = init_agents_from_yaml_text(st.session_state.yaml_text)
            st.success("YAML validated and applied")
        except Exception as e:
            st.error(f"Validation failed: {e}")
    st.download_button("Download agents.yaml", data=st.session_state.yaml_text.encode("utf-8"), file_name="agents.yaml", mime="text/yaml")
    st.divider()
    st.subheader("OCR / Extraction")
    st.session_state.vision_ocr = st.toggle("Use LLM Vision OCR for image/PDF pages", value=False)

# Main Page
st.header("智能文件處理系統 · Agentic AI Document Processor")
st.caption("Upload a document, configure agents, and run the pipeline with multi-model support.")

st.subheader("1) Upload and Preview")
uf = st.file_uploader("Upload PDF/DOCX/TXT/Image", type=["pdf", "docx", "txt", "png", "jpg", "jpeg"])
extract_col1, extract_col2 = st.columns([3, 2])
with extract_col1:
    if uf:
        with st.status("Extracting text...", state="running"):
            st.session_state.doc_text = extract_text_from_file(uf, try_vision=st.session_state.vision_ocr, api_keys=st.session_state.api_keys) or ""
        st.success("Extraction complete.")
        st.session_state.doc_text = st.text_area("Editable extracted content", value=st.session_state.doc_text, height=280)
    else:
        st.info("Upload a file to begin.")
with extract_col2:
    if uf and uf.type == "application/pdf":
        st.caption("PDF preview is not fully supported in all environments; extraction will still work.")
    elif uf:
        st.image(uf, caption=uf.name, use_column_width=True)

st.subheader("2) Configure Agents")
if not st.session_state.agents:
    st.session_state.agents = init_agents_from_yaml_text(safe_yaml_dump(DEFAULT_AGENTS))
st.dataframe(pd.DataFrame(st.session_state.agents), use_container_width=True)
with st.expander("Advanced per-agent edits"):
    for i, a in enumerate(st.session_state.agents):
        st.markdown(f"**Agent {i + 1}: {a.get('name', 'Unnamed')}**")
        c1, c2, c3 = st.columns([1.2, 1, 1])
        with c1: a["name"] = st.text_input("Name", value=a["name"], key=f"name_{i}")
        with c2: a["model"] = st.selectbox("Model", options=MODEL_OPTIONS, index=MODEL_OPTIONS.index(a["model"]) if a["model"] in MODEL_OPTIONS else 1, key=f"model_{i}")
        with c3: a["temperature"] = st.slider("Temperature", 0.0, 1.0, float(a.get("temperature", 0.3)), 0.05, key=f"temp_{i}")
        a["top_p"] = st.slider("Top P", 0.0, 1.0, float(a.get("top_p", 0.9)), 0.05, key=f"topp_{i}")
        a["max_tokens"] = st.number_input("Max tokens", min_value=128, max_value=8192, value=int(a.get("max_tokens", 1000)), step=64, key=f"maxtok_{i}")
        stop_val = "" if not a.get("stop") else ",".join(a["stop"])
        a["stop"] = st.text_input("Stop sequences (comma-separated)", value=stop_val, key=f"stop_{i}")
        a["stop"] = [s.strip() for s in a["stop"].split(",")] if a["stop"] else None
        a["system_prompt"] = st.text_area("System prompt", value=a.get("system_prompt", ""), height=100, key=f"sys_{i}")
        a["user_prompt_template"] = st.text_area("User prompt template", value=a.get("user_prompt_template", "{input}"), height=120, key=f"user_{i}")
        st.divider()

st.subheader("3) Execute Agents")
left, right = st.columns([2, 1])
with left:
    if st.button("Run pipeline", type="primary", disabled=(len(st.session_state.doc_text.strip()) == 0)):
        st.session_state.run_started = True
        run_pipeline(st.session_state.agents, st.session_state.doc_text)
    if st.session_state.agent_outputs:
        st.success("All agents executed.")
        for r in st.session_state.agent_outputs:
            status_cls = "badge-ok" if not r["error"] else "badge-danger"
            st.markdown(f'<span class="badge {status_cls}">{r["agent"]} · {r["model"]} · {r["latency_ms"]} ms</span>', unsafe_allow_html=True)
            if r["error"]:
                st.error(f"Error: {r['error']}")
            st.text_area(f"Output - {r['agent']}", value=r["output"], height=150, key=f"out_{r['agent']}")
        download_outputs_section()
with right:
    render_dashboard()

with st.expander("Grok Vision Example (sample code)"):
    st.code('''
import os
from xai_sdk import Client
from xai_sdk.chat import user, system, image
client = Client(api_key=os.getenv("XAI_API_KEY"), timeout=3600)
chat = client.chat.create(model="grok-4")
chat.append(system("You are Grok, a highly intelligent, helpful AI assistant."))
chat.append(user("What's in this image?", image("https://science.nasa.gov/wp-content/uploads/2023/09/web-first-images-release.png")))
response = chat.sample()
print(response.content)''', language="python")
