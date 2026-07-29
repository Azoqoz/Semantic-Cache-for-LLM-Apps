from __future__ import annotations

import pandas as pd
import streamlit as st

from src.cache_store import SQLiteSemanticCache
from src.config import settings
from src.embeddings import EmbeddingService
from src.evaluation import build_evaluation_report
from src.llm_providers import ProviderConfigurationError, build_provider
from src.semantic_cache import SemanticCacheService


st.set_page_config(page_title="Semantic Cache for LLM Apps", page_icon="⚡", layout="wide")

st.markdown(
    """
    <style>
    .stApp { color: #E6EDF7; background: #0B1220; }
    .block-container { max-width: 1280px; padding-top: 2.1rem; padding-bottom: 3rem; }
    [data-testid="stSidebar"] { background: #101827; border-right: 1px solid #34445C; }
    [data-testid="stSidebar"] .block-container { padding-top: 1.75rem; }
    [data-testid="stSidebar"] h4 { margin-top: .65rem; margin-bottom: .15rem; }
    [data-testid="stMetric"] {
        background: #1D2A40; border: 1px solid #34445C; border-radius: 13px;
        padding: .95rem 1rem; box-shadow: 0 3px 10px rgba(0, 0, 0, 0.16);
        transition: border-color .16s ease, transform .16s ease;
    }
    [data-testid="stMetric"]:hover { border-color: #465B78; transform: translateY(-1px); }
    [data-testid="stMetricLabel"] { color: #9EADC2; }
    [data-testid="stMetricValue"] { color: #F3F7FC; font-size: 1.72rem; font-weight: 700; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #172235; border-color: #34445C; border-radius: 14px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.17);
    }
    .stTabs [data-baseweb="tab-list"] { gap: .65rem; border-bottom: 1px solid #34445C; }
    .stTabs [data-baseweb="tab"] { color: #73839A; padding: .75rem 1.05rem; border-radius: 8px 8px 0 0; }
    .stTabs [aria-selected="true"] { background: #1D3557; color: #86B0FF; box-shadow: inset 0 -3px 0 #6C9EFF; }
    .stButton > button { border-radius: 10px; font-weight: 500; border-color: #34445C; }
    .stButton > button[kind="primary"] { background: #5F8FE8; color: #FFFFFF; border-color: #6C9EFF; }
    .stButton > button[kind="primary"]:hover { background: #75A3F5; border-color: #86B0FF; }
    .stButton > button[kind="secondary"] { background: #172235; color: #9EADC2; border-color: #34445C; }
    .stDataFrame { border: 1px solid #34445C; border-radius: 10px; overflow: hidden; }
    [data-testid="stDataFrame"] { background: #172235; }
    h1 { color: #F3F7FC; font-weight: 750; }
    h2 { color: #F3F7FC; font-weight: 700; margin-bottom: .65rem; }
    h3, h4 { color: #E6EDF7; font-weight: 650; letter-spacing: -0.015em; margin-bottom: .5rem; }
    p, label, [data-testid="stMarkdownContainer"] { color: #E6EDF7; }
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: #9EADC2; }
    .hero { padding: .4rem 0 1.35rem; }
    .hero h1 { color: #F3F7FC; margin: 0; letter-spacing: -0.035em; }
    .hero p { color: #9EADC2; font-size: 1.05rem; margin: .45rem 0 1rem; }
    .badge { display: inline-block; background: #1D3557; color: #86B0FF;
        border: 1px solid #34445C; border-radius: 999px; padding: .28rem .7rem;
        margin: 0 .35rem .3rem 0; font-size: .82rem; font-weight: 600; }
    .mode-badge { display: inline-block; background: #153F38; color: #A7F3D0;
        border: 1px solid #246B5B; border-radius: 999px; padding: .3rem .72rem;
        margin: 0 0 .85rem; font-size: .8rem; font-weight: 700; }
    .mode-info { background: #17324D; color: #BFDBFE; border: 1px solid #34445C;
        border-radius: 10px; padding: .65rem .75rem; margin: .4rem 0 .85rem;
        font-size: .82rem; line-height: 1.42; }
    .mode-info p { color: #BFDBFE; margin: 0; }
    .mode-info p + p { margin-top: .3rem; }
    .section-note { color: #9EADC2; margin-top: -.55rem; margin-bottom: .8rem; }
    [data-baseweb="input"] > div, [data-baseweb="select"] > div,
    [data-baseweb="textarea"] > div, textarea, input {
        background: #131E2F !important; color: #E6EDF7 !important; border-color: #34445C !important;
    }
    input::placeholder, textarea::placeholder { color: #73839A !important; opacity: 1; }
    [data-baseweb="input"] > div:focus-within, [data-baseweb="select"] > div:focus-within,
    [data-baseweb="textarea"] > div:focus-within { border-color: #6C9EFF !important; box-shadow: 0 0 0 1px #6C9EFF; }
    [data-baseweb="popover"], [role="listbox"] { background: #172235 !important; color: #E6EDF7 !important; }
    [data-testid="stExpander"] { background: #172235; border: 1px solid #34445C; border-radius: 12px; }
    [data-testid="stAlert"] { border: 1px solid #34445C; border-radius: 10px; }
    [data-testid="stAlert"] [data-testid="stMarkdownContainer"] p { color: inherit; }
    div[data-baseweb="notification"][kind="positive"], .stSuccess { background: #153F38; color: #A7F3D0; }
    div[data-baseweb="notification"][kind="info"], .stInfo { background: #17324D; color: #BFDBFE; }
    div[data-baseweb="notification"][kind="warning"], .stWarning { background: #4A3A1A; color: #FDE68A; }
    div[data-baseweb="notification"][kind="negative"], .stError { background: #4A232A; color: #FCA5A5; }
    hr { border-color: #34445C !important; }
    [data-testid="stSlider"] [role="slider"] { background: #6C9EFF; border-color: #6C9EFF; }
    [data-baseweb="slider"] > div > div { background-color: #6C9EFF; }
    .state-marker, .accent-metric-marker, .recommended-card-marker { display: none; }
    [data-testid="stColumn"]:has(.accent-metric-marker) [data-testid="stMetricValue"] { color: #86B0FF; }
    [data-testid="stColumn"]:has(.cache-hit-marker) [data-testid="stMetric"] { background: #123D35; border-color: #246B5B; }
    [data-testid="stColumn"]:has(.cache-hit-marker) [data-testid="stMetricValue"] { color: #A7F3D0; }
    [data-testid="stColumn"]:has(.cache-miss-marker) [data-testid="stMetric"] { background: #3B2F16; border-color: #80621F; }
    [data-testid="stColumn"]:has(.cache-miss-marker) [data-testid="stMetricValue"] { color: #FDE68A; }
    [data-testid="stVerticalBlockBorderWrapper"]:has(.recommended-card-marker) {
        background: #1D3557; border-color: #6C9EFF; box-shadow: 0 0 0 1px rgba(108, 158, 255, .15);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_services() -> tuple[EmbeddingService, SQLiteSemanticCache, SemanticCacheService]:
    """Create and retain expensive application services across reruns."""
    embedding_service = EmbeddingService(settings.embedding_model)
    cache = SQLiteSemanticCache(settings.database_path, embedding_service)
    return embedding_service, cache, SemanticCacheService(cache, embedding_service)


embedding_service, cache, semantic_cache = get_services()
st.markdown(
    """<div class="hero">
    <h1>Semantic Cache for LLM Apps</h1>
    <p>Reuse answers for semantically similar questions to reduce latency and avoid unnecessary LLM calls.</p>
    <span class="badge">Semantic matching</span><span class="badge">Lower latency</span>
    <span class="badge">Cost savings</span>
    </div>""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Settings")
    mode_label = "Hosted Demo" if settings.app_mode == "demo" else "Local Full Mode"
    st.markdown(f'<span class="mode-badge">{mode_label}</span>', unsafe_allow_html=True)
    st.markdown("#### Provider")
    if settings.app_mode == "demo":
        provider_name = "Demo"
        st.markdown("**Demo**")
        st.markdown(
            """<div class="mode-info">
            <p>OpenAI, Claude, Gemini, and Ollama are available in local mode.</p>
            <p>Clone the repository and add your own provider credentials to <code>.env</code> to use the full version.</p>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        provider_name = st.selectbox(
            "LLM provider",
            ["Demo", "OpenAI", "Claude", "Gemini", "Ollama"],
            help="Demo runs without an API key. Ollama runs locally.",
        )

    openai_model = settings.openai_model
    claude_model = settings.claude_model
    gemini_model = settings.gemini_model
    ollama_model = settings.ollama_model
    ollama_base_url = settings.ollama_base_url
    if provider_name == "Demo":
        st.caption("Runs without an API key.")
    elif provider_name == "OpenAI":
        openai_model = st.text_input("OpenAI model", settings.openai_model)
        st.caption("Requires `OPENAI_API_KEY` to be configured in `.env`.")
    elif provider_name == "Claude":
        claude_model = st.text_input("Claude model", settings.claude_model)
        st.caption("Requires `ANTHROPIC_API_KEY` to be configured in `.env`.")
    elif provider_name == "Gemini":
        gemini_model = st.text_input("Gemini model", settings.gemini_model)
        st.caption("Requires `GEMINI_API_KEY` to be configured in `.env`.")
    elif provider_name == "Ollama":
        ollama_model = st.text_input("Ollama model", settings.ollama_model)
        ollama_base_url = st.text_input("Ollama URL", settings.ollama_base_url)
        st.caption("Ollama must be running locally at the configured URL.")

    st.markdown("#### Cache")
    threshold = st.slider("Similarity threshold", 0.50, 0.99, settings.default_threshold, 0.01, help="Higher values require a closer semantic match.")
    ttl_hours = st.number_input("Cache TTL (hours)", 0, 8760, settings.default_ttl_hours, 24, help="Use 0 to keep entries indefinitely.")
    isolate_by_model = st.toggle("Isolate cache by provider/model", value=True, help="Prevents answers generated by one model from being reused by another.")

    st.markdown("#### Reset data")
    with st.container(border=True):
        st.caption("This removes all cached answers and query metrics.")
        if st.button("Clear cache and metrics", type="secondary", use_container_width=True):
            cache.clear()
            st.success("Cache and query metrics were cleared.")
            st.rerun()

tab_demo, tab_dashboard, tab_evaluation, tab_system = st.tabs(
    ["Demo", "Dashboard", "Evaluation", "Technical"]
)

with tab_demo:
    st.subheader("Try the semantic cache")
    if settings.app_mode == "demo":
        st.info(
            "This hosted version demonstrates semantic caching without external API calls."
        )
    with st.container(border=True):
        examples = ["What is semantic caching?", "How does semantic caching work?", "How can semantic caching reduce LLM cost?", "What is an embedding?", "What is cosine similarity?", "What is a cache hit?", "What is a cache miss?", "Why is threshold tuning important?"]
        selected_example = st.selectbox("Start with an example", [""] + examples)
        question = st.text_area("Your question", value=selected_example, height=120, placeholder="Enter a question...")
        generate_answer = st.button("Generate answer", type="primary", use_container_width=True)
    if generate_answer:
        if not question.strip():
            st.error("Unable to generate an answer: Question cannot be empty.")
            st.stop()
        try:
            provider = build_provider(
                provider_name=provider_name,
                openai_model=openai_model,
                claude_model=claude_model,
                gemini_model=gemini_model,
                ollama_base_url=ollama_base_url,
                ollama_model=ollama_model,
                app_mode=settings.app_mode,
            )
            with st.spinner("Checking semantic cache..."):
                result = semantic_cache.answer(question, provider, threshold, int(ttl_hours), isolate_by_model)
            with st.container(border=True):
                st.markdown("### Result")
                status, similarity, used_threshold, latency = st.columns(4)
                status.markdown(
                    '<span class="state-marker cache-hit-marker"></span>'
                    if result.cache_hit
                    else '<span class="state-marker cache-miss-marker"></span>',
                    unsafe_allow_html=True,
                )
                status.metric("Result", "CACHE HIT" if result.cache_hit else "CACHE MISS")
                similarity.metric("Similarity", f"{result.similarity:.3f}")
                used_threshold.metric("Threshold", f"{threshold:.2f}")
                latency.metric("Latency", f"{result.latency_ms:.0f} ms")
                provider_col, model_col, cost_col = st.columns(3)
                provider_col.metric("Provider", result.provider)
                model_col.metric("Model", result.model)
                cost_col.metric("Cost avoided" if result.cache_hit else "LLM cost", f"${(result.estimated_cost_usd or 0):.4f}")
                if result.cache_hit:
                    st.success(f'Cache hit — matched “{result.matched_question}”')
                else:
                    st.info("Cache miss — the provider generated and cached a new answer.")
                    if result.matched_question:
                        st.caption(f'Closest cached question below threshold: “{result.matched_question}”')
                st.markdown("#### Answer")
                st.write(result.answer)
        except ProviderConfigurationError as exc:
            st.error(f"Unable to generate an answer: {exc}")
        except Exception:
            st.error(
                "Unable to generate an answer. Check the selected provider's "
                "local configuration and try again."
            )

with tab_dashboard:
    st.subheader("Dashboard")
    st.markdown('<p class="section-note">Usage, performance, savings, and recent cache activity.</p>', unsafe_allow_html=True)
    metrics = cache.metrics()
    st.markdown("### Usage")
    with st.container(border=True):
        row = st.columns(4)
        row[0].metric("Total queries", metrics["total_queries"])
        row[1].metric("Cache hits", metrics["cache_hits"])
        row[2].metric("Cache misses", metrics["cache_misses"])
        row[3].markdown('<span class="accent-metric-marker"></span>', unsafe_allow_html=True)
        row[3].metric("Hit rate", f"{metrics['hit_rate'] * 100:.1f}%")
    st.markdown("### Performance")
    with st.container(border=True):
        average_miss_latency = float(metrics["average_miss_latency_ms"])
        average_hit_latency = float(metrics["average_hit_latency_ms"])
        latency_reduction = (
            (average_miss_latency - average_hit_latency) / average_miss_latency * 100
            if average_miss_latency > 0
            else 0.0
        )
        row = st.columns(3)
        row[0].metric("Average cache-hit latency", f"{metrics['average_hit_latency_ms']:.0f} ms")
        row[1].metric("Average cache-miss latency", f"{metrics['average_miss_latency_ms']:.0f} ms")
        row[2].markdown('<span class="accent-metric-marker"></span>', unsafe_allow_html=True)
        row[2].metric("Latency reduction", f"{latency_reduction:.1f}%")
        st.caption("Cache hits should be significantly faster than cache misses.")
    st.markdown("### Cost savings")
    with st.container(border=True):
        row = st.columns(4)
        row[0].metric("Actual LLM cost", f"${metrics['llm_cost_usd']:.4f}")
        row[1].metric("Avoided cost", f"${metrics['avoided_cost_usd']:.4f}")
        row[2].metric("Without caching", f"${metrics['total_cost_without_cache_usd']:.4f}")
        row[3].markdown('<span class="accent-metric-marker"></span>', unsafe_allow_html=True)
        row[3].metric("Savings", f"{metrics['savings_percentage']:.1f}%")
    st.caption("Cost values are illustrative estimates unless the selected provider returns exact billing information.")
    st.markdown("### Cache summary")
    entries = cache.list_entries()
    with st.container(border=True):
        summary = st.columns(2)
        summary[0].metric("Cached entries", metrics["cache_entries"])
        summary[1].metric("Total reuses", metrics["total_reuses"])
        if entries:
            recent_frame = pd.DataFrame(entries[:5])
            recent_frame["answer_preview"] = recent_frame["answer"].str.slice(0, 80)
            st.dataframe(
                recent_frame[["question", "answer_preview", "provider", "model", "access_count"]],
                use_container_width=True,
                hide_index=True,
                height=212,
            )
        else:
            st.info("No cached entries yet. Generate an answer in the Demo tab to get started.")

with tab_evaluation:
    st.subheader("Similarity Evaluation")
    st.markdown('<p class="section-note">Explore model quality across labeled Easy, Medium, and Hard question pairs.</p>', unsafe_allow_html=True)
    with st.container(border=True):
        evaluation_threshold = st.slider("Evaluation threshold", 0.50, 0.99, settings.default_threshold, 0.01, key="evaluation_threshold")
    try:
        with st.spinner("Evaluating question pairs..."):
            evaluation_report = build_evaluation_report(embedding_service, evaluation_threshold)
        st.markdown("### Recommended Threshold")
        with st.container(border=True):
            st.markdown('<span class="recommended-card-marker"></span>', unsafe_allow_html=True)
            st.success(f"Recommended threshold: {evaluation_report.recommended_threshold:.2f}")
            st.caption("Recommended based on the highest F1 score.")

        st.markdown("### Selected Threshold Results")
        evaluation_metrics = evaluation_report.selected_metrics
        with st.container(border=True):
            row = st.columns(4)
            row[0].metric("Accuracy", f"{evaluation_metrics['accuracy'] * 100:.1f}%")
            row[1].metric("Precision", f"{evaluation_metrics['precision'] * 100:.1f}%")
            row[2].metric("Recall", f"{evaluation_metrics['recall'] * 100:.1f}%")
            row[3].metric("F1 score", f"{evaluation_metrics['f1'] * 100:.1f}%")
            row = st.columns(4)
            row[0].metric("True positives", evaluation_metrics["true_positives"])
            row[1].metric("True negatives", evaluation_metrics["true_negatives"])
            row[2].metric("False positives", evaluation_metrics["false_positives"])
            row[3].metric("False negatives", evaluation_metrics["false_negatives"])

        st.markdown("### Results by Difficulty")
        difficulty_frame = pd.DataFrame(evaluation_report.difficulty_rows)
        difficulty_frame["Accuracy"] = difficulty_frame["Accuracy"].map(lambda value: f"{value * 100:.1f}%")
        with st.container(border=True):
            st.dataframe(difficulty_frame, use_container_width=True, hide_index=True)

        st.markdown("### Threshold Comparison")
        comparison_frame = pd.DataFrame(evaluation_report.comparison_rows)
        for metric_column in ("Accuracy", "Precision", "Recall", "F1 score"):
            comparison_frame[metric_column] = comparison_frame[metric_column].map(lambda value: f"{value * 100:.1f}%")
        comparison_frame["Threshold"] = comparison_frame["Threshold"].map(lambda value: f"{value:.2f}")
        recommended_threshold = f"{evaluation_report.recommended_threshold:.2f}"
        comparison_styler = comparison_frame.style.apply(
            lambda row: [
                "background-color: #1D3557; color: #E6EDF7; font-weight: 600"
                if row["Threshold"] == recommended_threshold
                else ""
                for _ in row
            ],
            axis=1,
        )
        with st.container(border=True):
            st.dataframe(comparison_styler, use_container_width=True, hide_index=True)

        st.markdown("### Detailed Pair Results")
        with st.expander("View detailed evaluation pairs"):
            st.dataframe(pd.DataFrame(evaluation_report.selected_rows), use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(f"Unable to run evaluation: {exc}")

with tab_system:
    st.subheader("System")
    st.markdown('<p class="section-note">Request architecture and cache administration.</p>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("### Request flow")
        st.code("""User question
    ↓
Local embedding model
    ↓
Exact match check, then cosine similarity
    ↓
Similarity ≥ threshold?
    ├── Yes → Return cached answer → Record cache hit
    └── No  → Call LLM → Cache result → Record cache miss""", language="text")
    with st.container(border=True):
        st.markdown("""### Engineering decisions

- Embeddings run locally, so similarity lookup does not require an embedding API.
- SQLite provides persistent storage, TTL, access counters, and analytics.
- Provider/model isolation prevents accidental cross-model answer reuse.
- Query events separately track hit/miss latency and estimated cost.
- Evaluation logic remains separate from the Streamlit UI.""")

    st.markdown("### Cache Explorer")
    entries = cache.list_entries()
    if not entries:
        st.info("The cache is empty. Generate an answer in Demo to create the first entry.")
    else:
        frame = pd.DataFrame(entries)
        frame["answer_preview"] = frame["answer"].str.slice(0, 120)
        display_columns = ["id", "question", "answer_preview", "provider", "model", "access_count", "created_at", "expires_at"]
        with st.container(border=True):
            st.dataframe(frame[display_columns], use_container_width=True, hide_index=True, height=420)
            st.caption("Deletion is permanent and only affects the selected cache entries.")
            selected_ids = st.multiselect("Select entries to delete", options=frame["id"].tolist())
            if st.button("Delete selected entries", disabled=not selected_ids, type="secondary"):
                cache.delete_entries(selected_ids)
                st.success("Selected entries were deleted.")
                st.rerun()
