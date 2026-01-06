"""
Factor - Fact Bank Demo UI

Streamlit application for document ingestion and fact bank visualization.
"""

import asyncio
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import Factor modules
import uuid
from src.extraction.pdf_parser import extract_pdf
from src.extraction.excel_parser import extract_excel
from src.extraction.normalizer import ExtractedDocument
from src.extraction.fact_extractor import FactExtractor
from src.storage.models import Fact, Document, SourceType, RelevanceLevel, Believability, STANDARD_CHAPTERS, FactCorrection
from src.storage.repository import FactRepository
from src.utils.embeddings import deduplicate_facts, EmbeddingClient
from src.scoring.scorer import FactScorer, get_chapter_summary
from src.analysis.conflict_detector import ConflictDetector, FactConflict, ConflictSeverity

# Page config
st.set_page_config(
    page_title="Factor - Fact Bank",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1400px;
    }
    
    /* Header styling */
    .factor-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .factor-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    .factor-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }
    
    /* Stats cards */
    .stat-card {
        background: linear-gradient(145deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        transition: transform 0.2s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e3a5f;
        line-height: 1.2;
    }
    
    .stat-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.3rem;
    }
    
    /* Relevance badges */
    .badge-critical {
        background-color: #dc2626;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .badge-chapter-specific {
        background-color: #2563eb;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .badge-additional-context {
        background-color: #6b7280;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .badge-noise {
        background-color: #9ca3af;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        opacity: 0.7;
    }
    
    /* Believability badges */
    .badge-verified {
        background-color: #16a34a;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .badge-needs-verification {
        background-color: #f59e0b;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .badge-opinion {
        background-color: #7c3aed;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    /* Legacy importance badges (for backwards compat) */
    .badge-high {
        background-color: #dc2626;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .badge-medium {
        background-color: #f59e0b;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .badge-low {
        background-color: #6b7280;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    /* Processing status */
    .processing-status {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Sidebar improvements */
    .css-1d391kg {
        padding-top: 1rem;
    }
    
    /* Table styling */
    .dataframe {
        font-size: 0.9rem;
    }
    
    /* Filter section */
    .filter-section {
        background: #f8fafc;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Conflict severity badges */
    .badge-critical {
        background-color: #7f1d1d;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .badge-conflict-high {
        background-color: #dc2626;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .badge-conflict-medium {
        background-color: #f59e0b;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .badge-conflict-low {
        background-color: #6b7280;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    /* Conflict card */
    .conflict-card {
        background: linear-gradient(145deg, #fef2f2 0%, #fee2e2 100%);
        border: 1px solid #fecaca;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    
    .conflict-card.critical {
        background: linear-gradient(145deg, #fef2f2 0%, #fecaca 100%);
        border-color: #f87171;
    }
    
    .conflict-card.high {
        background: linear-gradient(145deg, #fef2f2 0%, #fee2e2 100%);
        border-color: #fca5a5;
    }
    
    .conflict-card.medium {
        background: linear-gradient(145deg, #fffbeb 0%, #fef3c7 100%);
        border-color: #fcd34d;
    }
    
    .conflict-card.low {
        background: linear-gradient(145deg, #f9fafb 0%, #f3f4f6 100%);
        border-color: #d1d5db;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
def init_session_state():
    """Initialize session state variables."""
    if "repository" not in st.session_state:
        st.session_state.repository = FactRepository()
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "last_extraction_count" not in st.session_state:
        st.session_state.last_extraction_count = 0
    if "detected_conflicts" not in st.session_state:
        st.session_state.detected_conflicts = []
    if "conflicts_checked" not in st.session_state:
        st.session_state.conflicts_checked = False
    if "show_noise" not in st.session_state:
        st.session_state.show_noise = False
    # Pending files with individual source type assignments
    if "pending_files" not in st.session_state:
        st.session_state.pending_files = {}  # {filename: {"file": file_obj, "source_type": SourceType}}
    # Track last known facts for detecting edits
    if "last_facts_df" not in st.session_state:
        st.session_state.last_facts_df = None


init_session_state()


def get_believability_color(believability: str) -> str:
    """Get color for believability level."""
    colors = {
        "verified": "#16a34a",  # Green
        "needs_verification": "#f59e0b",  # Yellow/Amber
        "opinion": "#7c3aed",  # Purple
    }
    return colors.get(believability, "#6b7280")


def get_relevance_color(relevance: str) -> str:
    """Get color for relevance level."""
    colors = {
        "critical": "#dc2626",  # Red
        "chapter_specific": "#2563eb",  # Blue
        "additional_context": "#6b7280",  # Gray
        "noise": "#9ca3af",  # Light Gray
    }
    return colors.get(relevance, "#6b7280")


def format_relevance_badge(relevance: str) -> str:
    """Format relevance as HTML badge."""
    display_names = {
        "critical": "Critical",
        "chapter_specific": "Chapter Specific",
        "additional_context": "Additional Context",
        "noise": "Noise",
    }
    color = get_relevance_color(relevance)
    display_name = display_names.get(relevance, relevance)
    opacity = "opacity: 0.7;" if relevance == "noise" else ""
    return f'<span style="background-color:{color};color:white;padding:0.2rem 0.6rem;border-radius:9999px;font-size:0.75rem;font-weight:600;{opacity}">{display_name}</span>'


def format_believability_badge(believability: str) -> str:
    """Format believability as HTML badge."""
    display_names = {
        "verified": "Verified",
        "needs_verification": "Needs Verification",
        "opinion": "Opinion",
    }
    color = get_believability_color(believability)
    display_name = display_names.get(believability, believability)
    return f'<span style="background-color:{color};color:white;padding:0.2rem 0.6rem;border-radius:9999px;font-size:0.75rem;font-weight:600;">{display_name}</span>'


async def process_documents(
    files_with_types: dict[str, dict],
    run_dedup: bool = True,
) -> tuple[int, int]:
    """
    Process uploaded documents and extract facts.
    
    Args:
        files_with_types: Dictionary mapping filename to {"file": file_obj, "source_type": SourceType}
        run_dedup: Whether to run deduplication
    
    Returns:
        Tuple of (facts_before_dedup, facts_after_dedup)
    """
    repo = st.session_state.repository
    all_facts = []
    
    # Create extractor
    extractor = FactExtractor()
    
    try:
        for file_name, file_info in files_with_types.items():
            uploaded_file = file_info["file"]
            source_type = file_info["source_type"]
            file_ext = Path(file_name).suffix.lower()
            
            source_type_display = {
                SourceType.COMPANY_PRESENTATION: "Company Presentation",
                SourceType.MARKET_RESEARCH: "Market Research",
                SourceType.FINANCIALS: "Financial Data",
            }
            st.write(f"Processing: **{file_name}** as *{source_type_display.get(source_type, source_type)}*")
            
            # Extract document content
            if file_ext == ".pdf":
                doc = extract_pdf(uploaded_file, filename=file_name)
            elif file_ext in (".xlsx", ".xls"):
                doc = extract_excel(uploaded_file, filename=file_name)
            else:
                st.warning(f"Unsupported file type: {file_ext}")
                continue
            
            # Store document metadata
            document = Document(
                id=str(uuid.uuid4()),
                filename=file_name,
                file_type=file_ext.lstrip("."),
                source_type=source_type,
                raw_content=doc.get_full_text()[:50000],  # Limit stored content
                page_count=doc.total_pages if file_ext == ".pdf" else None,
                sheet_count=doc.total_pages if file_ext in (".xlsx", ".xls") else None,
            )
            repo.insert_document(document)
            
            # Extract facts
            st.write("Extracting facts...")
            facts = await extractor.extract_facts(doc, source_type)
            
            st.write(f"Found **{len(facts)}** facts")
            all_facts.extend(facts)
            
            # Track processed file
            st.session_state.processed_files.append(file_name)
        
        facts_before = len(all_facts)
        
        # Deduplicate if enabled and multiple documents
        if run_dedup and len(all_facts) > 1:
            st.write("Running deduplication...")
            all_facts = await deduplicate_facts(all_facts)
        
        facts_after = len(all_facts)
        
        # Store facts in database
        if all_facts:
            repo.insert_facts(all_facts)
            
            # Update document fact counts
            for doc in repo.get_all_documents():
                fact_count = len(repo.get_facts_by_source(doc.filename))
                repo.update_document_fact_count(doc.id, fact_count)
        
        return facts_before, facts_after
        
    finally:
        await extractor.close()


def render_header():
    """Render the page header."""
    st.markdown("""
    <div class="factor-header">
        <h1>📊 Factor — Fact Bank</h1>
        <p>Extract, score, and manage investment facts from your documents</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with upload and controls."""
    with st.sidebar:
        st.header("📁 Document Upload")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload PDF or Excel files",
            type=["pdf", "xlsx", "xls"],
            accept_multiple_files=True,
            help="Upload company presentations, market research, or financial documents",
            key="file_uploader",
        )
        
        # Track uploaded files and sync with pending_files state
        if uploaded_files:
            current_filenames = {f.name for f in uploaded_files}
            
            # Add new files to pending_files with default type
            for f in uploaded_files:
                if f.name not in st.session_state.pending_files:
                    st.session_state.pending_files[f.name] = {
                        "file": f,
                        "source_type": SourceType.COMPANY_PRESENTATION,
                    }
                else:
                    # Update file object (in case of re-upload)
                    st.session_state.pending_files[f.name]["file"] = f
            
            # Remove files that were removed from uploader
            to_remove = [fn for fn in st.session_state.pending_files if fn not in current_filenames]
            for fn in to_remove:
                del st.session_state.pending_files[fn]
        else:
            # No files uploaded, clear pending files
            st.session_state.pending_files = {}
        
        # Show individual file type selectors
        if st.session_state.pending_files:
            st.markdown("### Assign Document Types")
            st.caption("Select a category for each document:")
            
            source_type_options = [
                ("Company Presentation", SourceType.COMPANY_PRESENTATION),
                ("Market Research", SourceType.MARKET_RESEARCH),
                ("Financial Data", SourceType.FINANCIALS),
            ]
            
            for filename in list(st.session_state.pending_files.keys()):
                file_info = st.session_state.pending_files[filename]
                
                # Create a container for each file
                with st.container():
                    col1, col2 = st.columns([2, 3])
                    
                    with col1:
                        # Truncate long filenames for display
                        display_name = filename if len(filename) <= 20 else filename[:17] + "..."
                        st.markdown(f"**📄 {display_name}**")
                    
                    with col2:
                        # Get current selection index
                        current_type = file_info["source_type"]
                        current_idx = next(
                            (i for i, opt in enumerate(source_type_options) if opt[1] == current_type),
                            0
                        )
                        
                        selected = st.selectbox(
                            f"Type for {filename}",
                            options=source_type_options,
                            index=current_idx,
                            format_func=lambda x: x[0],
                            key=f"type_{filename}",
                            label_visibility="collapsed",
                        )
                        
                        # Update the source type
                        st.session_state.pending_files[filename]["source_type"] = selected[1]
            
            st.divider()
        
        # Deduplication toggle
        run_dedup = st.checkbox(
            "Enable deduplication",
            value=True,
            help="Detect and merge duplicate facts across documents",
        )
        
        # Process button
        if st.session_state.pending_files:
            if st.button("🚀 Process Documents", type="primary", use_container_width=True):
                st.session_state.processing = True
                
                with st.spinner("Processing documents..."):
                    # Run async processing
                    loop = asyncio.new_event_loop()
                    try:
                        before, after = loop.run_until_complete(
                            process_documents(st.session_state.pending_files.copy(), run_dedup)
                        )
                        st.session_state.last_extraction_count = after
                        
                        # Clear pending files after successful processing
                        st.session_state.pending_files = {}
                        
                        if run_dedup and before != after:
                            st.success(f"✅ Extracted {before} facts, {before - after} duplicates merged → {after} unique facts")
                        else:
                            st.success(f"✅ Extracted {after} facts successfully!")
                        
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error processing documents: {e}")
                        logger.exception("Document processing error")
                    finally:
                        loop.close()
                        st.session_state.processing = False
        
        st.divider()
        
        # Database actions
        st.header("🗄️ Database")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.repository.clear_all_facts()
                st.session_state.repository.clear_all_documents()
                st.session_state.repository.clear_all_corrections()
                st.session_state.processed_files = []
                st.session_state.pending_files = {}
                st.success("Database cleared!")
                st.rerun()
        
        # API Status
        st.divider()
        st.header("🔌 API Status")
        
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")
        
        if openrouter_key:
            st.success("OpenRouter: Connected")
        else:
            st.warning("OpenRouter: No API key")
        
        if groq_key:
            st.success("Groq: Connected")
        else:
            st.info("Groq: Not configured")
    
    return uploaded_files


def render_statistics():
    """Render the statistics panel."""
    repo = st.session_state.repository
    stats = repo.get_statistics()
    
    # Main stats row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{stats['total_documents']}</div>
            <div class="stat-label">Documents</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{stats['total_facts']}</div>
            <div class="stat-label">Total Facts</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        verified_count = stats.get('believability_breakdown', {}).get('verified', 0)
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{verified_count}</div>
            <div class="stat-label">Verified Facts</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        critical_count = stats.get('relevance_breakdown', {}).get('critical', 0)
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{critical_count}</div>
            <div class="stat-label">Critical Facts</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Relevance breakdown
    if stats['total_facts'] > 0:
        st.markdown("### Relevance Breakdown")
        
        relevance = stats.get('relevance_breakdown', {})
        total = sum(relevance.values())
        
        if total > 0:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                critical_pct = relevance.get('critical', 0) / total * 100
                st.metric("🔴 Critical", f"{relevance.get('critical', 0)}", f"{critical_pct:.1f}%")
            
            with col2:
                chapter_pct = relevance.get('chapter_specific', 0) / total * 100
                st.metric("🔵 Chapter Specific", f"{relevance.get('chapter_specific', 0)}", f"{chapter_pct:.1f}%")
            
            with col3:
                context_pct = relevance.get('additional_context', 0) / total * 100
                st.metric("⚪ Additional Context", f"{relevance.get('additional_context', 0)}", f"{context_pct:.1f}%")
            
            with col4:
                noise_pct = relevance.get('noise', 0) / total * 100
                st.metric("🔇 Noise", f"{relevance.get('noise', 0)}", f"{noise_pct:.1f}%")
        
        st.markdown("### Believability Breakdown")
        
        believability = stats.get('believability_breakdown', {})
        total_bel = sum(believability.values())
        
        if total_bel > 0:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                verified_pct = believability.get('verified', 0) / total_bel * 100
                st.metric("✅ Verified", f"{believability.get('verified', 0)}", f"{verified_pct:.1f}%")
            
            with col2:
                needs_ver_pct = believability.get('needs_verification', 0) / total_bel * 100
                st.metric("⚠️ Needs Verification", f"{believability.get('needs_verification', 0)}", f"{needs_ver_pct:.1f}%")
            
            with col3:
                opinion_pct = believability.get('opinion', 0) / total_bel * 100
                st.metric("💭 Opinion", f"{believability.get('opinion', 0)}", f"{opinion_pct:.1f}%")


def render_filters():
    """Render the filter controls."""
    repo = st.session_state.repository
    stats = repo.get_statistics()
    
    st.markdown("### 🔍 Filters")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        # Source document filter
        source_options = ["All Documents"] + stats.get('source_documents', [])
        selected_source = st.selectbox(
            "Source Document",
            options=source_options,
        )
    
    with col2:
        # Relevance filter
        relevance_options = [
            ("All", "all"),
            ("Critical", "critical"),
            ("Chapter Specific", "chapter_specific"),
            ("Additional Context", "additional_context"),
            ("Noise", "noise"),
        ]
        selected_relevance = st.selectbox(
            "Relevance",
            options=[opt[1] for opt in relevance_options],
            format_func=lambda x: next((opt[0] for opt in relevance_options if opt[1] == x), x),
        )
    
    with col3:
        # Believability filter
        believability_options = [
            ("All", "all"),
            ("Verified", "verified"),
            ("Needs Verification", "needs_verification"),
            ("Opinion", "opinion"),
        ]
        selected_believability = st.selectbox(
            "Believability",
            options=[opt[1] for opt in believability_options],
            format_func=lambda x: next((opt[0] for opt in believability_options if opt[1] == x), x),
        )
    
    with col4:
        # Chapter filter
        chapter_options = ["All Chapters"] + STANDARD_CHAPTERS
        selected_chapter = st.selectbox(
            "Chapter Relevance",
            options=chapter_options,
        )
    
    with col5:
        # Search box
        search_query = st.text_input(
            "Search Facts",
            placeholder="Enter keywords...",
        )
    
    # Noise toggle
    st.session_state.show_noise = st.checkbox(
        "🔇 Show Noise facts",
        value=st.session_state.show_noise,
        help="Noise facts are low-value or irrelevant. Hide them for cleaner analysis.",
    )
    
    return selected_source, selected_relevance, selected_believability, selected_chapter, search_query


def render_fact_table(
    selected_source: str,
    selected_relevance: str,
    selected_believability: str,
    selected_chapter: str,
    search_query: str,
):
    """Render the main fact table with editing support."""
    repo = st.session_state.repository
    
    # Get all facts
    facts = repo.get_all_facts()
    
    if not facts:
        st.info("📭 No facts in the database yet. Upload documents to get started!")
        return
    
    # Apply filters
    filtered_facts = facts
    
    # Noise filter (hide by default unless explicitly showing)
    if not st.session_state.show_noise:
        filtered_facts = [
            f for f in filtered_facts 
            if f.relevance != RelevanceLevel.NOISE.value
        ]
    
    # Source filter
    if selected_source != "All Documents":
        filtered_facts = [
            f for f in filtered_facts 
            if selected_source in f.source_documents
        ]
    
    # Relevance filter
    if selected_relevance != "all":
        filtered_facts = [
            f for f in filtered_facts 
            if f.relevance == selected_relevance
        ]
    
    # Believability filter
    if selected_believability != "all":
        filtered_facts = [
            f for f in filtered_facts 
            if f.believability == selected_believability
        ]
    
    # Chapter filter
    if selected_chapter != "All Chapters":
        filtered_facts = [
            f for f in filtered_facts 
            if f.chapter_relevance.get(selected_chapter, 0) >= 0.5
        ]
    
    # Search filter
    if search_query:
        search_lower = search_query.lower()
        filtered_facts = [
            f for f in filtered_facts 
            if search_lower in f.content.lower()
        ]
    
    # Display count
    noise_hidden = len([f for f in facts if f.relevance == RelevanceLevel.NOISE.value]) if not st.session_state.show_noise else 0
    hidden_msg = f" ({noise_hidden} noise facts hidden)" if noise_hidden > 0 else ""
    st.markdown(f"**Showing {len(filtered_facts)} of {len(facts)} facts{hidden_msg}**")
    st.caption("💡 You can edit Content, Believability, and Relevance directly in the table. Changes are logged for model improvement.")
    
    if not filtered_facts:
        st.warning("No facts match the current filters.")
        return
    
    # Display names for enums (bi-directional mapping)
    relevance_display = {
        "critical": "Critical",
        "chapter_specific": "Chapter Specific",
        "additional_context": "Additional Context",
        "noise": "Noise",
    }
    relevance_reverse = {v: k for k, v in relevance_display.items()}
    
    believability_display = {
        "verified": "Verified",
        "needs_verification": "Needs Verification",
        "opinion": "Opinion",
    }
    believability_reverse = {v: k for k, v in believability_display.items()}
    
    # Build a mapping of row index to fact ID for tracking edits
    fact_id_map = {i: fact.id for i, fact in enumerate(filtered_facts)}
    
    # Convert to DataFrame for display
    display_data = []
    for fact in filtered_facts:
        top_chapters = fact.get_top_chapters(3)
        top_chapters_str = ", ".join(
            f"{ch} ({score:.0%})" for ch, score in top_chapters if score > 0.3
        ) or "General"
        
        display_data.append({
            "ID": fact.id[:8],
            "Content": fact.content,  # Full content for editing
            "Source(s)": ", ".join(fact.source_documents),
            "Believability": believability_display.get(fact.believability, fact.believability),
            "Relevance": relevance_display.get(fact.relevance, fact.relevance),
            "Top Chapters": top_chapters_str,
            "Usage": fact.usage_count,
        })
    
    df = pd.DataFrame(display_data)
    
    # Create editable data editor
    edited_df = st.data_editor(
        df,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        disabled=["ID", "Source(s)", "Top Chapters", "Usage"],  # Read-only columns
        column_config={
            "ID": st.column_config.TextColumn("ID", width="small", disabled=True),
            "Content": st.column_config.TextColumn("Fact Content", width="large"),
            "Source(s)": st.column_config.TextColumn("Source", width="medium", disabled=True),
            "Believability": st.column_config.SelectboxColumn(
                "Believability",
                width="medium",
                options=["Verified", "Needs Verification", "Opinion"],
                required=True,
            ),
            "Relevance": st.column_config.SelectboxColumn(
                "Relevance",
                width="medium",
                options=["Critical", "Chapter Specific", "Additional Context", "Noise"],
                required=True,
            ),
            "Top Chapters": st.column_config.TextColumn("Relevant Chapters", width="medium", disabled=True),
            "Usage": st.column_config.NumberColumn("Uses", width="small", disabled=True),
        },
        key="fact_editor",
    )
    
    # Detect and apply changes
    if edited_df is not None:
        changes_made = 0
        for i in range(len(df)):
            fact_id = fact_id_map[i]
            original_row = df.iloc[i]
            edited_row = edited_df.iloc[i]
            
            updates = {}
            
            # Check Content
            if original_row["Content"] != edited_row["Content"]:
                updates["content"] = edited_row["Content"]
                # Log the correction
                correction = FactCorrection(
                    id=str(uuid.uuid4()),
                    fact_id=fact_id,
                    field_name="content",
                    old_value=original_row["Content"],
                    new_value=edited_row["Content"],
                )
                repo.insert_correction(correction)
                changes_made += 1
            
            # Check Believability
            if original_row["Believability"] != edited_row["Believability"]:
                new_believability = believability_reverse.get(edited_row["Believability"], "needs_verification")
                updates["believability"] = new_believability
                # Log the correction
                correction = FactCorrection(
                    id=str(uuid.uuid4()),
                    fact_id=fact_id,
                    field_name="believability",
                    old_value=believability_reverse.get(original_row["Believability"], original_row["Believability"]),
                    new_value=new_believability,
                )
                repo.insert_correction(correction)
                changes_made += 1
            
            # Check Relevance
            if original_row["Relevance"] != edited_row["Relevance"]:
                new_relevance = relevance_reverse.get(edited_row["Relevance"], "additional_context")
                updates["relevance"] = new_relevance
                # Log the correction
                correction = FactCorrection(
                    id=str(uuid.uuid4()),
                    fact_id=fact_id,
                    field_name="relevance",
                    old_value=relevance_reverse.get(original_row["Relevance"], original_row["Relevance"]),
                    new_value=new_relevance,
                )
                repo.insert_correction(correction)
                changes_made += 1
            
            # Apply updates to database
            if updates:
                repo.update_fact(fact_id, updates)
        
        if changes_made > 0:
            st.success(f"✅ Saved {changes_made} correction(s) and logged for model improvement!")
            st.rerun()
    
    st.divider()
    
    # Export options
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Export to CSV (excl. Noise)"):
            # Export without noise facts
            export_facts = [f for f in filtered_facts if f.relevance != RelevanceLevel.NOISE.value]
            export_data = [f.model_dump() for f in export_facts]
            export_df = pd.DataFrame(export_data)
            
            # Convert to CSV
            csv = export_df.to_csv(index=False)
            
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"fact_bank_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
    
    with col2:
        if st.button("📥 Export All (incl. Noise)"):
            # Full export with all fields including noise
            export_data = [f.model_dump() for f in filtered_facts]
            export_df = pd.DataFrame(export_data)
            
            # Convert to CSV
            csv = export_df.to_csv(index=False)
            
            st.download_button(
                label="Download Full CSV",
                data=csv,
                file_name=f"fact_bank_full_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )


def render_chapter_analysis():
    """Render chapter-based analysis."""
    repo = st.session_state.repository
    facts = repo.get_all_facts()
    
    if not facts:
        return
    
    st.markdown("### 📑 Chapter Analysis")
    
    chapter_stats = get_chapter_summary(facts)
    
    # Create summary table with new columns
    chapter_data = []
    for chapter, stats in chapter_stats.items():
        relevance = stats.get('relevance_breakdown', {})
        believability = stats.get('believability_breakdown', {})
        
        chapter_data.append({
            "Chapter": chapter,
            "Facts": stats['fact_count'],
            "🔴 Critical": relevance.get('critical', 0),
            "🔵 Chapter": relevance.get('chapter_specific', 0),
            "⚪ Context": relevance.get('additional_context', 0),
            "✅ Verified": believability.get('verified', 0),
            "⚠️ Needs Verif.": believability.get('needs_verification', 0),
            "💭 Opinion": believability.get('opinion', 0),
        })
    
    chapter_df = pd.DataFrame(chapter_data)
    
    st.dataframe(
        chapter_df,
        use_container_width=True,
        hide_index=True,
    )


async def run_conflict_detection(
    facts: list[Fact],
    relevance_threshold: float = 0.5,
    max_comparisons: int = 500,
) -> list[FactConflict]:
    """Run async conflict detection."""
    detector = ConflictDetector()
    try:
        conflicts = await detector.detect_conflicts(
            facts,
            relevance_threshold=relevance_threshold,
            max_comparisons=max_comparisons,
        )
        return conflicts
    finally:
        await detector.close()


def get_severity_color(severity: str) -> str:
    """Get color for conflict severity."""
    colors = {
        "critical": "#7f1d1d",
        "high": "#dc2626",
        "medium": "#f59e0b",
        "low": "#6b7280",
    }
    return colors.get(severity, "#6b7280")


def get_conflict_type_emoji(conflict_type: str) -> str:
    """Get emoji for conflict type."""
    emojis = {
        "numerical": "🔢",
        "temporal": "📅",
        "factual": "📋",
        "definitional": "📖",
        "strategic": "🎯",
    }
    return emojis.get(conflict_type, "⚠️")


def render_conflict_card(conflict: FactConflict, index: int):
    """Render a single conflict card."""
    severity = conflict.severity
    severity_color = get_severity_color(severity)
    type_emoji = get_conflict_type_emoji(conflict.conflict_type)
    
    with st.container():
        st.markdown(f"""
        <div class="conflict-card {severity}">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-weight: 600; font-size: 1.1rem;">
                    {type_emoji} Conflict #{index + 1}: {conflict.conflict_type.replace('_', ' ').title()}
                </span>
                <span style="background-color:{severity_color};color:white;padding:0.25rem 0.75rem;border-radius:9999px;font-size:0.75rem;font-weight:600;">
                    {severity.upper()}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Fact A:**")
            st.info(conflict.fact_a_content)
            st.caption(f"ID: `{conflict.fact_a_id[:8]}`")
        
        with col2:
            st.markdown("**Fact B:**")
            st.info(conflict.fact_b_content)
            st.caption(f"ID: `{conflict.fact_b_id[:8]}`")
        
        st.markdown("**Explanation:**")
        st.write(conflict.explanation)
        
        if conflict.resolution_suggestion:
            st.markdown("**💡 Resolution Suggestion:**")
            st.success(conflict.resolution_suggestion)
        
        if conflict.affected_chapters:
            st.markdown(f"**📑 Affected Chapters:** {', '.join(conflict.affected_chapters)}")
        
        st.divider()


def render_corrections_tab():
    """Render the corrections history tab."""
    repo = st.session_state.repository
    
    st.markdown("### 📝 Correction History")
    st.markdown("""
    This log tracks all corrections you've made to facts extracted by the LLM.
    These corrections can be used to analyze extraction patterns and improve model accuracy.
    """)
    
    # Get correction statistics
    stats = repo.get_correction_statistics()
    
    # Summary stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Corrections", stats["total_corrections"])
    with col2:
        st.metric("Recent (7 days)", stats["recent_corrections"])
    with col3:
        most_corrected = max(stats["by_field"].items(), key=lambda x: x[1])[0] if stats["by_field"] else "N/A"
        st.metric("Most Corrected Field", most_corrected.replace("_", " ").title() if most_corrected != "N/A" else "N/A")
    
    st.divider()
    
    # Corrections by field breakdown
    if stats["by_field"]:
        st.markdown("### Corrections by Field")
        field_df = pd.DataFrame([
            {"Field": k.replace("_", " ").title(), "Count": v}
            for k, v in stats["by_field"].items()
        ])
        st.bar_chart(field_df.set_index("Field"))
    
    st.divider()
    
    # Full corrections log
    corrections = repo.get_all_corrections()
    
    if not corrections:
        st.info("📭 No corrections logged yet. Edit facts in the Fact Bank tab to start tracking corrections.")
        return
    
    st.markdown(f"### Correction Log ({len(corrections)} entries)")
    
    # Convert to DataFrame for display
    display_data = []
    for c in corrections:
        display_data.append({
            "Timestamp": c.correction_timestamp.strftime("%Y-%m-%d %H:%M"),
            "Fact ID": c.fact_id[:8],
            "Field": c.field_name.replace("_", " ").title(),
            "Original Value": c.old_value[:100] + ("..." if len(c.old_value) > 100 else ""),
            "Corrected Value": c.new_value[:100] + ("..." if len(c.new_value) > 100 else ""),
        })
    
    corrections_df = pd.DataFrame(display_data)
    
    st.dataframe(
        corrections_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Timestamp": st.column_config.TextColumn("When", width="small"),
            "Fact ID": st.column_config.TextColumn("Fact", width="small"),
            "Field": st.column_config.TextColumn("Field", width="small"),
            "Original Value": st.column_config.TextColumn("Original", width="large"),
            "Corrected Value": st.column_config.TextColumn("Corrected", width="large"),
        },
    )
    
    # Export corrections for training
    st.divider()
    st.markdown("### Export for Model Training")
    st.markdown("""
    Export corrections in a format suitable for fine-tuning or prompt engineering.
    This data shows patterns where the LLM made mistakes that you corrected.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Export Corrections CSV"):
            export_data = []
            for c in corrections:
                # Get the original fact if it still exists
                fact = repo.get_fact(c.fact_id)
                export_data.append({
                    "correction_id": c.id,
                    "fact_id": c.fact_id,
                    "field_name": c.field_name,
                    "old_value": c.old_value,
                    "new_value": c.new_value,
                    "correction_timestamp": c.correction_timestamp.isoformat(),
                    "fact_content": fact.content if fact else "[deleted]",
                    "source_documents": ", ".join(fact.source_documents) if fact else "[deleted]",
                })
            
            export_df = pd.DataFrame(export_data)
            csv = export_df.to_csv(index=False)
            
            st.download_button(
                label="Download Corrections CSV",
                data=csv,
                file_name=f"fact_corrections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
    
    with col2:
        if st.button("🗑️ Clear Correction History", type="secondary"):
            repo.clear_all_corrections()
            st.success("Correction history cleared!")
            st.rerun()


def render_conflicts_tab():
    """Render the conflicts detection tab."""
    repo = st.session_state.repository
    facts = repo.get_all_facts()
    
    st.markdown("### ⚠️ Fact Conflict Detection")
    st.markdown("""
    This tool analyzes your fact bank to find contradictions and inconsistencies.
    Conflicts can indicate:
    - Outdated information that needs updating
    - Different sources with conflicting data
    - Errors in document extraction
    """)
    
    if len(facts) < 2:
        st.info("📭 Need at least 2 facts to check for conflicts. Upload more documents to get started!")
        return
    
    # Configuration
    col1, col2, col3 = st.columns(3)
    
    with col1:
        relevance_threshold = st.slider(
            "Relevance Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Minimum chapter relevance to consider facts in the same context",
        )
    
    with col2:
        max_comparisons = st.number_input(
            "Max Comparisons",
            min_value=10,
            max_value=2000,
            value=500,
            step=50,
            help="Maximum number of fact pairs to check (controls API cost)",
        )
    
    with col3:
        st.metric("Facts Available", len(facts))
    
    # Run detection button
    if st.button("🔍 Detect Conflicts", type="primary", use_container_width=True):
        with st.spinner("Analyzing facts for conflicts... This may take a moment."):
            loop = asyncio.new_event_loop()
            try:
                conflicts = loop.run_until_complete(
                    run_conflict_detection(
                        facts,
                        relevance_threshold=relevance_threshold,
                        max_comparisons=max_comparisons,
                    )
                )
                st.session_state.detected_conflicts = conflicts
                st.session_state.conflicts_checked = True
                
                if conflicts:
                    st.warning(f"⚠️ Found {len(conflicts)} potential conflicts!")
                else:
                    st.success("✅ No conflicts detected in your fact bank!")
                    
            except Exception as e:
                st.error(f"❌ Error detecting conflicts: {e}")
                logger.exception("Conflict detection error")
            finally:
                loop.close()
    
    st.divider()
    
    # Display detected conflicts
    conflicts = st.session_state.detected_conflicts
    
    if st.session_state.conflicts_checked:
        if conflicts:
            # Summary stats
            st.markdown(f"### Conflict Summary: {len(conflicts)} Issues Found")
            
            # Count by severity
            severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
            type_counts = {}
            
            for c in conflicts:
                severity_counts[c.severity] = severity_counts.get(c.severity, 0) + 1
                type_counts[c.conflict_type] = type_counts.get(c.conflict_type, 0) + 1
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🔴 Critical", severity_counts.get("critical", 0))
            with col2:
                st.metric("🟠 High", severity_counts.get("high", 0))
            with col3:
                st.metric("🟡 Medium", severity_counts.get("medium", 0))
            with col4:
                st.metric("⚪ Low", severity_counts.get("low", 0))
            
            # Filter options
            st.markdown("### Filter Conflicts")
            
            filter_col1, filter_col2 = st.columns(2)
            
            with filter_col1:
                severity_filter = st.multiselect(
                    "Severity",
                    options=["critical", "high", "medium", "low"],
                    default=["critical", "high", "medium"],
                    format_func=lambda x: x.upper(),
                )
            
            with filter_col2:
                type_filter = st.multiselect(
                    "Conflict Type",
                    options=list(type_counts.keys()),
                    default=list(type_counts.keys()),
                    format_func=lambda x: x.replace("_", " ").title(),
                )
            
            # Filter conflicts
            filtered_conflicts = [
                c for c in conflicts
                if c.severity in severity_filter and c.conflict_type in type_filter
            ]
            
            st.markdown(f"**Showing {len(filtered_conflicts)} of {len(conflicts)} conflicts**")
            st.divider()
            
            # Render conflict cards
            for i, conflict in enumerate(filtered_conflicts):
                render_conflict_card(conflict, i)
            
            # Export conflicts
            if st.button("📥 Export Conflicts Report"):
                export_data = []
                for c in conflicts:
                    export_data.append({
                        "Fact A ID": c.fact_a_id[:8],
                        "Fact A Content": c.fact_a_content,
                        "Fact B ID": c.fact_b_id[:8],
                        "Fact B Content": c.fact_b_content,
                        "Conflict Type": c.conflict_type,
                        "Severity": c.severity,
                        "Explanation": c.explanation,
                        "Resolution": c.resolution_suggestion,
                        "Affected Chapters": ", ".join(c.affected_chapters),
                    })
                
                export_df = pd.DataFrame(export_data)
                csv = export_df.to_csv(index=False)
                
                st.download_button(
                    label="Download Conflicts CSV",
                    data=csv,
                    file_name=f"fact_conflicts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
        else:
            st.markdown("### ✅ No Conflicts Detected")
            st.info("""
            Your fact bank appears consistent! No contradictions were found between facts 
            in the same chapter context.
            
            **Note:** This analysis checks facts that share chapter relevance. 
            Facts in completely different contexts are not compared.
            """)


def main():
    """Main application entry point."""
    render_header()
    
    # Sidebar
    uploaded_files = render_sidebar()
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Fact Bank", "📈 Analytics", "⚠️ Conflicts", "📝 Corrections"])
    
    with tab1:
        render_statistics()
        st.divider()
        
        selected_source, selected_relevance, selected_believability, selected_chapter, search_query = render_filters()
        st.divider()
        
        render_fact_table(selected_source, selected_relevance, selected_believability, selected_chapter, search_query)
    
    with tab2:
        render_statistics()
        st.divider()
        render_chapter_analysis()
    
    with tab3:
        render_conflicts_tab()
    
    with tab4:
        render_corrections_tab()


if __name__ == "__main__":
    main()

