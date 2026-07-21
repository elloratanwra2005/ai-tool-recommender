"""
AI Tool Recommendation System
==============================
A Streamlit front-end that consumes a PRE-TRAINED recommendation engine.

IMPORTANT:
This file does NOT perform any data cleaning, stemming, vectorization,
or similarity computation. All of that already happened in the notebook
and was persisted to:

    - tools.pkl       -> pandas.DataFrame with columns ["Tool Name", "tag"]
    - similarity.pkl  -> numpy.ndarray, shape (n_tools, n_tools)

This app only LOADS those two artifacts and builds a UI + a thin
integration function around them.

Author: <Your Name>
"""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------------------------------
# This must be the first Streamlit command in the script.
st.set_page_config(
    page_title="AI Tool Recommendation System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# 2. CONSTANTS
# --------------------------------------------------------------------------
# Centralizing file paths and config values makes the app easy to adapt
# later (e.g. if you rename files or move them into a /data folder).
TOOLS_PKL_PATH = Path("tools.pkl")
SIMILARITY_PKL_PATH = Path("similarity.pkl")
TOP_N_RECOMMENDATIONS = 5

# Columns your dataframe is GUARANTEED to have today.
TOOL_NAME_COL = "Tool Name"
DESCRIPTION_COL = "tag"

# Columns that DON'T exist yet but might be added later (Website, Category,
# Pricing, Logo URL, Developer, Launch Date, etc.). The UI checks for these
# dynamically instead of hard-coding them, so the day you add a "Website"
# column to tools.pkl, the card will automatically show a real link instead
# of "Website not available" -- with zero code changes required here.
OPTIONAL_COLUMNS = {
    "website": ["Website", "website", "Link", "URL", "url"],
    "category": ["Category", "category", "Tags", "Type"],
    "pricing": ["Pricing", "pricing", "Price"],
    "logo": ["Logo URL", "Logo", "logo_url", "Image"],
}


# --------------------------------------------------------------------------
# 3. DATA LOADING (with caching + friendly error handling)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """
    Load tools.pkl and similarity.pkl from disk.

    st.cache_resource ensures these large objects are loaded from disk only
    ONCE per app session, not on every user interaction/rerun. Streamlit
    reruns the whole script top-to-bottom on every button click, so without
    caching we'd be re-reading the pickle files constantly.

    Returns
    -------
    (pd.DataFrame, np.ndarray)

    Raises
    ------
    FileNotFoundError, pickle.UnpicklingError, ValueError
        These are caught by the caller (main()) and turned into friendly
        Streamlit error messages -- we don't swallow them here so the
        caller has full context on what went wrong.
    """
    if not TOOLS_PKL_PATH.exists():
        raise FileNotFoundError(f"'{TOOLS_PKL_PATH}' not found.")
    if not SIMILARITY_PKL_PATH.exists():
        raise FileNotFoundError(f"'{SIMILARITY_PKL_PATH}' not found.")

    with open(TOOLS_PKL_PATH, "rb") as f:
        tools_df = pickle.load(f)

    with open(SIMILARITY_PKL_PATH, "rb") as f:
        similarity_matrix = pickle.load(f)

    # --- Sanity checks (fail loudly & clearly rather than crashing later) ---
    if not isinstance(tools_df, pd.DataFrame):
        raise ValueError("tools.pkl did not contain a pandas DataFrame.")

    if tools_df.empty:
        raise ValueError("tools.pkl loaded successfully but the DataFrame is empty.")

    if TOOL_NAME_COL not in tools_df.columns:
        raise ValueError(
            f"Expected column '{TOOL_NAME_COL}' not found in tools.pkl. "
            f"Available columns: {list(tools_df.columns)}"
        )

    if similarity_matrix.ndim != 2 or similarity_matrix.shape[0] != similarity_matrix.shape[1]:
        raise ValueError("similarity.pkl is not a square matrix as expected.")

    if similarity_matrix.shape[0] != len(tools_df):
        raise ValueError(
            "Shape mismatch: similarity matrix has "
            f"{similarity_matrix.shape[0]} rows but tools.pkl has {len(tools_df)} rows. "
            "These two files must have been generated together from the same "
            "notebook run."
        )

    # Reset index so DataFrame row position always matches similarity matrix
    # row/column position (i.e. tools_df.iloc[i] <-> similarity_matrix[i]).
    tools_df = tools_df.reset_index(drop=True)

    return tools_df, similarity_matrix


def get_optional_field(row: pd.Series, field_key: str):
    """
    Look up an optional field (website, category, pricing, logo) on a
    dataframe row using a list of possible column-name aliases.

    This is the piece of "future-proofing" you asked for: if tomorrow you
    add a "Website" column to tools.pkl, this function will find it
    automatically -- no changes needed to the card-rendering code below.

    Returns None if none of the known aliases exist in the dataframe.
    """
    for column_name in OPTIONAL_COLUMNS[field_key]:
        if column_name in row.index:
            value = row[column_name]
            if pd.notna(value) and str(value).strip():
                return str(value)
    return None


# --------------------------------------------------------------------------
# 4. RECOMMENDATION INTEGRATION LAYER
# --------------------------------------------------------------------------
# NOTE ON INTEGRATION:
# Your notebook's recommend() function likely just returns a list of tool
# NAMES. This app also needs the numeric SIMILARITY SCORE for each result
# (to show a percentage) and the row itself (to show the description).
# So this wraps the same lookup logic your notebook uses -- same matrix,
# same index-based lookup, same cosine similarity values -- and additionally
# returns the score and full row. No re-vectorization, no retraining,
# just reading values that already exist in similarity.pkl.
def recommend_tools(tool_name: str, tools_df: pd.DataFrame, similarity_matrix, top_n: int = 5):
    """
    Given a tool name, return the top_n most similar tools.

    Parameters
    ----------
    tool_name : str
        The tool selected by the user in the dropdown.
    tools_df : pd.DataFrame
        The loaded tools dataframe (Tool Name, tag, ...).
    similarity_matrix : np.ndarray
        The precomputed cosine similarity matrix.
    top_n : int
        How many recommendations to return.

    Returns
    -------
    list[dict]
        Each dict has keys: name, score (0-100 float), row (pd.Series).

    Raises
    ------
    ValueError
        If the tool_name is not found in the dataframe.
    """
    matches = tools_df.index[tools_df[TOOL_NAME_COL] == tool_name].tolist()
    if not matches:
        raise ValueError(f"'{tool_name}' was not found in the dataset.")

    tool_index = matches[0]

    # distances: list of (row_index, similarity_score) for every tool
    distances = list(enumerate(similarity_matrix[tool_index]))

    # Sort by similarity score, descending. Skip index 0 after sorting
    # because the most similar tool to itself is always itself (score 1.0).
    sorted_distances = sorted(distances, key=lambda x: x[1], reverse=True)
    top_matches = [item for item in sorted_distances if item[0] != tool_index][:top_n]

    recommendations = []
    for row_index, score in top_matches:
        row = tools_df.iloc[row_index]
        recommendations.append(
            {
                "name": row[TOOL_NAME_COL],
                "score": round(float(score) * 100, 1),  # convert to %
                "row": row,
            }
        )

    return recommendations


# --------------------------------------------------------------------------
# 5. CUSTOM CSS (cards, hover effects, spacing, dark-mode friendly)
# --------------------------------------------------------------------------
def inject_custom_css():
    st.markdown(
        """
        <style>
        /* Import a clean modern font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Hero header */
        .hero-title {
            font-size: 2.6rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .hero-subtitle {
            font-size: 1.1rem;
            opacity: 0.75;
            margin-bottom: 1.5rem;
        }

        /* Recommendation card */
        .tool-card {
            border-radius: 16px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1rem;
            border: 1px solid rgba(128, 128, 128, 0.25);
            background: rgba(128, 128, 128, 0.06);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .tool-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
            border-color: rgba(99, 102, 241, 0.5);
        }
        .tool-card h4 {
            margin: 0 0 0.4rem 0;
            font-size: 1.15rem;
            font-weight: 700;
        }
        .tool-meta {
            font-size: 0.85rem;
            opacity: 0.7;
            margin-bottom: 0.5rem;
        }
        .tool-description {
            font-size: 0.92rem;
            line-height: 1.5;
            opacity: 0.9;
        }
        .badge {
            display: inline-block;
            padding: 0.15rem 0.6rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            background: rgba(99, 102, 241, 0.15);
            color: #6366f1;
            margin-right: 0.4rem;
        }

        /* Footer */
        .footer {
            text-align: center;
            padding: 1.5rem 0 0.5rem 0;
            margin-top: 2rem;
            border-top: 1px solid rgba(128, 128, 128, 0.2);
            font-size: 0.85rem;
            opacity: 0.7;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# 6. SIDEBAR
# --------------------------------------------------------------------------
def render_sidebar(tools_df: pd.DataFrame):
    with st.sidebar:
        st.markdown("## 🤖 About This Project")
        st.write(
            "An AI Tool Recommendation System that suggests similar AI "
            "tools using **content-based filtering** with **TF-IDF** and "
            "**cosine similarity**."
        )

        st.markdown("---")
        st.markdown("### 🛠️ Technology Stack")
        st.markdown(
            "- Python\n"
            "- Pandas\n"
            "- Scikit-Learn\n"
            "- NLTK (Porter Stemmer)\n"
            "- Streamlit"
        )

        st.markdown("---")
        st.markdown("### 📊 Dataset Information")
        st.write("Source: FutureTools.io tool listings")
        st.metric("Total AI Tools", f"{len(tools_df):,}")

        st.markdown("---")
        st.markdown("### 👤 Developer")
        st.write("Built by **Ellora Tanwar**")
        st.write("[GitHub](https://github.com/elloratanwra2005) · [LinkedIn](https://www.linkedin.com/in/ellora-tanwar-115599289/)")


# --------------------------------------------------------------------------
# 7. RECOMMENDATION CARD RENDERING
# --------------------------------------------------------------------------
def render_recommendation_card(recommendation: dict):
    row = recommendation["row"]
    name = recommendation["name"]
    score = recommendation["score"]

    description = row.get(DESCRIPTION_COL, "")
    if not description or pd.isna(description):
        description = "No description available."

    website = get_optional_field(row, "website")
    category = get_optional_field(row, "category")

    website_html = (
        f'<a href="{website}" target="_blank">🔗 Visit Website</a>'
        if website
        else '<span style="opacity:0.6;">🔗 Website not available</span>'
    )
    category_html = (
        f'<span class="badge">{category}</span>'
        if category
        else '<span class="badge" style="opacity:0.6;">Category unavailable</span>'
    )

    st.markdown(
        f"""
        <div class="tool-card">
            <h4>{name}</h4>
            <div class="tool-meta">{category_html} {website_html}</div>
            <div class="tool-description">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(min(int(score), 100), text=f"Similarity: {score}%")


# --------------------------------------------------------------------------
# 8. FOOTER
# --------------------------------------------------------------------------
def render_footer():
    st.markdown(
        """
        <div class="footer">
            Built with ❤️ using <b>Python</b> · <b>Scikit-Learn</b> ·
            <b>NLTK</b> · <b>TF-IDF</b> · <b>Streamlit</b><br>
            © 2026 AI Tool Recommendation System
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# 9. MAIN APP
# --------------------------------------------------------------------------
def main():
    inject_custom_css()

    # ---- Header ----
    st.markdown('<div class="hero-title">🤖 AI Tool Recommendation System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Discover similar AI tools using Machine Learning.</div>',
        unsafe_allow_html=True,
    )

    # ---- Load artifacts (with friendly error handling) ----
    try:
        with st.spinner("Loading recommendation engine..."):
            tools_df, similarity_matrix = load_artifacts()
    except FileNotFoundError as e:
        st.error(
            f"❌ Could not find a required file: {e}\n\n"
            "Make sure `tools.pkl` and `similarity.pkl` are in the same "
            "folder as `app.py`."
        )
        st.stop()
    except (pickle.UnpicklingError, EOFError):
        st.error(
            "❌ One of the pickle files appears to be corrupted and could "
            "not be loaded. Try regenerating tools.pkl / similarity.pkl "
            "from your notebook."
        )
        st.stop()
    except ValueError as e:
        st.error(f"❌ Data validation failed: {e}")
        st.stop()
    except Exception as e:  # noqa: BLE001 - final safety net for unexpected errors
        st.error(f"❌ An unexpected error occurred while loading data: {e}")
        st.stop()

    render_sidebar(tools_df)

    # ---- Searchable dropdown ----
    tool_names = sorted(tools_df[TOOL_NAME_COL].dropna().unique().tolist())
    selected_tool = st.selectbox(
        "🔍 Search and select an AI tool",
        options=tool_names,
        index=None,
        placeholder="Start typing a tool name...",
    )

    # ---- Recommend button ----
    recommend_clicked = st.button("✨ Recommend Similar Tools", type="primary", use_container_width=False)

    if recommend_clicked:
        if not selected_tool:
            st.warning("⚠️ Please select a tool before requesting recommendations.")
        else:
            try:
                with st.spinner("Generating recommendations..."):
                    recommendations = recommend_tools(
                        selected_tool, tools_df, similarity_matrix, TOP_N_RECOMMENDATIONS
                    )

                if not recommendations:
                    st.info("No similar tools were found for this selection.")
                else:
                    st.success(f"✅ Top {len(recommendations)} recommendations generated successfully.")
                    st.markdown(f"#### Tools similar to **{selected_tool}**")

                    cols = st.columns(2)
                    for i, rec in enumerate(recommendations):
                        with cols[i % 2]:
                            render_recommendation_card(rec)

            except ValueError as e:
                st.error(f"❌ {e}")
            except Exception as e:  # noqa: BLE001
                st.error(f"❌ Something went wrong while generating recommendations: {e}")

    render_footer()


if __name__ == "__main__":
    main()
