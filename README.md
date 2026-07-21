# 🤖 AI Tool Recommendation System

A content-based recommendation engine that suggests similar AI tools based on
their descriptions, built with **TF-IDF**, **cosine similarity**, and a
**Streamlit** front end.

> Discover similar AI tools using Machine Learning.

---

## 📖 Project Overview

This project recommends AI tools similar to one you already know or use.
Given a tool name, it returns the top 5 most similar tools based on the
textual similarity of their descriptions.

The project is split into two clear parts:

1. **Machine Learning (Jupyter Notebook)** — data cleaning, Porter stemming,
   TF-IDF vectorization, and cosine similarity computation. The trained
   artifacts are exported as `tools.pkl` and `similarity.pkl`.
2. **Application (Streamlit)** — `app.py` loads those two artifacts and
   serves an interactive web UI. It does **not** retrain or reprocess any
   data — it purely consumes the precomputed model.

---

## ✨ Features

- 🔍 Searchable dropdown of all AI tools in the dataset
- ✨ One-click "Recommend Similar Tools" button
- 📊 Similarity percentage shown per recommendation
- 📝 Tool description shown per recommendation
- 🔗 Website link shown automatically if a website column is present
  (gracefully shows "Website not available" otherwise)
- ⏳ Loading spinner while recommendations are generated
- ⚠️ Friendly error handling for missing/corrupted files, invalid
  selections, and shape mismatches
- 📚 Sidebar with project, dataset, and developer information
- 🎨 Modern, responsive, dark-mode-compatible UI with custom CSS cards
- 🧩 Modular design — new metadata columns (Website, Category, Pricing,
  Logo, etc.) added to `tools.pkl` in the future will automatically appear
  in the UI without any code changes

---

## 🗂️ Dataset

- **Source:** FutureTools.io AI tool listings
- **Current columns used:**
  - `Tool Name` — the name of the AI tool
  - `tag` — a cleaned, stemmed text description used for similarity
- The dataset is preprocessed and vectorized entirely in the notebook; the
  Streamlit app only reads the final `tools.pkl` / `similarity.pkl` outputs.

---

## 🛠️ Technology Stack

| Layer            | Tools                              |
|-------------------|-------------------------------------|
| Language          | Python 3.10+                       |
| Data processing   | Pandas                             |
| NLP               | NLTK (Porter Stemmer)              |
| ML / Vectorization| Scikit-Learn (TF-IDF, cosine similarity) |
| Frontend          | Streamlit                          |
| Model storage     | Pickle (`.pkl`)                    |

---

## 📁 Project Structure

```
ai-tool-recommender/
├── app.py                              # Streamlit application
├── tools.pkl                           # Cleaned dataframe (from notebook)
├── similarity.pkl                      # Cosine similarity matrix (from notebook)
├── ai_tool_recommendation.ipynb        # ML notebook (data prep + model)
├── requirements.txt                    # Python dependencies
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/ai-tool-recommender.git
cd ai-tool-recommender
```

### 2. Create and activate a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Make sure the model files are present
Place `tools.pkl` and `similarity.pkl` (generated from your notebook) in the
same folder as `app.py`.

---

## ▶️ How to Run

```bash
streamlit run app.py
```

Then open the URL shown in your terminal (usually `http://localhost:8501`).

---

## 📸 Screenshots

> _Add screenshots of the running app here._

```
[Landing page screenshot]
[Recommendation results screenshot]
[Sidebar screenshot]
```

---

## 🚀 Future Improvements

- Add `Website`, `Category`, `Pricing`, and `Logo URL` columns to `tools.pkl`
  (the UI is already built to display these automatically)
- Search suggestions while typing
- "Read more" expandable descriptions
- Download recommendations as CSV
- Recently searched / favorite tools (session-based)
- Deploy to Streamlit Community Cloud

---

## 📄 License

This project is licensed under the MIT License — feel free to use and modify it.

---

## 👤 Author

Built by **Your Name**
[GitHub](https://github.com) · [LinkedIn](https://linkedin.com)
