# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken
import openai

# --- Setup ---
st.set_page_config(page_title="🔑 Keyword Mapping Tool", layout="wide")
st.title("🔑 Keyword Mapping Tool")

st.markdown("""
Dieses Tool kombiniert Website-Inhalte, relevante Keywords und GSC-Daten, um automatisch ein optimiertes Keyword-Mapping vorzuschlagen.
""")

# --- OpenAI API Key ---
api_key = st.text_input("🔐 OpenAI API Key", type="password")
if not api_key:
    st.stop()
client = openai.OpenAI(api_key=api_key)

# --- Upload Files ---
st.subheader("📁 Dateien hochladen")
website_file = st.file_uploader("Website Content (CSV)", type="csv")
keywords_file = st.file_uploader("Keywords + Cluster (CSV)", type="csv")
gsc_file = st.file_uploader("GSC Export (CSV)", type="csv")
serp_file = st.file_uploader("Top 3 SERPs (CSV)", type="csv")

if not all([website_file, keywords_file, gsc_file, serp_file]):
    st.info("Bitte alle vier Dateien hochladen.")
    st.stop()

# --- Read Data ---
website_df = pd.read_csv(website_file)
keywords_df = pd.read_csv(keywords_file)
gsc_df = pd.read_csv(gsc_file)
serp_df = pd.read_csv(serp_file)

# --- Normalize Column Names ---
for df in [website_df, keywords_df, gsc_df, serp_df]:
    df.columns = df.columns.str.lower()

# --- Embedding Functions ---
encoding = tiktoken.get_encoding("cl100k_base")
def get_embedding(text: str, model="text-embedding-3-small") -> list:
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text[:8000]], model=model)
    return response.data[0].embedding

st.info("🔄 Berechne Embeddings. Dies kann einige Minuten dauern...")

# --- Create Page Embeddings ---
website_df["embedding"] = website_df.apply(
    lambda row: get_embedding(f"{row[0]}\n{row[1]}\n{row[2]}\n{row[3]}"), axis=1
)

# --- Create Keyword Embeddings ---
keywords_df["combined"] = keywords_df.iloc[:, 0].astype(str) + " " + keywords_df.iloc[:, 1].astype(str)
keywords_df["embedding"] = keywords_df["combined"].apply(lambda x: get_embedding(x))

# --- Cosine Similarity ---
page_embeddings = np.vstack(website_df["embedding"].to_numpy())
keyword_embeddings = np.vstack(keywords_df["embedding"].to_numpy())
similarity_matrix = cosine_similarity(page_embeddings, keyword_embeddings)

# --- Assign Keywords ---
st.info("📊 Weise Keywords zu URLs zu...")
final_rows = []
all_urls = website_df.iloc[:, 0].tolist()

for k_idx, keyword in enumerate(keywords_df.iloc[:, 0]):
    gsc_matches = gsc_df[(gsc_df['query'] == keyword) & (gsc_df['position'] <= 5)]
    gsc_matches = gsc_matches[gsc_matches['page'].isin(all_urls)]

    if not gsc_matches.empty:
        gsc_matches = gsc_matches.sort_values(by=['position'])
        top_pos = gsc_matches['position'].min()
        candidates = gsc_matches[gsc_matches['position'] == top_pos]
        best_url = candidates.iloc[0]['page'] if len(candidates) == 1 else max([
            (row['page'], similarity_matrix[website_df[website_df.iloc[:, 0] == row['page']].index[0], k_idx])
            for _, row in candidates.iterrows()
        ], key=lambda x: x[1])[0]
        url_idx = website_df[website_df.iloc[:, 0] == best_url].index[0]
    else:
        url_idx = np.argmax(similarity_matrix[:, k_idx])
        best_url = website_df.iloc[url_idx, 0]
        top_pos = None

    similarity = similarity_matrix[url_idx, k_idx]
    clicks = gsc_matches[gsc_matches['page'] == best_url]['clicks'].values[0] if not gsc_matches.empty else None

    final_rows.append({
        "url": best_url,
        "keyword": keyword,
        "similarity": similarity,
        "ranking": top_pos,
        "clicks": clicks
    })

# --- Final DataFrame ---
final_df = pd.DataFrame(final_rows)
st.success("✅ Mapping abgeschlossen")
st.dataframe(final_df)

csv = final_df.to_csv(index=False).encode("utf-8")
st.download_button("📥 CSV herunterladen", csv, "keyword_mapping.csv", "text/csv")
