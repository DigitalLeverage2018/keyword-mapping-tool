import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import openai
import tiktoken
from io import StringIO

# --- OpenAI Setup ---
st.title("🔍 Keyword Mapping Tool mit GPT-Scoring")
api_key = st.text_input("🔑 OpenAI API Key", type="password")
if not api_key:
    st.warning("Bitte gib deinen OpenAI API Key ein.")
    st.stop()
client = openai.OpenAI(api_key=api_key)

# --- Helper Functions ---
encoding = tiktoken.get_encoding("cl100k_base")

def get_embedding(text: str, model="text-embedding-3-small") -> list:
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text[:8000]], model=model)
    return response.data[0].embedding

def get_gpt_score(url, title, meta, content, keyword, serp_text):
    prompt = f"""
Du bist ein erfahrener SEO-Analyst. Deine Aufgabe ist es, die Relevanz eines Keywords für eine bestimmte Webseite einzuschätzen.

Hier sind die Details der Webseite:
URL: {url}
Title Tag: {title}
Meta Description: {meta}
Content: {content}

Das zu bewertende Keyword lautet: "{keyword}"

Zur Hilfe findest du hier die Top 3 Suchresultate zu diesem Keyword:
{serp_text}

Wie gut passt das Keyword zur oben beschriebenen Webseite? Nutze dazu dein eigenes Wissen bezüglich Keywords und Suchabsicht sowie die Meta Daten der Top-Ranking URLs.

Bewerte auf einer Skala von 1 bis 5 und gib nur eine Zahl als Antwort zurück.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except:
        return "ERROR"

# --- Upload Files ---
st.header("📁 CSV-Dateien hochladen")

def load_csv(label):
    uploaded_file = st.file_uploader(label, type="csv")
    if uploaded_file:
        return pd.read_csv(uploaded_file)
    else:
        st.stop()

website_df = load_csv("1. Website Content CSV")
keywords_df = load_csv("2. Keywords CSV")
gsc_df = load_csv("3. GSC-Daten CSV")
serp_df = load_csv("4. SERP-Daten CSV")

# --- Vorbereitung ---
website_df.columns = website_df.columns.str.lower()
keywords_df.columns = keywords_df.columns.str.lower()
gsc_df.columns = gsc_df.columns.str.lower()
serp_df.columns = serp_df.columns.str.lower()

# --- Embeddings ---
st.info("🔍 Erstelle Embeddings...")
website_df["embedding"] = website_df.apply(lambda row: get_embedding(
    f"{str(row[0])}\n{str(row[1])}\n{str(row[2])}\n{str(row[3])}"), axis=1)

keywords_df["combined"] = keywords_df.iloc[:, 0].astype(str) + " " + keywords_df.iloc[:, 1].astype(str)
keywords_df["embedding"] = keywords_df["combined"].apply(lambda x: get_embedding(x))

# --- Cosine Similarity ---
page_embeddings = np.vstack(website_df["embedding"].to_numpy())
keyword_embeddings = np.vstack(keywords_df["embedding"].to_numpy())
similarity_matrix = cosine_similarity(page_embeddings, keyword_embeddings)

# --- Keyword-Zuweisung ---
st.info("📊 Weise Keywords zu...")
keyword_assignment = {}
all_urls = website_df.iloc[:, 0].tolist()

for k_idx, keyword in enumerate(keywords_df.iloc[:, 0]):
    gsc_matches = gsc_df[(gsc_df['query'] == keyword) & (gsc_df['position'] <= 5)]
    gsc_matches = gsc_matches[gsc_matches['page'].isin(all_urls)]

    best_url = None
    if not gsc_matches.empty:
        top_pos = gsc_matches['position'].min()
        candidates = gsc_matches[gsc_matches['position'] == top_pos]
        if len(candidates) == 1:
            best_url = candidates.iloc[0]['page']
        else:
            sims = [
                (row['page'], similarity_matrix[website_df[website_df.iloc[:, 0] == row['page']].index[0], k_idx])
                for _, row in candidates.iterrows()
            ]
            best_url = max(sims, key=lambda x: x[1])[0]

        if best_url:
            url_idx = website_df[website_df.iloc[:, 0] == best_url].index[0]
            similarity = similarity_matrix[url_idx, k_idx]
            keyword_assignment[keyword] = {
                "url": best_url,
                "similarity": similarity,
                "ranking": top_pos,
                "clicks": gsc_matches[gsc_matches['page'] == best_url]['clicks'].values[0]
            }
            continue

    best_url_idx = np.argmax(similarity_matrix[:, k_idx])
    best_url = website_df.iloc[best_url_idx, 0]
    similarity = similarity_matrix[best_url_idx, k_idx]
    keyword_assignment[keyword] = {
        "url": best_url,
        "similarity": similarity,
        "ranking": None,
        "clicks": None
    }

# --- Finalisierung ---
url_to_keywords = {}
for kw, data in keyword_assignment.items():
    url = data["url"]
    url_to_keywords.setdefault(url, [])
    if len(url_to_keywords[url]) < 5:
        url_to_keywords[url].append({
            "url": url,
            "keyword": kw,
            "ranking": data["ranking"],
            "clicks": data["clicks"],
            "similarity": data["similarity"]
        })

final_df = pd.DataFrame([item for sublist in url_to_keywords.values() for item in sublist])

# --- GPT Scoring ---
st.info("🤖 Führe GPT-Scoring aus...")
gpt_scores, titles, metas = [], [], []
for _, row in final_df.iterrows():
    url = row["url"]
    keyword = row["keyword"]
    page_row = website_df[website_df.iloc[:, 0] == url]
    if page_row.empty:
        gpt_scores.append("N/A")
        titles.append("")
        metas.append("")
        continue
    title = str(page_row.iloc[0, 1])[:300]
    meta = str(page_row.iloc[0, 2])[:300]
    content = str(page_row.iloc[0, 3])[:1500]
    serp_text = ""
    for i in range(1, 4):
        t = serp_df.loc[serp_df["keyword"] == keyword, f"title_pos_{i}"].values
        m = serp_df.loc[serp_df["keyword"] == keyword, f"meta_description_pos_{i}"].values
        if len(t) > 0:
            serp_text += f"Resultat {i}:\nTitle: {t[0]}\nMeta: {m[0] if len(m) > 0 else ''}\n\n"
    score = get_gpt_score(url, title, meta, content, keyword, serp_text)
    gpt_scores.append(score)
    titles.append(title)
    metas.append(meta)

# --- Ergebnis-Tabelle ---
final_df["gpt_score"] = pd.to_numeric(gpt_scores, errors="coerce")
final_df["title"] = titles
final_df["meta"] = metas
final_df = final_df.sort_values(by=["url", "gpt_score", "similarity"], ascending=[True, False, False])

st.success("✅ Analyse abgeschlossen!")
st.dataframe(final_df)

# --- Download ---
csv = final_df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Ergebnis als CSV herunterladen", data=csv, file_name="keyword_mapping_final.csv", mime="text/csv")
