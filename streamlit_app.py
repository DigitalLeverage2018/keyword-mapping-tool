import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import tiktoken

# --- OpenAI API Key Eingabe ---
st.title("🔍 Automatisches Keyword-Mapping mit OpenAI & Embeddings")

api_key = st.text_input("🔑 OpenAI API Key", type="password")
if not api_key:
    st.warning("Bitte gib deinen OpenAI API Key ein, um fortzufahren.")
    st.stop()

client = OpenAI(api_key=api_key)

# --- Tokenizer Setup ---
encoding = tiktoken.get_encoding("cl100k_base")

def get_embedding(text: str, model="text-embedding-3-small") -> list:
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

# --- Datei-Uploads ---
st.header("📁 Dateien hochladen")

website_file = st.file_uploader("Website Content CSV", type=["csv"])
keywords_file = st.file_uploader("Keywords CSV", type=["csv"])
gsc_file = st.file_uploader("Google Search Console CSV", type=["csv"])
serp_file = st.file_uploader("Top 3 Suchresultate CSV", type=["csv"])

if not (website_file and keywords_file and gsc_file and serp_file):
    st.info("Bitte lade alle vier CSV-Dateien hoch, um den Prozess zu starten.")
    st.stop()

# CSV-Dateien laden
website_df = pd.read_csv(website_file)
keywords_df = pd.read_csv(keywords_file)
gsc_df = pd.read_csv(gsc_file)
serp_df = pd.read_csv(serp_file)

# Spaltennamen vereinheitlichen
website_df.columns = website_df.columns.str.lower()
keywords_df.columns = keywords_df.columns.str.lower()
gsc_df.columns = gsc_df.columns.str.lower()
serp_df.columns = serp_df.columns.str.lower()

# --- Embeddings für Seiten erstellen ---
st.info("🔍 Erstelle Embeddings für Seiten...")
@st.cache_data(show_spinner=False)
def create_page_embeddings(df):
    def combine_page_fields(row):
        url = str(row[0])
        title = str(row[1])
        meta = str(row[2])
        content = str(row[3])
        combined = f"{url}\n{title}\n{meta}\n{content}"
        return get_embedding(combined[:8000])  # sicherheitshalber kürzen
    return df.apply(combine_page_fields, axis=1)

with st.spinner("Seiten-Embeddings werden erstellt..."):
    website_df["embedding"] = create_page_embeddings(website_df)

# --- Embeddings für Keywords ---
st.info("🧠 Erstelle Embeddings für Keywords...")
@st.cache_data(show_spinner=False)
def create_keyword_embeddings(df):
    combined = df.iloc[:, 0].astype(str) + " " + df.iloc[:, 1].astype(str)
    return combined.apply(lambda x: get_embedding(x))

with st.spinner("Keyword-Embeddings werden erstellt..."):
    keywords_df["embedding"] = create_keyword_embeddings(keywords_df)

# --- Cosine Similarity berechnen ---
page_embeddings = np.vstack(website_df["embedding"].to_numpy())
keyword_embeddings = np.vstack(keywords_df["embedding"].to_numpy())
similarity_matrix = cosine_similarity(page_embeddings, keyword_embeddings)

# --- Keyword-Zuweisung ---
st.info("📊 Weise Keywords den besten URLs zu...")

keyword_assignment = {}
all_urls = website_df.iloc[:, 0].tolist()

progress_bar = st.progress(0)
total_keywords = len(keywords_df)
for k_idx, keyword in enumerate(keywords_df.iloc[:, 0]):
    gsc_matches = gsc_df[(gsc_df['query'] == keyword) & (gsc_df['position'] <= 5)]
    gsc_matches = gsc_matches[gsc_matches['page'].isin(all_urls)]

    if not gsc_matches.empty:
        gsc_matches = gsc_matches.sort_values(by=['position'])
        top_pos = gsc_matches['position'].min()
        candidates = gsc_matches[gsc_matches['position'] == top_pos]

        if len(candidates) == 1:
            best_url = candidates.iloc[0]['page']
        else:
            sims = [
                (row['page'], similarity_matrix[website_df[website_df.iloc[:, 0] == row['page']].index[0], k_idx])
                for _, row in candidates.iterrows()
                if row['page'] in all_urls
            ]
            best_url = max(sims, key=lambda x: x[1])[0] if sims else None

        if best_url and best_url in all_urls:
            url_idx = website_df[website_df.iloc[:, 0] == best_url].index[0]
            similarity = similarity_matrix[url_idx, k_idx]
            keyword_assignment[keyword] = {
                "url_idx": url_idx,
                "url": best_url,
                "similarity": similarity,
                "ranking": top_pos,
                "clicks": gsc_matches[gsc_matches['page'] == best_url]['clicks'].values[0]
            }
            progress_bar.progress((k_idx + 1) / total_keywords)
            continue

    best_url_idx = np.argmax(similarity_matrix[:, k_idx])
    best_url = website_df.iloc[best_url_idx, 0]
    similarity = similarity_matrix[best_url_idx, k_idx]
    keyword_assignment[keyword] = {
        "url_idx": best_url_idx,
        "url": best_url,
        "similarity": similarity,
        "ranking": None,
        "clicks": None
    }
    progress_bar.progress((k_idx + 1) / total_keywords)

# --- Maximal 5 Keywords pro URL ---
url_to_keywords = {}
for kw, data in keyword_assignment.items():
    url = data["url"]
    if url not in url_to_keywords:
        url_to_keywords[url] = []
    if len(url_to_keywords[url]) < 5:
        url_to_keywords[url].append({
            "url": url,
            "keyword": kw,
            "ranking": data["ranking"],
            "clicks": data["clicks"],
            "similarity": data["similarity"]
        })

# --- Ergebnis-Tabelle ---
final_df = pd.DataFrame([item for sublist in url_to_keywords.values() for item in sublist])

# --- SERP URLs ausgeben ---
def get_serp_urls(keyword):
    row = serp_df[serp_df["keyword"] == keyword]
    if row.empty:
        return "", "", ""
    row = row.iloc[0]
    return (
        row.get("url_pos_1", ""),
        row.get("url_pos_2", ""),
        row.get("url_pos_3", "")
    )

# --- GPT Scoring Funktion ---
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
    except Exception as e:
        return "ERROR"

# --- GPT-Scoring ---
st.info("🤖 GPT-Scoring & SERP-Zusatz wird ausgeführt... (kann einige Zeit dauern)")

gpt_scores = []
titles = []
metas = []
serp1_list = []
serp2_list = []
serp3_list = []

progress_bar_gpt = st.progress(0)
total_final = len(final_df)

for i, row in final_df.iterrows():
    url = row["url"]
    keyword = row["keyword"]
    page_row = website_df[website_df.iloc[:, 0] == url]
    if page_row.empty:
        gpt_scores.append("N/A")
        titles.append("")
        metas.append("")
        serp1_list.append("")
        serp2_list.append("")
        serp3_list.append("")
        progress_bar_gpt.progress((i + 1) / total_final)
        continue

    title = str(page_row.iloc[0, 1])[:300]
    meta = str(page_row.iloc[0, 2])[:300]
    content = str(page_row.iloc[0, 3])[:1500]
    serp_text = ""
    for pos in range(1, 4):
        title_pos = serp_df.loc[serp_df["keyword"] == keyword, f"title_pos_{pos}"].values
        meta_pos = serp_df.loc[serp_df["keyword"] == keyword, f"meta_description_pos_{pos}"].values
        if len(title_pos) > 0:
            serp_text += f"Resultat {pos}:\nTitle: {title_pos[0]}\nMeta: {meta_pos[0] if len(meta_pos) > 0 else ''}\n\n"

    score = get_gpt_score(url, title, meta, content, keyword, serp_text)
    gpt_scores.append(score)
    titles.append(title)
    metas.append(meta)
    s1, s2, s3 = get_serp_urls(keyword)
    serp1_list.append(s1)
    serp2_list.append(s2)
    serp3_list.append(s3)
    progress_bar_gpt.progress((i + 1) / total_final)

# --- Ergebnisse ergänzen ---
final_df["gpt_score"] = pd.to_numeric(gpt_scores, errors="coerce")
final_df["title"] = titles
final_df["meta"] = metas
final_df["serp_url_1"] = serp1_list
final_df["serp_url_2"] = serp2_list
final_df["serp_url_3"] = serp3_list

# Spalten anordnen: gpt_score direkt nach similarity
cols = final_df.columns.tolist()
if "similarity" in cols and "gpt_score" in cols:
    sim_idx = cols.index("similarity")
    cols.remove("gpt_score")
    cols.insert(sim_idx + 1, "gpt_score")
    final_df = final_df[cols]

# Sortierung
final_df = final_df.sort_values(by=["url", "gpt_score", "similarity"], ascending=[True, False, False])

# --- Ergebnis anzeigen und Download ---
st.header("📊 Ergebnis")
st.dataframe(final_df)

csv = final_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="CSV mit Keyword-Mapping herunterladen",
    data=csv,
    file_name="keyword_mapping_final.csv",
    mime="text/csv"
)
