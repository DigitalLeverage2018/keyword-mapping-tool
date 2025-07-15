import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import openai
import tiktoken
from io import StringIO

# --- OpenAI API Key ---
st.subheader("🔑 OpenAI API Key")
api_key = st.text_input("Bitte gib deinen API Key ein", type="password")
st.markdown("[💡 API-Key generieren](https://platform.openai.com/api-keys)", unsafe_allow_html=True)

if not api_key:
    st.warning("Bitte gib deinen OpenAI API Key ein.")
    st.stop()
client = openai.OpenAI(api_key=api_key)


# --- Beschreibung & Anleitung ---
st.header("📁 CSV-Dateien hochladen")

with st.expander("ℹ️ Anleitung & Beispieldateien anzeigen"):
    st.markdown("""
Damit das Tool funktioniert, braucht ihr folgende vier CSV-Dateien:

1. **[URLs & Content](https://docs.google.com/spreadsheets/d/1uvKWUdmiQYrc76CJLoFkJTmn_WxpKpcIeNs26XCKhOc/edit?gid=1408231942)**  
   ➤ CSV mit `URL`, `Title Tag`, `Meta Description`, `Content`  
   ➤ Content muss zuerst mit Screaming Frog extrahiert werden

2. **[Keywords & Cluster](https://docs.google.com/spreadsheets/d/1uvKWUdmiQYrc76CJLoFkJTmn_WxpKpcIeNs26XCKhOc/edit?gid=1581107905)**  
   ➤ Spalte 1 = Hauptkeyword  
   ➤ Spalte 2 = Cluster-Keyword

3. **[Google Search Console Report (non-brand)](https://docs.google.com/spreadsheets/d/1uvKWUdmiQYrc76CJLoFkJTmn_WxpKpcIeNs26XCKhOc/edit?gid=1574634156)**  
   ➤ Spalten: `query`, `page`, `clicks`, `impressions`, `ctr`, `position`  
   ➤ Nur non-brand Keywords verwenden

4. **[SERP-OnPage-Daten](https://docs.google.com/spreadsheets/d/1uvKWUdmiQYrc76CJLoFkJTmn_WxpKpcIeNs26XCKhOc/edit?gid=2138481856)**  
   ➤ Suchresultate von allen Hauptkeywords mit dem [AirOps Tool](https://app.airops.com/digital-leverage-1/workflows/94174/run/once) ziehen.
   ➤ Danach [diese Tabelle](https://app.airops.com/digital-leverage-1/grids/24678/sheets/31380) als CSV exportieren  
   ➤ Wichtig: Zuerst immer alle alten URLs aus Tabelle löschen.
""")

# --- Upload CSVs ---
def load_csv(label):
    uploaded_file = st.file_uploader(label, type="csv")
    if uploaded_file:
        return pd.read_csv(uploaded_file)
    else:
        st.stop()

website_df = load_csv("1️⃣ URLs & Content (Screaming Frog Export)")
keywords_df = load_csv("2️⃣ Keywords & Cluster (2-Spalten-CSV)")
gsc_df = load_csv("3️⃣ GSC Report (query, page, clicks, etc.)")
serp_df = load_csv("4️⃣ SERP-OnPage-Daten (AirOps Export)")

# --- Preprocessing ---
website_df.columns = website_df.columns.str.lower()
keywords_df.columns = keywords_df.columns.str.lower()
gsc_df.columns = gsc_df.columns.str.lower()
serp_df.columns = serp_df.columns.str.lower()

encoding = tiktoken.get_encoding("cl100k_base")

def get_embedding(text: str, model="text-embedding-3-small") -> list:
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text[:8000]], model=model)
    return response.data[0].embedding

# --- Embeddings erstellen ---
st.info("🔍 Erstelle Embeddings für Seiten...")
website_df["embedding"] = website_df.apply(lambda row: get_embedding(
    f"{str(row[0])}\n{str(row[1])}\n{str(row[2])}\n{str(row[3])}"), axis=1)

st.info("🧠 Erstelle Embeddings für Keywords + Cluster...")
keywords_df["combined"] = keywords_df.iloc[:, 0].astype(str) + " " + keywords_df.iloc[:, 1].astype(str)
keywords_df["embedding"] = keywords_df["combined"].apply(lambda x: get_embedding(x))

# --- Similarity-Berechnung ---
page_embeddings = np.vstack(website_df["embedding"].to_numpy())
keyword_embeddings = np.vstack(keywords_df["embedding"].to_numpy())
similarity_matrix = cosine_similarity(page_embeddings, keyword_embeddings)

# --- Keyword-Zuweisung ---
st.info("📊 Weise Keywords den besten URLs zu...")
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

# --- Max. 5 Keywords pro URL ---
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

# --- SERP & GPT-Scoring ---
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

# --- GPT-Analyse ---
st.info("🤖 Führe GPT-Scoring durch...")
gpt_scores, titles, metas, serp1_list, serp2_list, serp3_list = [], [], [], [], [], []

for _, row in final_df.iterrows():
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
    s1 = serp_df.loc[serp_df["keyword"] == keyword, "url_pos_1"].values[0] if keyword in serp_df["keyword"].values else ""
    s2 = serp_df.loc[serp_df["keyword"] == keyword, "url_pos_2"].values[0] if keyword in serp_df["keyword"].values else ""
    s3 = serp_df.loc[serp_df["keyword"] == keyword, "url_pos_3"].values[0] if keyword in serp_df["keyword"].values else ""
    serp1_list.append(s1)
    serp2_list.append(s2)
    serp3_list.append(s3)

final_df["gpt_score"] = pd.to_numeric(gpt_scores, errors="coerce")
final_df["title"] = titles
final_df["meta"] = metas
final_df["serp_url_1"] = serp1_list
final_df["serp_url_2"] = serp2_list
final_df["serp_url_3"] = serp3_list

# --- Spaltenreihenfolge ---
cols = final_df.columns.tolist()
if "similarity" in cols and "gpt_score" in cols:
    sim_idx = cols.index("similarity")
    cols.remove("gpt_score")
    cols.insert(sim_idx + 1, "gpt_score")
    final_df = final_df[cols]

# --- Sortieren & Exportieren ---
final_df = final_df.sort_values(by=["url", "gpt_score", "similarity"], ascending=[True, False, False])
st.success("✅ Analyse abgeschlossen!")
st.dataframe(final_df)

csv = final_df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Ergebnis als CSV herunterladen", data=csv, file_name="keyword_mapping_final.csv", mime="text/csv")
