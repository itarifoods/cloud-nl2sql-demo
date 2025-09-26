
import os
import streamlit as st
import pandas as pd
import duckdb
import json
import re
import requests
from io import StringIO
import plotly.express as px

# =====================
# CONFIG (env-driven)
# =====================
# Works with any OpenAI-compatible "chat/completions" endpoint.
# Defaults to OpenAI if OPENAI_API_BASE is not set.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # change as needed

# Security: Only SELECT allowed
FORBIDDEN_TOKENS = ["UPDATE","DELETE","INSERT","DROP","ALTER","CREATE","ATTACH","COPY","EXPORT","IMPORT","PRAGMA"]

st.set_page_config(page_title="Cloud NL→SQL Agent", layout="wide")
st.title("🧠 Cloud NL→SQL Agent (Excel or CSV → DuckDB)")

with st.sidebar:
    st.markdown("### Model Settings")
    st.write("Using an OpenAI-compatible API.")
    model = st.text_input("Model", value=OPENAI_MODEL)
    if model:
        OPENAI_MODEL = model
    st.caption("Set OPENAI_API_KEY in environment/secrets. For Azure-compatible providers, also set OPENAI_API_BASE.")

st.markdown("""
Upload your dataset (Excel/CSV). The app builds an in-memory DuckDB table and answers natural-language questions by converting them to **safe SQL SELECT** via a hosted LLM.
""")

# =====================
# File upload
# =====================
uploaded = st.file_uploader("Upload .xlsx or .csv", type=["xlsx","csv"], accept_multiple_files=False)
table_name = st.text_input("DuckDB table name", value="sales")

def load_df(file):
    if file.name.lower().endswith(".csv"):
        df = pd.read_csv(file)
    else:
        # Excel -> first sheet
        df = pd.read_excel(file, sheet_name=None)
        # pick first non-empty sheet
        for name, d in df.items():
            if len(d.columns) > 0:
                df = d
                break
        if isinstance(df, dict):
            # fallback empty
            df = pd.DataFrame()
    # normalize columns
    df.columns = [str(c).strip().replace(" ", "_").replace("-", "_") for c in df.columns]
    return df

def build_schema_description(df: pd.DataFrame):
    cols = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        sample_vals = df[col].dropna().astype(str).head(5).tolist()
        cols.append({"name": col, "dtype": dtype, "samples": sample_vals})
    return json.dumps({"table": table_name, "columns": cols}, ensure_ascii=False, indent=2)

def only_select(sql: str) -> bool:
    s = sql.strip().strip(";")
    if ";" in s:
        return False
    up = s.upper()
    if not up.startswith("SELECT"):
        return False
    if any(tok in up for tok in FORBIDDEN_TOKENS):
        return False
    return True

SYSTEM_PROMPT = """You are a senior data analyst. Convert the user's question into a single valid SQL SELECT for DuckDB.
Return only the SQL code block. Do not add explanation.

Constraints:
- The table name is {table_name}.
- Use only the provided columns.
- If the question is ambiguous, choose the most reasonable interpretation and proceed.
- Prefer aggregations and GROUP BY when the user asks totals per dimension.
- Use CAST/TRY_CAST for dates or numerics if needed.
- LIMIT 1000 by default to keep results small.

Example format:
```sql
SELECT ...
```
"""

USER_PROMPT_TEMPLATE = """
The schema (DuckDB table) is:

{schema}

Question (can be Greek or English):
{question}

Remember:
- Table name: {table_name}
- Return ONLY one SQL in a fenced block ```sql ... ```
"""

def llm_chat_complete(messages, temperature=0.1):
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    url = f"{OPENAI_API_BASE}/chat/completions"
    payload = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

def to_sql_from_llm(schema_desc: str, question: str) -> str:
    sys = SYSTEM_PROMPT.format(table_name=table_name)
    user = USER_PROMPT_TEMPLATE.format(schema=schema_desc, question=question, table_name=table_name)
    content = llm_chat_complete([
        {"role": "system", "content": sys},
        {"role": "user", "content": user}
    ])
    m = re.search(r"```sql\s*(.*?)```", content, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"(SELECT[\\s\\S]+)$", content, flags=re.IGNORECASE)
    return m2.group(1).strip() if m2 else content.strip()

# =====================
# Main app
# =====================
if uploaded is not None:
    df = load_df(uploaded)
    if df.empty:
        st.error("Το αρχείο δεν περιείχε δεδομένα ή δεν διαβάστηκε σωστά.")
        st.stop()
    st.success(f"Loaded dataset with shape {df.shape}.")
    st.dataframe(df.head(15), use_container_width=True)

    schema_desc = build_schema_description(df)

    con = duckdb.connect(":memory:")
    con.register("df_view", df)
    con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df_view;")

    st.subheader("Ask a question")
    question = st.text_input(
        "Ρώτα κάτι για τα δεδομένα (π.χ. 'Top 10 πελάτες σε αξία' ή 'Πωλήσεις ανά μήνα 2024')",
        value=""
    )
    run = st.button("Run")

    if run and question.strip():
        if not OPENAI_API_KEY:
            st.error("Λείπει το OPENAI_API_KEY από τα secrets/περιβάλλον.")
        else:
            with st.spinner("Thinking..."):
                try:
                    sql = to_sql_from_llm(schema_desc, question)
                except Exception as e:
                    st.error(f"LLM call error: {e}")
                    st.stop()
            st.code(sql, language="sql")

            if not only_select(sql):
                st.error("Για λόγους ασφαλείας εκτελώ μόνο ένα απλό SELECT χωρίς άλλα statements.")
            else:
                try:
                    res = con.execute(sql).fetchdf()
                    st.success(f"OK. Rows: {len(res)}")
                    st.dataframe(res, use_container_width=True)

                    # Try simple chart
                    if 1 < len(res.columns) <= 3:
                        num_cols = [c for c in res.columns if pd.api.types.is_numeric_dtype(res[c])]
                        if num_cols:
                            x = [c for c in res.columns if c not in num_cols][0] if len(res.columns) > 1 else res.columns[0]
                            y = num_cols[0]
                            fig = px.bar(res, x=x, y=y)
                            st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"SQL error: {e}")

    with st.expander("See schema used by the model"):
        st.code(schema_desc, language="json")
else:
    st.info("📄 Ανέβασε ένα Excel ή CSV για να ξεκινήσεις.")
