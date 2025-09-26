
# Cloud NL→SQL Agent (DuckDB + Streamlit)

A portable Streamlit app that answers natural-language questions about an uploaded dataset (Excel/CSV) by generating **safe SQL `SELECT`** for DuckDB using a **hosted LLM (OpenAI-compatible API)**.

## Why this fits "runs from anywhere"
- No on-prem dependencies.
- Works on Streamlit Community Cloud, Azure App Service, Hugging Face Spaces, or any VM.
- Uses environment variables for API credentials.
- Upload the file from your browser at runtime (no local path assumptions).

## Quickstart (Streamlit Community Cloud)
1. Push these files to a new GitHub repo.
2. Create a new Streamlit app from that repo.
3. In app settings → **Secrets**, set:
   - `OPENAI_API_KEY = sk-...`
   - *(Optional)* `OPENAI_API_BASE = https://your-azure-openai-endpoint/openai/deployments/...` (OpenAI-compatible)
   - *(Optional)* `OPENAI_MODEL = gpt-4o-mini` (or your deployed model name)
4. Open the app, upload your Excel/CSV, and ask questions in Greek or English.

## Quickstart (Local)
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...  # Windows: set OPENAI_API_KEY=...
streamlit run app.py
```

## Azure App Service (Linux, container-less)
- Create a Web App (Python), enable **App Settings**:
  - `OPENAI_API_KEY` (required)
  - `OPENAI_API_BASE` (if using Azure/OpenAI-compatible)
  - `OPENAI_MODEL` (optional)
- Deploy the folder (via GitHub Actions or Zip Deploy).
- Browse to the site, upload your file, and query.

## Security Notes
- The app **executes only one `SELECT`** per run; rejects write/DDL keywords.
- Data stays in-memory (DuckDB). No persistence by default.
- For production: add row/column masking and explicit column allowlists.

## Extending to Azure SQL (read-only, still "runs anywhere")
- Swap the file-upload step with a direct read-only connection string (e.g., `pyodbc` or `sqlalchemy` to Azure SQL).
- Generate SQL over defined views; keep the LLM prompt constrained to those views.
- Use managed identity if hosting in Azure for secretless auth.
