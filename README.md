# DMV Document Processing

CLI tool for processing scanned DMV document PDFs.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env` (or Vertex credentials when using `AI_PROVIDER=vertex`).

## Usage

Classify pages in one or more PDFs, split them into documents, preprocess each page, and write artifacts:

```bash
dmv path/to/scan.pdf
dmv file1.pdf file2.pdf
```

### Output

For each input file `scan.pdf`, the tool creates `artifacts/scan/` containing:

- `original.pdf` — copy of the input
- `doc_classification.json` — AI classification result
- `classified/` — one PDF per classified document (lossless split from the original)
- `corrected/` — preprocessed PDFs (deskew, perspective correction, content crop)
- `extracted/` — JSON field extraction per corrected document (raw AI output)

Preprocessing is **doc-type aware**:
- Forms/invoices (`dealer_invoice`, title apps, etc.) — Hough deskew + safe crop only (no perspective warp)
- ID cards (`driver_license`) — card detection, orientation, and crop tuned for wallet/flatbed scans
- Insurance and other types — Hough deskew + gated page-level perspective + safe crop

Deskew uses Hough line angles (better for full-page forms). Perspective warp only runs when a detected quadrilateral covers ~88–99.5% of the page, avoiding inner table boxes. Crop is skipped if it would remove more than ~15% of the page area.

**Extraction** runs after preprocessing on each corrected PDF (excluding debug-mode test fixtures). Each document is uploaded to the AI provider, fields are extracted using a JSON schema with canonical names from `artifacts/blanks/`, and results are saved to `extracted/*.json`.

### Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI API key (when `AI_PROVIDER=openai`) |
| `AI_PROVIDER` | `openai` | AI provider (`openai` or `vertex`) |
| `OPENAI_MODEL` | `gpt-4o` | OpenAI model |
| `VERTEX_PROJECT` | — | GCP project (optional; falls back to SA `project_id`) |
| `VERTEX_LOCATION` | `us-central1` | Vertex AI region |
| `VERTEX_MODEL` | `gemini-3.1-pro-preview` | Vertex Gemini model |
| `VERTEX_SERVICE_ACCOUNT_JSON` | — | Path to GCP service account JSON |
| `WORKER_POOL_SIZE` | `5` | Max concurrent workers (classification + preprocessing) |
| `MAX_AI_RETRIES` | `3` | Retries with exponential backoff |
| `AI_RETRY_BASE_DELAY_SECONDS` | `1.0` | Base delay between retries |
| `ARTIFACTS_DIR` | `artifacts` | Output directory |
| `BLANKS_DIR` | `artifacts/blanks` | Blank form templates (cover + 3 NJMVC PDFs) |
| `PREPROCESS_DPI` | `200` | Page rasterization DPI for preprocessing (150–200) |
| `DEBUG_MODE` | `false` | Exclude test-fixture output forms from further processing |
| `OPENAI_INPUT_PRICE_PER_MILLION` | — | Optional USD override for cost estimates |
| `OPENAI_OUTPUT_PRICE_PER_MILLION` | — | Optional USD override for cost estimates |
| `OPENAI_CACHED_INPUT_PRICE_PER_MILLION` | — | Optional cached-input rate override |

Built-in pricing is used for known models (e.g. `gpt-5.4`, `gpt-4o`, `gemini-3.1-pro-preview`) when overrides are not set. The CLI prints processing time, token usage, and an estimated USD cost after each file.

## Tests

```bash
pytest
```
