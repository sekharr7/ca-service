# UPSC Current Affairs Ingestion & Synthesis Pipeline

An automated, local pipeline designed to ingest news articles from Excel spreadsheets, crawl content bypassing bot-detection blocks, synthesize high-yield UPSC Civil Services examination coaching notes, automatically classify topics against syllabus taxonomies, generate high-resolution editorial infographics, and store records across database tiers.

---

## 1. System Architecture

The pipeline is organized as a modular, decoupled Python package:

```
yooki/
├── current_affairs/                   # Core modular package
│   ├── __init__.py                    # Public API exports
│   ├── config.py                      # Environment variables, timeouts, project paths
│   ├── database.py                    # SQLAlchemy Models & Session Factory
│   ├── crawler.py                     # Crawl4AI local Chromium engine & site-specific extractors
│   ├── synthesizer.py                 # Perplexity note synthesis & markdown parsers
│   ├── classifier.py                  # gpt-5.6-luna taxonomy classification (mtr_subjects/topics/subtopics)
│   ├── image_generator.py             # Azure OpenAI (gpt-image-2) infographic generation & Blob upload
│   ├── validator.py                   # Data quality verification & dummy content guards
│   ├── pipeline.py                    # Single article orchestration service
│   └── excel_processor.py             # Batch Excel reader, concurrent image tasks, bulk job tracking
├── logs/                              # Audit logs (ca_status_log_<timestamp>.csv, ca_status_log_latest.csv)
├── ca_local_main.py                   # CLI entrypoint wrapper
├── pyproject.toml                     # Dependency declarations
└── README.md                          # System documentation
```

---

## 2. Models in Use

| Capability | Model | Provider | Key / Configuration |
| :--- | :--- | :--- | :--- |
| **Taxonomy Classification** | **`gpt-5.6-luna`** | **Azure OpenAI** | `MODEL_GENERATOR` |
| **UPSC Notes Synthesis (Primary)** | **`sonar-pro`** | **Perplexity AI** | `CA_GEN_PPLEX_API_KEY` |
| **UPSC Notes Synthesis (Fallback)** | **`sonar-reasoning`** | **Perplexity AI** | `CA_GEN_PPLEX_API_KEY` |
| **Infographic Image Generation** | **`gpt-image-2`** | **Azure OpenAI** | `CA_GEN_AZURE_OPENAI_IMAGE_DEPLOYMENT` |

---

## 3. Key Pipeline Features

### A. Parallel, Non-Blocking Execution
- **Zero Idle Wait**: Once an article completes notes generation and database staging (~8s), its infographic generation (`gpt-image-2`) and Azure Blob upload run as a concurrent background task.
- The crawler immediately proceeds to fetch the subsequent row without waiting for the 15–20s image generation to finish.
- All background tasks are awaited before finalizing the bulk job status.

### B. Automated Taxonomy Classification
- When `theme_topic` or `theme_subtopic` is blank in the input spreadsheet, the pipeline triggers `current_affairs.classifier.classify_topic_subtopic()`.
- It dynamically queries `mtr_subjects`, `mtr_topics`, and `mtr_subtopics` in the development database.
- It leverages the synthesized `title`, `subtitle`, and `keywords` (instead of raw 3,000-word HTML) with `gpt-5.6-luna` to pick the exact topic and subtopic with minimal token consumption.
- The classified topic and subtopic are saved into `mtr_current_affairs_edk` **and** written back into the Excel spreadsheet cells.

### C. Pre-Crawl Duplicate Guard
- Before initiating browser crawls, the pipeline checks `mtr_current_affairs_edk` for existing matching URLs.
- Duplicates are skipped in **under 1 second**, recorded in `mtr_ca_bulk_jobs` with details:
  ```json
  [{"row": 2, "url": "...", "reason": "Duplicate (id=9877)", "status": "skipped"}]
  ```
- Directly compatible with SQL queries:
  ```sql
  SELECT * FROM mtr_ca_bulk_jobs WHERE details LIKE '%Duplicate%';
  ```

### D. File-Preserved Dates
- Dates (`ca_created_date` and `created_at`) strictly follow the date specified in the Excel sheet (`DD-MM-YYYY`), with `created_at` set to `16:07:07` on that specified date.

### E. Isolated Logging
- Link-by-link outcome logs are written directly to `logs/ca_status_log_<timestamp>.csv`, and a copy is maintained at `logs/ca_status_log_latest.csv`.

---

## 4. Multi-Stage Database Promotion (Dev ➔ Test ➔ Prod)

### Validation Rules (`current_affairs.validator`)
Before any record is promoted across environments, `validate_article_for_db(article_id, db)` verifies:
1. **Topic**: Must not contain dummy strings (`lorem ipsum`, `dummy`, `test`, `untitled`) or HTTP error markers (`403 forbidden`, `404 not found`, `access denied`).
2. **Subtitle**: Must be substantive (> 20 chars) and free of placeholder text.
3. **Taxonomy**: `theme`, `theme_topic`, and `theme_subtopic` must all be populated and valid.
4. **Highlights**: UPSC level-5 markdown headers (`#####`) present, minimum 400 characters.
5. **Image URL**: Must match Azure Blob format (`https://...blob.core.windows.net/thumbnail/{article_id}.jpg`).
6. **Keywords**: At least 2 comma-separated tags present in `mtr_ca_kword_edk`.

### Exact ID Preservation
Because the infographic image is generated once during initial ingestion and uploaded as `thumbnail/<article_id>.jpg`, the same primary key `id` must be preserved when promoting records from `mntor_dev` ➔ `mntor_test` ➔ `mntor_prod`.

Migration script pattern:
```sql
-- Explicitly insert existing primary key ID
INSERT INTO mtr_current_affairs_edk (id, topic, content, theme, theme_topic, theme_subtopic, ca_created_date, created_at, source, view_count, pinecone_synced)
VALUES (:id, :topic, :content, :theme, :theme_topic, :theme_subtopic, :ca_created_date, :created_at, :source, :view_count, :pinecone_synced);

INSERT INTO mtr_ca_highlights_edk (id, current_affair_id, highlight_text, created_at)
VALUES (:id, :id, :highlight_text, :created_at);

INSERT INTO mtr_ca_images_edk (id, current_affair_id, image_url, created_at)
VALUES (:id, :id, :image_url, :created_at);

INSERT INTO mtr_ca_kword_edk (id, current_affair_id, kword)
VALUES (:id, :id, :kword);
```

---

## 5. Usage & CLI Commands

### Run the Ingestion Batch
To process articles from `ca_lists.xlsx`:
```bash
./.venv/bin/python ca_local_main.py ca_lists.xlsx
```

Or process a custom spreadsheet:
```bash
./.venv/bin/python ca_local_main.py custom_file.xlsx
```

### Programmatic Python Usage
```python
import asyncio
from current_affairs import process_excel_file, validate_article_for_db, get_db_session

# Run batch
asyncio.run(process_excel_file("ca_lists.xlsx"))

# Validate article quality in DB
with get_db_session() as session:
    is_valid, issues = validate_article_for_db(article_id=9877, db=session)
    print(f"Article 9877 valid: {is_valid}, Issues: {issues}")
```
