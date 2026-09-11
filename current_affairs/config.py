import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure stdout is unbuffered for real-time progress logging
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# Project Paths
BASE_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_OUTPUT_DIR = BASE_DIR / "generated_images"
IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Crawl4AI Runtime & Playwright Setup
CRAWL4AI_RUNTIME = "/Volumes/WD_1TB/LLM/runtimes/crawl4ai/.venv/lib/python3.13/site-packages"
if os.path.exists(CRAWL4AI_RUNTIME) and CRAWL4AI_RUNTIME not in sys.path:
    sys.path.insert(0, CRAWL4AI_RUNTIME)

os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", "/Volumes/WD_1TB/LLM/cache/playwright")

CHROMIUM_EXECUTABLE_PATH = (
    "/Volumes/WD_1TB/LLM/cache/playwright/chromium-1234/"
    "chrome-mac-arm64/Google Chrome for Testing.app/"
    "Contents/MacOS/Google Chrome for Testing"
)
if os.path.exists(CHROMIUM_EXECUTABLE_PATH):
    os.environ["CHROME_BIN"] = CHROMIUM_EXECUTABLE_PATH

# Load environment variables
load_dotenv(dotenv_path=BASE_DIR / ".env")

# Database
DB_URI = os.getenv("DB_URI")

# Perplexity API
PPLEX_API_KEY = os.getenv("CA_GEN_PPLEX_API_KEY") or os.getenv("PPLEX_API_KEY")

# Azure Blob Storage
AZ_CONN_STR = os.getenv("CA_GEN_AZURE_STORAGE_CONNECTION_STRING") or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZ_CONTAINER = os.getenv("CA_GEN_AZURE_CONTAINER_NAME", "thumbnail")

# Azure OpenAI
CA_GEN_AZURE_OPENAI_API_KEY = os.getenv("CA_GEN_AZURE_OPENAI_API_KEY") or os.getenv("AZURE_API_KEY")
CA_GEN_AZURE_OPENAI_API_VERSION = os.getenv("CA_GEN_AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
CA_GEN_AZURE_OPENAI_ENDPOINT = os.getenv("CA_GEN_AZURE_OPENAI_ENDPOINT")

# Azure OpenAI Models
AZURE_OPENAI_IMAGE_DEPLOYMENT = (
    os.getenv("CA_GEN_AZURE_OPENAI_IMAGE_DEPLOYMENT")
    or os.getenv("AZURE_OPENAI_IMAGE_DEPLOYMENT", "gpt-image-2")
)
MODEL_GENERATOR = os.getenv("MODEL_GENERATOR", "gpt-5.6-luna")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

# Request Timeouts & Retries
LLM_TIMEOUT = 60.0
IMAGE_TIMEOUT = 90.0
TAXONOMY_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 2
