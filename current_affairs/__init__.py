"""
Current Affairs ingestion and synthesis package.
"""

from current_affairs.config import (
    BASE_DIR,
    LOGS_DIR,
    IMAGE_OUTPUT_DIR,
    DB_URI,
    MODEL_GENERATOR,
    AZURE_OPENAI_IMAGE_DEPLOYMENT,
)
from current_affairs.database import (
    Base,
    CurrentAffair,
    Highlight,
    Image,
    Keyword,
    BulkJob,
    get_db_session,
)
from current_affairs.crawler import fetch_article, preprocess_url
from current_affairs.synthesizer import (
    generate_content,
    build_system_prompt,
    clean_and_format_markdown,
    split_markdown_sections,
)
from current_affairs.classifier import classify_topic_subtopic
from current_affairs.image_generator import generate_and_save_image
from current_affairs.validator import validate_article_for_db, classify_failure_reason
from current_affairs.pipeline import process_article_service
from current_affairs.excel_processor import process_excel_file

__all__ = [
    "BASE_DIR",
    "LOGS_DIR",
    "IMAGE_OUTPUT_DIR",
    "DB_URI",
    "MODEL_GENERATOR",
    "AZURE_OPENAI_IMAGE_DEPLOYMENT",
    "Base",
    "CurrentAffair",
    "Highlight",
    "Image",
    "Keyword",
    "BulkJob",
    "get_db_session",
    "fetch_article",
    "preprocess_url",
    "generate_content",
    "build_system_prompt",
    "clean_and_format_markdown",
    "split_markdown_sections",
    "classify_topic_subtopic",
    "generate_and_save_image",
    "validate_article_for_db",
    "classify_failure_reason",
    "process_article_service",
    "process_excel_file",
]
