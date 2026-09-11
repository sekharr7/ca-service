#!/usr/bin/env python3
"""
Current Affairs Service CLI Entrypoint.
Reads an input Excel file (default: ca_lists.xlsx), scrapes news articles using Crawl4AI,
synthesizes UPSC notes via Perplexity, classifies topics/subtopics via gpt-5.6-luna,
generates infographics via Azure OpenAI, uploads them to Azure Blob Storage,
and stores records into the development MySQL database (mntor_dev).

Usage:
    ./.venv/bin/python ca_local_main.py [excel_filename]
"""

import asyncio
import sys

from current_affairs import (
    process_excel_file,
    process_article_service,
    fetch_article,
    generate_content,
    generate_and_save_image,
    classify_topic_subtopic,
    validate_article_for_db,
    get_db_session,
    CurrentAffair,
    Highlight,
    Image,
    Keyword,
    BulkJob,
)


def main():
    target_file = sys.argv[1] if len(sys.argv) > 1 else "ca_lists.xlsx"
    print(f"[CLI] Launching Current Affairs pipeline on: {target_file}", flush=True)
    asyncio.run(process_excel_file(target_file))


if __name__ == "__main__":
    main()
