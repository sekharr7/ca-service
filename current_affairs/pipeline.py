import asyncio
import random
import time
from datetime import date, datetime
from urllib.parse import urlparse
from sqlalchemy import orm

from crawl4ai import AsyncWebCrawler

from current_affairs.crawler import fetch_article, preprocess_url, _is_blocked_response
from current_affairs.synthesizer import (
    build_system_prompt,
    generate_content,
    generate_content_from_restricted_url,
    clean_and_format_markdown,
    split_markdown_sections,
)
from current_affairs.classifier import classify_topic_subtopic
from current_affairs.image_generator import generate_and_save_image
from current_affairs.validator import is_error_page_title
from current_affairs.database import CurrentAffair, Highlight, Image, Keyword


async def process_article_service(
    ca_created_date: str | datetime | date,
    theme: str,
    article_url: str,
    db: orm.Session,
    theme_topic: str | None = None,
    theme_subtopic: str | None = None,
    wait_for_image: bool = False,
    crawler: AsyncWebCrawler | None = None,
) -> dict:
    """
    Process a single article:
    1. Scrapes URL using local Crawl4AI Chromium.
    2. Synthesizes UPSC coaching notes via Perplexity.
    3. Resolves missing theme_topic/theme_subtopic via gpt-5.6-luna against DB taxonomy.
    4. Inserts CurrentAffair, Highlight, Image, and Keyword records with exact file date.
    5. Dispatches infographic image generation and Azure Blob upload (synchronously or background task).
    """
    t_row_start = time.time()

    # 1. Fetch article
    processed_url = preprocess_url(article_url)
    domain = urlparse(processed_url).netloc
    prompt = f"news article from {domain} current affairs"
    t_scrape = time.time()
    
    article_text = ""
    scrape_failed = False
    try:
        article_text = await fetch_article(processed_url, prompt, crawler=crawler)
        scrape_duration = round(time.time() - t_scrape, 1)
        print(f"[SCRAPE] Extracted {len(article_text)} chars in {scrape_duration}s", flush=True)
        if not article_text or len(article_text.strip()) < 50 or _is_blocked_response(article_text):
            scrape_failed = True
    except Exception as scrape_err:
        print(f"[SCRAPE] Crawl failed for {processed_url} ({scrape_err}), using restricted URL fallback...", flush=True)
        scrape_failed = True

    # 2. Synthesize UPSC educational notes
    system_prompt = build_system_prompt()
    t_llm = time.time()
    
    if scrape_failed:
        print(f"[FALLBACK] Source paywalled/blocked for {processed_url}. Synthesizing via gpt-5.6-luna...", flush=True)
        generated_content = generate_content_from_restricted_url(url=processed_url, system_prompt=system_prompt)
    else:
        generated_content = generate_content(article_text, system_prompt)
    
    llm_duration = round(time.time() - t_llm, 1)

    if not generated_content:
        raise ValueError("No content generated")

    clean_content = clean_and_format_markdown(generated_content)
    title, subtitle, rest_content, keywords = split_markdown_sections(clean_content)
    print(f"Keywords: {keywords}", flush=True)

    # Validate extracted title to prevent error pages from entering the DB
    if is_error_page_title(title):
        raise ValueError(f"Generated content is an error page: '{title}'")

    # 3. Classify missing theme_topic or theme_subtopic against mtr_topics/mtr_subtopics
    if not theme_topic or not theme_subtopic or str(theme_topic).strip() == "" or str(theme_subtopic).strip() == "":
        print(f"[TAXONOMY] Classifying missing topic/subtopic against mtr_topics/mtr_subtopics using gpt-5.6-luna...", flush=True)
        detected_topic, detected_subtopic = classify_topic_subtopic(
            db=db,
            theme=theme,
            title=title,
            subtitle=subtitle,
            keywords=keywords,
            existing_topic=theme_topic,
        )
        if (not theme_topic or str(theme_topic).strip() == "") and detected_topic:
            theme_topic = detected_topic
            print(f"[TAXONOMY] Assigned theme_topic: '{theme_topic}'", flush=True)
        if (not theme_subtopic or str(theme_subtopic).strip() == "") and detected_subtopic:
            theme_subtopic = detected_subtopic
            print(f"[TAXONOMY] Assigned theme_subtopic: '{theme_subtopic}'", flush=True)

    # 4. Parse date strictly from the date specified in the file
    if isinstance(ca_created_date, (datetime, date)):
        parsed_date = datetime(ca_created_date.year, ca_created_date.month, ca_created_date.day)
    elif isinstance(ca_created_date, str) and ca_created_date.strip():
        for fmt in ("%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"):
            try:
                parsed_date = datetime.strptime(ca_created_date.strip(), fmt)
                break
            except ValueError:
                pass
        else:
            raise ValueError(f"Could not parse date from file: '{ca_created_date}'")
    else:
        raise ValueError(f"Missing date in file for this article (got {repr(ca_created_date)})")

    # ca_created_date and created_at strictly follow the file's specified date
    created_at = parsed_date.replace(hour=16, minute=7, second=7)

    # 5. Database staging and commit
    ca_entry = CurrentAffair(
        topic=title,
        content=subtitle,
        theme=theme,
        theme_topic=theme_topic,
        theme_subtopic=theme_subtopic,
        ca_created_date=parsed_date,
        created_at=created_at,
        source=processed_url,
        view_count=random.randint(7, 56),
    )
    db.add(ca_entry)
    db.flush()

    highlight = Highlight(
        id=ca_entry.id,
        current_affair_id=ca_entry.id,
        highlight_text=rest_content,
        created_at=created_at,
    )
    db.add(highlight)

    image_url = f"https://yookicaimages.blob.core.windows.net/thumbnail/{ca_entry.id}.jpg"
    db.add(Image(
        id=ca_entry.id,
        current_affair_id=ca_entry.id,
        image_url=image_url,
        created_at=created_at,
    ))

    db.add(Keyword(
        id=ca_entry.id,
        current_affair_id=ca_entry.id,
        kword=keywords,
    ))

    db.commit()
    print(f"[DB] Article record {ca_entry.id} committed successfully", flush=True)

    # 6. Image generation (parallel async task or synchronous wait)
    image_task = None
    if wait_for_image:
        print(f"[IMAGE] Generating and uploading image for article {ca_entry.id}...", flush=True)
        await asyncio.to_thread(
            generate_and_save_image,
            ca_entry.id,
            title,
            subtitle,
        )
    else:
        print(f"[IMAGE] Spawning concurrent background image generation for article {ca_entry.id}...", flush=True)
        image_task = asyncio.create_task(
            asyncio.to_thread(
                generate_and_save_image,
                ca_entry.id,
                title,
                subtitle,
            )
        )

    total_article_duration = round(time.time() - t_row_start, 1)
    print(f"[TIMING] Article pipeline completed in {total_article_duration}s (Scrape: {scrape_duration}s | LLM: {llm_duration}s)", flush=True)

    return {
        "id": ca_entry.id,
        "topic": title,
        "theme_topic": theme_topic,
        "theme_subtopic": theme_subtopic,
        "markdown": clean_content,
        "images": [image_url],
        "image_task": image_task,
    }
