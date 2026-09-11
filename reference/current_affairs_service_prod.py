import asyncio
import base64
import os
import random
import re
from io import BytesIO
from datetime import datetime
from typing import List, Tuple
from urllib.parse import urlparse

from azure.storage.blob import BlobServiceClient
from bs4 import BeautifulSoup
from openai import AzureOpenAI, OpenAI
from PIL import Image as PILImage
from sqlalchemy import orm

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.content_filter_strategy import BM25ContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

from app.models.current_affairs import CurrentAffair, Highlight, Image, Keyword
# from app.models.current_affairs_test import CurrentAffairTest, HighlightTest, ImageTest, KeywordTest
from app.services.image_service import ImageService


from dotenv import load_dotenv
load_dotenv()

# ── Current Affairs Generation – env config ─────────────────────────────
CA_GEN_AZURE_OPENAI_ENDPOINT = os.getenv("CA_GEN_AZURE_OPENAI_ENDPOINT")
CA_GEN_AZURE_OPENAI_API_KEY = os.getenv("CA_GEN_AZURE_OPENAI_API_KEY")
CA_GEN_AZURE_OPENAI_API_VERSION = os.getenv("CA_GEN_AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
CA_GEN_AZURE_OPENAI_IMAGE_DEPLOYMENT = os.getenv("CA_GEN_AZURE_OPENAI_IMAGE_DEPLOYMENT", "gpt-image-2")
PPLEX_API_KEY = os.getenv("CA_GEN_PPLEX_API_KEY")
AZ_CONN_STR = os.getenv("CA_GEN_AZURE_STORAGE_CONNECTION_STRING")
AZ_CONTAINER = os.getenv("CA_GEN_AZURE_CONTAINER_NAME", "thumbnail")

image_client = AzureOpenAI(
    api_key=CA_GEN_AZURE_OPENAI_API_KEY,
    api_version=CA_GEN_AZURE_OPENAI_API_VERSION,
    azure_endpoint=CA_GEN_AZURE_OPENAI_ENDPOINT,
)

blob_service_client = BlobServiceClient.from_connection_string(AZ_CONN_STR)

def gen_image_prompt(topic: str, summary: str) -> str:
    return f"""
{topic}

{summary[:500]}

You're an AI capable of converting textual information into concrete images and your task is to convert the information in this article into symbolic images as you understand them.
Create a clean educational infographic for UPSC aspirants.

Style guidelines:
- Modern newspaper explainer style
- Structured infographic layout with visual sections
- Educational and fact-focused presentation
- Use icons, symbols, charts, policy visuals, governance visuals, science visuals, or institutional imagery when relevant
- Clear typography and concise educational labels are allowed
- Flat infographic or polished editorial infographic style
- Avoid cinematic photorealism
- Avoid fictional or highly detailed geographic maps
- Prefer schematic or symbolic map representations when needed
- Landscape orientation
- Suitable for serious civil services exam preparation
""".strip()


def generate_and_save_image(article_id: int, topic: str, summary: str):
    try:
        print(f"[IMAGE] Starting image generation for article {article_id}: {topic}")

        prompt = gen_image_prompt(topic, summary)

        response = image_client.images.generate(
            model=CA_GEN_AZURE_OPENAI_IMAGE_DEPLOYMENT,
            prompt=prompt,
            size="1536x1024"
        )

        image_b64 = response.data[0].b64_json
        image_bytes = base64.b64decode(image_b64)

        image = PILImage.open(BytesIO(image_bytes)).convert("RGB")
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=82, optimize=True)
        buffer.seek(0)

        blob_name = f"{article_id}.jpg"
        print(f"[IMAGE] Uploading image to Azure Blob Storage: {blob_name}")

        blob_client = blob_service_client.get_blob_client(
            container=AZ_CONTAINER,
            blob=blob_name
        )

        blob_client.upload_blob(buffer, overwrite=True)
        buffer.close()
        blob_url = blob_client.url

        print(f"[IMAGE] Azure Blob upload successful for article {article_id}: {blob_url}")

    except Exception as e:
        print(f"[IMAGE] Image generation failed for article {article_id}: {e}")

def build_system_prompt() -> str:
    return """
You are an advanced content generator creating markdown material for UPSC aspirants, specifically tailored for database storage and frontend markdown rendering. Your outputs will be referenced by serious exam candidates, so maintain a formal, informative, and concise tone throughout, as if writing official coaching notes or GS exam preparation material. 

Use Bold, underline and Italics as appropriate to highlight important areas (most important don't skip this)

**Task Structure:**
- Use *level-5 markdown headers* (#####) only, never lower or higher.
- Each output consists of strict sections: 
---
##### [Topic Name]
##### [Description in 15-20 words]
<span class="evaluation-result-card-but">GS Paper(s) with topic area e.g. GS III (Economy), GS II (Governance)</span>
(You MUST output exactly ONE span tag like above with relevant UPSC GS papers. Do NOT duplicate or nest span tags.)
##### Key Takeaways  
> [list of highlights from the article important for UPSC examination. Use colored circle emojis like 🔵 🟢 🟠 🔴 🟣 at the start of each bullet point for better cognition]
---
##### Background  
- [Use bullet points to list key facts, historical context, root causes. Each point must start with a hyphen (-) on a new line.]

---
##### Why is it Important for India?
- [Use bullet points to list reasons for national significance, sectors affected, policy impact, impact to institutions. Each point must start with a hyphen (-) on a new line.]

---
##### Data or Statistics  
- [Use bullet points to list up to 5 key statistics. Each point must start with a hyphen (-) on a new line.]

---
##### UPSC - Strategic Viewpoint 
> [Provide 4 or 5 strategic and visionary perspectives like a civil servant explaining the deeper WHY, showing reasoning and judgement, in bullets. Use contextually relevant emojis at the start of each bullet based on the topic - e.g., 🎯 for goals, ⚖️ for policy/law, 🌍 for global, 🇮🇳 for India-specific, 💡 for insights, 📈 for economy, 🛡️ for security, 🌱 for environment, 🏛️ for governance]


---
##### Keywords
[ONLY 5 comma-separated UPSC-relevant keywords including: key terms, constitutional provisions, acts/policies, institutions, concepts, and thematic connections]



- Never add formatting, no block quotes, code blocks, bold/italics, or links in output.

**Content Rules:**
- Synthesize content directly from the provided source article.
- Data or Statistics must be precise, sourced from official or primary documents/sources, and must accurately reflect the correct sequence of events, dates, implementations, and policy mandates for the topic. Always ensure numbers, years, and factual details are consistent.
- Do not repeat output instructions, give meta comments, or include citations/URLs.
- Use proper apostrophes (’) rather than straight (') for all contractions or possessives, ensuring UTF-8 compatibility.
- Be concise, yet insightful and content-rich, distilling only what matters for high-level civil services preparation.
- Write natural, human-like sentences — prioritize clarity, directness, and easy comprehension, avoiding complicated or robotic phrasing.
- STRICTLY DO NOT generate citations, references, footnotes, endnotes, superscripts, or bracketed markers of any kind (e.g. [1], [2], (Source), (Report), etc.)
- The "##### Keywords" section is MANDATORY and must ALWAYS be the last section in your output. You MUST include the exact header "##### Keywords" followed by exactly 5 comma-separated keywords on the next line. Never omit this section — output without it is considered invalid and incomplete.

**Technical:**
- Strictly do NOT wrap output in quotes or code blocks.
- Output must be usable in a SQL or markdown database without post-processing.

**Overall:**
- Prioritize authenticity, clarity, and subject authority in tone and organization.
- Never explain the prompt or process. Output ONLY the formatted markdown content for UPSC use as described.
"""

def is_pib_url(url: str) -> bool:
    parsed = urlparse(url.lower())
    domain = parsed.netloc.replace("www.", "")
    return domain == "pib.gov.in"


def is_bbc_url(url: str) -> bool:
    parsed = urlparse(url.lower())
    domain = parsed.netloc.replace("www.", "")
    return domain == "bbc.com" or domain.endswith("bbc.co.uk")


def is_idsa_url(url: str) -> bool:
    parsed = urlparse(url.lower())
    domain = parsed.netloc.replace("www.", "")
    return domain == "idsa.in"


def is_indianexpress_url(url: str) -> bool:
    parsed = urlparse(url.lower())
    domain = parsed.netloc.replace("www.", "")
    return domain == "indianexpress.com"


def is_thehindu_url(url: str) -> bool:
    parsed = urlparse(url.lower())
    domain = parsed.netloc.replace("www.", "")
    return domain == "thehindu.com"


def is_newindianexpress_url(url: str) -> bool:
    return "newindianexpress.com" in url.lower()


def is_toi_url(url: str) -> bool:
    return "timesofindia.indiatimes.com" in url.lower()


def is_downtoearth_url(url: str) -> bool:
    return "downtoearth.org.in" in url.lower()


def _extract_clean_paragraphs(container, min_len: int = 40):
    paragraphs = []

    for tag in container.find_all([
        'script', 'style', 'aside', 'button', 'nav',
        'header', 'footer', 'figure', 'img'
    ]):
        tag.decompose()

    for p in container.find_all('p'):
        text = p.get_text(" ", strip=True)
        text = re.sub(r'\s+', ' ', text).strip()

        if text and len(text) > min_len and not any(
            noise in text for noise in [
                'Subscribe', 'Advertisement', 'Show Comments',
                'Related Stories', 'Download the', 'Weather', 'Trending'
            ]
        ):
            paragraphs.append(text)

    return paragraphs


def _build_clean_article(soup, article_container, title_selector=None):
    clean_parts = []

    title = None
    if title_selector:
        title = soup.select_one(title_selector)
    if not title:
        title = soup.find('h1')

    if title:
        title_text = title.get_text(" ", strip=True)
        if title_text:
            clean_parts.append(title_text)

    if article_container:
        clean_parts.extend(_extract_clean_paragraphs(article_container))

    cleaned = "\n\n".join(clean_parts).strip()
    return cleaned if len(cleaned) > 150 else ""

def extract_best_content(result, url: str = "") -> str:
    contents = []

    # BBC pages often return heavily hydrated HTML. Extract readable
    # article paragraphs directly instead of returning raw DOM content.
    if url and is_bbc_url(url):
        try:
            html_content = ""
            if hasattr(result, 'cleaned_html') and result.cleaned_html:
                html_content = result.cleaned_html

            if html_content:
                soup = BeautifulSoup(html_content, 'lxml')
                article = soup.find('article')

                if article:
                    paragraphs = []
                    for p in article.find_all('p'):
                        text = p.get_text(" ", strip=True)
                        if text and len(text) > 40:
                            paragraphs.append(text)

                    if paragraphs:
                        return "\n\n".join(paragraphs)
        except Exception:
            pass

    if url and is_idsa_url(url):
        try:
            html_content = ""
            if hasattr(result, 'cleaned_html') and result.cleaned_html:
                html_content = result.cleaned_html

            if html_content:
                soup = BeautifulSoup(html_content, 'lxml')

                clean_parts = []

                title = soup.find('h1', id='posttitle')
                if title:
                    title_text = title.get_text(" ", strip=True)
                    if title_text:
                        clean_parts.append(title_text)

                article_container = (
                    soup.find('div', class_='footnote')
                    or soup.find('div', class_='inner-content-area')
                )

                if article_container:
                    paragraphs = []

                    for p in article_container.find_all('p'):
                        # Remove citation markers like [1], [2]
                        for a in p.find_all('a'):
                            a.decompose()

                        text = p.get_text(" ", strip=True)
                        text = re.sub(r'\[\d+\]', '', text).strip()

                        if text and len(text) > 40:
                            paragraphs.append(text)

                    # Some IDSA articles start with direct text before first <p>
                    direct_text = article_container.get_text(" ", strip=True)
                    if direct_text and len(direct_text) > 100:
                        first_para = direct_text.split("  ")[0].strip()
                        if first_para and first_para not in paragraphs:
                            paragraphs.insert(0, first_para)

                    clean_parts.extend(paragraphs)

                cleaned_content = "\n\n".join(clean_parts).strip()

                if cleaned_content and len(cleaned_content) > 100:
                    return cleaned_content
        except Exception:
            pass

    if url and is_indianexpress_url(url):
        try:
            html_content = ""
            if hasattr(result, 'cleaned_html') and result.cleaned_html:
                html_content = result.cleaned_html

            if html_content:
                soup = BeautifulSoup(html_content, 'lxml')

                clean_parts = []

                title = soup.find('h1', id='main-heading-article')
                subtitle = soup.find('h2', class_='synopsis')

                if title:
                    title_text = title.get_text(" ", strip=True)
                    if title_text:
                        clean_parts.append(title_text)

                if subtitle:
                    subtitle_text = subtitle.get_text(" ", strip=True)
                    if subtitle_text:
                        clean_parts.append(subtitle_text)

                article_container = soup.find('div', id='pcl-full-content')

                if article_container:
                    for tag in article_container.find_all([
                        'script', 'style', 'aside', 'figure', 'img', 'button'
                    ]):
                        tag.decompose()

                    paragraphs = []
                    for p in article_container.find_all('p'):
                        text = p.get_text(" ", strip=True)
                        text = re.sub(r'\s+', ' ', text).strip()

                        if text and len(text) > 40:
                            paragraphs.append(text)

                    clean_parts.extend(paragraphs)

                cleaned_content = "\n\n".join(clean_parts).strip()

                if cleaned_content and len(cleaned_content) > 150:
                    return cleaned_content
        except Exception:
            pass

    if url and is_thehindu_url(url):
        try:
            html_content = ""
            if hasattr(result, 'cleaned_html') and result.cleaned_html:
                html_content = result.cleaned_html

            if html_content:
                soup = BeautifulSoup(html_content, 'lxml')

                clean_parts = []

                title = soup.find('h1')
                if title:
                    title_text = title.get_text(" ", strip=True)
                    if title_text:
                        clean_parts.append(title_text)

                article_container = (
                    soup.select_one('div.articlebodycontent')
                    or soup.select_one('div.storyline')
                    or soup.select_one('div.story-content')
                    or soup.find('article')
                )

                if article_container:
                    for tag in article_container.find_all([
                        'script', 'style', 'aside', 'button', 'nav',
                        'header', 'footer', 'figure', 'img'
                    ]):
                        tag.decompose()

                    paragraphs = []
                    for p in article_container.find_all('p'):
                        text = p.get_text(" ", strip=True)
                        text = re.sub(r'\s+', ' ', text).strip()

                        if (
                            text
                            and len(text) > 40
                            and 'customersupport@thehindu.co.in' not in text
                            and 'Need help with your subscription' not in text
                            and 'Account Settings' not in text
                        ):
                            paragraphs.append(text)

                    clean_parts.extend(paragraphs)

                cleaned_content = "\n\n".join(clean_parts).strip()

                if cleaned_content and len(cleaned_content) > 150:
                    return cleaned_content
        except Exception:
            pass

    if url and is_newindianexpress_url(url):
        try:
            soup = BeautifulSoup(result.cleaned_html or "", 'lxml')
            article = (
                soup.select_one('.full-details')
                or soup.select_one('#pcl-full-content')
                or soup.select_one('.story_details')
            )
            if article:
                cleaned = _build_clean_article(soup, article)
                if cleaned:
                    return cleaned
        except Exception:
            pass

    if url and is_toi_url(url):
        try:
            soup = BeautifulSoup(result.cleaned_html or "", 'lxml')
            article = (
                soup.select_one('div[data-articlebody="1"]')
                or soup.select_one('.article_content')
                or soup.select_one('._s30J')
            )
            if article:
                cleaned = _build_clean_article(soup, article)
                if cleaned:
                    return cleaned
        except Exception:
            pass

    if url and is_downtoearth_url(url):
        try:
            soup = BeautifulSoup(result.cleaned_html or "", 'lxml')
            article = (
                soup.select_one('article')
                or soup.select_one('.story-element-text')
                or soup.select_one('.article-content')
            )
            if article:
                cleaned = _build_clean_article(soup, article)
                if cleaned:
                    return cleaned
        except Exception:
            pass

    try:
        if hasattr(result, 'cleaned_html') and result.cleaned_html:
            soup = BeautifulSoup(result.cleaned_html, 'lxml')

            if url and is_pib_url(url):
                press_div = (
                    soup.find('div', id='divPressRelease')
                    or soup.find('div', id='divPressNote')
                    or soup.find(id='ContentPlaceHolder1_divContent')
                    or soup.select_one('.content-area')
                    or soup.select_one('.main-content')
                    or soup.select_one('.article-content')
                    or soup.select_one('.field-content')
                    or soup.select_one('.content')
                    or soup.find('div', class_='innner-page-main-about-us-content-right-part')
                )

                if press_div:
                    clean_parts = []

                    title = soup.find('h2', id='Titleh2')
                    subtitle = soup.find('h3', id='Subtitleh3')

                    if title:
                        title_text = title.get_text(" ", strip=True)
                        if title_text:
                            clean_parts.append(title_text)

                    if subtitle:
                        subtitle_text = subtitle.get_text(" ", strip=True)
                        if subtitle_text:
                            clean_parts.append(subtitle_text)

                    paragraphs = []
                    for p in press_div.find_all('p'):
                        text = p.get_text(" ", strip=True)
                        if text and len(text) > 30:
                            paragraphs.append(text)

                    clean_parts.extend(paragraphs)

                    cleaned_content = "\n\n".join(clean_parts).strip()

                    if cleaned_content and len(cleaned_content) > 100:
                        return cleaned_content

            common_article_selectors = [
                'article',
                "[role='main']",
                '.article-body',
                '.story-body',
                '.article',
                '.article-content',
                '.post-content',
                '.entry-content',
                '.story-content',
                '.content-body',
                '.description',
                '#DivListing',
                '#content',
                'main',
            ]

            article_container = None
            for selector in common_article_selectors:
                article_container = soup.select_one(selector)
                if article_container:
                    break

            if article_container:
                clean_parts = []

                for tag in article_container.find_all(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                    tag.decompose()

                title_tag = article_container.find(['h1', 'h2'])
                if title_tag:
                    title_text = title_tag.get_text(" ", strip=True)
                    if title_text:
                        clean_parts.append(title_text)

                paragraphs = []
                for p in article_container.find_all('p'):
                    text = p.get_text(" ", strip=True)
                    text = re.sub(r'\s+', ' ', text).strip()
                    text = re.sub(r'\[\d+\]', '', text)

                    if text and len(text) > 40:
                        paragraphs.append(text)

                clean_parts.extend(paragraphs)

                cleaned_content = "\n\n".join(clean_parts).strip()

                if cleaned_content and len(cleaned_content) > 150:
                    return cleaned_content

            tables = soup.find_all('table')
            table_content = "\n".join([t.get_text(separator='\n', strip=True) for t in tables])
            if table_content and len(table_content.strip()) > 100:
                contents.append(("pib_tables", table_content, len(table_content)))
    except Exception:
        pass
    skip_markdown_domains = any([
        is_newindianexpress_url(url),
        is_toi_url(url),
        is_downtoearth_url(url),
    ])

    if not skip_markdown_domains and hasattr(result, 'markdown') and result.markdown:
        if hasattr(result.markdown, 'fit_markdown') and result.markdown.fit_markdown:
            contents.append(("fit_markdown", result.markdown.fit_markdown, len(result.markdown.fit_markdown)))
        if hasattr(result.markdown, 'raw_markdown') and result.markdown.raw_markdown:
            contents.append(("raw_markdown", result.markdown.raw_markdown, len(result.markdown.raw_markdown)))
        if isinstance(result.markdown, str):
            contents.append(("markdown_str", result.markdown, len(result.markdown)))

    if hasattr(result, 'cleaned_html') and result.cleaned_html:
        try:
            soup = BeautifulSoup(result.cleaned_html, 'lxml')

            body = soup.find('body') or soup.find('article') or soup

            for tag in body.find_all([
                'script', 'style', 'nav', 'footer', 'header',
                'aside', 'form', 'button'
            ]):
                tag.decompose()

            extracted_parts = []

            title_tag = body.find(['h1', 'h2', 'title'])
            if title_tag:
                title_text = title_tag.get_text(" ", strip=True)
                if title_text:
                    extracted_parts.append(title_text)

            for p in body.find_all('p'):
                text = p.get_text(" ", strip=True)
                text = re.sub(r'\s+', ' ', text).strip()
                text = re.sub(r'\[\d+\]', '', text)

                if text and len(text) > 30:
                    extracted_parts.append(text)

            cleaned_text = "\n\n".join(extracted_parts).strip()

            if cleaned_text and len(cleaned_text) > 150:
                contents.append(("generic_cleaned_text", cleaned_text, len(cleaned_text)))
            else:
                contents.append(("cleaned_html", result.cleaned_html, len(result.cleaned_html)))

        except Exception:
            contents.append(("cleaned_html", result.cleaned_html, len(result.cleaned_html)))

    contents.sort(key=lambda x: x[2], reverse=True)
    for content_type, content, length in contents:
        if content and length >= 100:
            return content.strip()
    return ""


import httpx


async def fetch_with_fallback(url: str) -> str:
    """Fetch raw page HTML via public cache/archive endpoints when direct fetch fails."""
    candidates = [
        f"https://webcache.googleusercontent.com/search?q=cache:{url}",
        f"https://archive.org/wayback/available?url={url}",
        f"https://12ft.io/api/proxy?q={url}",
    ]

    async with httpx.AsyncClient(
        timeout=20.0,
        follow_redirects=True,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
    ) as client:
        for fallback_url in candidates:
            try:
                resp = await client.get(fallback_url)
                if resp.status_code == 200 and resp.text and len(resp.text) > 500:
                    return resp.text
            except Exception:
                continue

    return ""


async def fetch_article(url: str, prompt: str = "") -> str:
    try:
        browser_config = BrowserConfig(
            headless=True,
            text_mode=False,
            light_mode=True,
            extra_args=[
                "--disable-blink-features=AutomationControlled",
                "--no-first-run",
                "--disable-background-timer-throttling",
                "--disable-renderer-backgrounding",
                "--disable-backgrounding-occluded-windows",
                "--disable-web-security",
                "--disable-features=VizDisplayCompositor",
                "--no-sandbox",
                "--disable-dev-shm-usage"
            ]
        )

        if is_pib_url(url):
            crawler_config = CrawlerRunConfig(
                markdown_generator=DefaultMarkdownGenerator(),
                excluded_tags=["nav", "footer", "header", "script", "style"],
                only_text=True,
                cache_mode=CacheMode.BYPASS,
                remove_overlay_elements=True,
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                page_timeout=65000,
                delay_before_return_html=9,
                wait_for="body",
                js_code=[
                    "window.scrollTo(0, document.body.scrollHeight);",
                    "await new Promise(resolve => setTimeout(resolve, 2500));"
                ]
            )
        elif is_bbc_url(url):
            crawler_config = CrawlerRunConfig(
                markdown_generator=DefaultMarkdownGenerator(),
                excluded_tags=[
                    "nav", "footer", "header", "script", "style",
                    "aside", "form", "ads"
                ],
                only_text=True,
                cache_mode=CacheMode.BYPASS,
                remove_overlay_elements=True,
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                page_timeout=75000,
                delay_before_return_html=10,
                wait_for="article",
                css_selector="article",
                js_code=[
                    "window.scrollTo(0, document.body.scrollHeight);",
                    "await new Promise(resolve => setTimeout(resolve, 4000));"
                ]
            )
        elif is_idsa_url(url):
            crawler_config = CrawlerRunConfig(
                markdown_generator=DefaultMarkdownGenerator(),
                excluded_tags=[
                    "nav", "footer", "header", "script", "style",
                    "aside", "form", "ads"
                ],
                only_text=True,
                cache_mode=CacheMode.BYPASS,
                remove_overlay_elements=True,
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                page_timeout=60000,
                delay_before_return_html=9,
                wait_for="body",
                js_code=[
                    "window.scrollTo(0, document.body.scrollHeight);",
                    "await new Promise(resolve => setTimeout(resolve, 7000));"
                ]
            )
        else:
            bm25_filter = BM25ContentFilter(
                user_query=prompt or "current affairs news article",
                bm25_threshold=0.8
            )
            md_generator = DefaultMarkdownGenerator(content_filter=bm25_filter)
            crawler_config = CrawlerRunConfig(
                markdown_generator=md_generator,
                excluded_tags=["nav", "footer", "header", "form", "script", "style", "ads"],
                only_text=True,
                exclude_social_media_links=True,
                keep_data_attributes=False,
                cache_mode=CacheMode.BYPASS,
                remove_overlay_elements=True,
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                page_timeout=75000,
                delay_before_return_html=8,
                wait_for="body"
            )

        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url=url, config=crawler_config)
            if not result.success:
                raise ValueError(f"Crawling failed: {result.error_message}")
            content = extract_best_content(result, url)
            if content and len(content.strip()) >= 100:
                return content.strip()
            else:
                raise ValueError("Insufficient content extracted")
    except Exception as e:
        try:
            fallback_config = CrawlerRunConfig(
                markdown_generator=DefaultMarkdownGenerator(),
                excluded_tags=["nav", "footer", "header", "script", "style"],
                only_text=True,
                cache_mode=CacheMode.BYPASS,
                remove_overlay_elements=True,
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                page_timeout=70000,
                delay_before_return_html=10,
                js_code=[
                    "window.scrollTo(0, document.body.scrollHeight);",
                    "await new Promise(resolve => setTimeout(resolve, 3500));"
                ]
            )
            async with AsyncWebCrawler(config=browser_config) as crawler:
                result = await crawler.arun(url=url, config=fallback_config)
                content = extract_best_content(result, url)
                if content and len(content.strip()) >= 50:
                    return content.strip()
                else:
                    ultimate_config = CrawlerRunConfig(
                        markdown_generator=DefaultMarkdownGenerator(),
                        excluded_tags=[],
                        only_text=True,
                        cache_mode=CacheMode.BYPASS,
                        remove_overlay_elements=False,
                        user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                        page_timeout=90000,
                        delay_before_return_html=12,
                        js_code=[
                            "window.scrollTo(0, document.body.scrollHeight);",
                            "await new Promise(resolve => setTimeout(resolve, 4500));"
                        ]
                    )
                    async with AsyncWebCrawler(config=browser_config) as crawler2:
                        result = await crawler2.arun(url=url, config=ultimate_config)
                        content = extract_best_content(result, url)
                        if content and len(content.strip()) >= 30:
                            return content.strip()
                        else:
                            raise ValueError("No content found in ultimate fallback")
        except Exception as fallback_error:
            if "indianexpress.com" in url or "thehindu.com" in url:
                cached_html = await fetch_with_fallback(url)
                if cached_html:
                    soup = BeautifulSoup(cached_html, "lxml")
                    for tag in soup.find_all(["script", "style", "nav", "footer", "header", "aside", "form"]):
                        tag.decompose()
                    extracted = " ".join(
                        t.get_text(" ", strip=True)
                        for t in soup.find_all(["p", "h1", "h2", "h3"])
                        if t.get_text(" ", strip=True)
                    )
                    extracted = re.sub(r"\s+", " ", extracted).strip()
                    if len(extracted) >= 300 and not _is_blocked_response(extracted):
                        print(f"[FALLBACK] Cache-based extraction succeeded for {url}")
                        return extracted

            raise ValueError(f"All crawling attempts failed. Original: {str(e)}, Ultimate fallback: {str(fallback_error)}")


def generate_content(article_text: str, system_prompt: str) -> str:
    client = OpenAI(
        api_key=PPLEX_API_KEY,
        base_url="https://api.perplexity.ai"
    )
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": f"""
ARTICLE TO ANALYZE:
{article_text}

Generate comprehensive educational content following the exact template structure provided above. Use your knowledge base to add historical context, relevant data, and background information that enhances the educational value for UPSC aspirants.
"""
        }
    ]
    try:
        response = client.chat.completions.create(
            model="sonar-pro",
            messages=messages,
            temperature=0.3,
            max_tokens=3000,
        )
        return response.choices[0].message.content if response.choices[0].message.content else ""
    except Exception:
        response = client.chat.completions.create(
            model="sonar-reasoning",
            messages=messages,
            temperature=0.3,
            max_tokens=3000,
        )
        return response.choices[0].message.content if response.choices[0].message.content else ""


def clean_and_format_markdown(raw_content: str) -> str:
    """Clean and format the generated markdown content."""
    
    content = re.sub(r'```markdown\n?', '', raw_content)
    content = re.sub(r'\n?```', '', content)
    content = content.replace("'", "'")
    content = re.sub(r'^#{1,6}\s*', '##### ', content, flags=re.MULTILINE)
    content = re.sub(r'https?://\S+', '', content)
    content = re.sub(r'\[(.*?)\]\([^)]*\)', r'\1', content)
    content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
    return content.strip()

def extract_topic_from_content(content: str) -> str:
    """Extract topic name from the generated content."""
    lines = content.split('\n')
    for line in lines:
        if line.strip().startswith('#####') and line.strip() != '#####':
            topic = line.replace('#####', '').strip()
            if topic and len(topic) < 100:
                return topic
    return "Current_Affairs_Topic"


from typing import Tuple

def split_markdown_sections(md: str) -> Tuple[str, str, str, str]:
    """
    Split markdown into (title, subtitle, remaining_content, keywords).
    Handles optional blank line(s) between title and subtitle.
    Assumes keywords section starts with '##### Keywords'.
    Keywords are returned as a single comma-separated string (as-is).
    """
    lines = md.splitlines()

    # strip leading/trailing blank lines
    while lines and lines[0].strip() == "":
        lines.pop(0)
    while lines and lines[-1].strip() == "":
        lines.pop()

    title, subtitle, remaining_content, keywords = "", "", "", ""

    if not lines:
        return "", "", "", ""

    # always take the first non-empty line as title
    title = lines[0].replace("#####", "").strip()

    # find subtitle (first non-empty line after title)
    subtitle_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip():  # non-empty
            subtitle = lines[i].replace("#####", "").strip()
            subtitle_idx = i
            break

    if subtitle_idx is None:
        # no subtitle, only title
        return title, "", "", ""

    # find the keywords header — robust detection handles LLM variations
    # e.g. "##### Keywords", "##### **Keywords**", "**Keywords**", "Keywords:"
    keyword_idx = None
    for i in range(subtitle_idx + 1, len(lines)):
        line_stripped = re.sub(r'^#+\s*', '', lines[i].strip())   # remove heading markers
        line_stripped = re.sub(r'[\*_]+', '', line_stripped).strip().lower()  # remove bold/italic
        if line_stripped.startswith("keywords"):
            keyword_idx = i
            break

    if keyword_idx is not None:
        remaining_content = "\n".join(lines[subtitle_idx + 1:keyword_idx]).strip()
        # Remove trailing --- separators that precede the Keywords section
        remaining_content = re.sub(r'(\n---\s*)+$', '', remaining_content).strip()
        # Extract keyword text: strip heading markers, bold/italic, and the label itself
        header_text = re.sub(r'^#+\s*', '', lines[keyword_idx])
        header_text = re.sub(r'[\*_]+', '', header_text)
        header_text = re.sub(r'(?i)^keywords\s*:?\s*', '', header_text).strip()
        keyword_lines = []
        if header_text:
            keyword_lines.append(header_text)
        keyword_lines.extend([
            l.strip() for l in lines[keyword_idx+1:]
            if l.strip() and l.strip() != "---"
        ])
        keywords = " ".join(keyword_lines).strip()
    else:
        remaining_content = "\n".join(lines[subtitle_idx + 1:]).strip()

    return title, subtitle, remaining_content, keywords


def preprocess_url(url: str) -> str:
    url = url.strip()

    if "pib.gov.in" in url.lower():
        url = re.sub(r"https?://www\.pib\.gov\.in", "https://pib.gov.in", url, flags=re.IGNORECASE)

        prid_match = re.search(r"PRID=(\d+)", url, flags=re.IGNORECASE)

        if "PressReleaseDetail.aspx" in url and prid_match:
            url = f"https://pib.gov.in/PressReleasePage.aspx?PRID={prid_match.group(1)}"

    if "idsa.in" in url.lower():
        url = re.sub(r"https?://idsa\.in", "https://www.idsa.in", url, flags=re.IGNORECASE)

    if '/amp/' in url or url.endswith('/amp'):
        url = url.replace('/amp/', '/').replace('/amp', '')
    if '?' in url:
        base_url = url.split('?')[0]
        if 'article' in url or 'story' in url:
            return base_url
    return url


def _is_blocked_response(text: str) -> bool:
    if not text:
        return True

    lowered = text.lower()

    block_indicators = [
        "http 403",
        "403 forbidden",
        "forbidden",
        "access denied",
        "access denied",
        "request blocked",
        "blocked by",
        "your ip has been blocked",
        "your request has been blocked",
        "you are not authorized",
        "rate limit",
        "captcha",
        "robot",
        "automated access",
        "cloudflare",
        "waf",
        "access has been restricted",
        "this page could not be loaded",
        "page not found",
        "404 not found",
    ]

    if any(indicator in lowered for indicator in block_indicators):
        return True

    if len(text.strip()) < 300:
        return True

    return False


async def process_article_service(
    ca_created_date: str,
    theme: str,
    article_url: str,
    db: orm.Session,
    theme_topic: str | None = None,
    theme_subtopic: str | None = None,
):

    # 1. Fetch article
    processed_url = preprocess_url(article_url)
    domain = urlparse(processed_url).netloc
    prompt = f"news article from {domain} current affairs"
    article_text = await fetch_article(processed_url, prompt)

    if not article_text or len(article_text.strip()) < 50:
        raise ValueError("Insufficient article content retrieved")

    if _is_blocked_response(article_text):
        raise ValueError(f"Source blocked or returned error page (HTTP/403/WAF) for {processed_url}")

    system_prompt = build_system_prompt()
    generated_content = generate_content(article_text, system_prompt)

    if not generated_content:
        raise ValueError("No content generated by Perplexity")

    clean_content = clean_and_format_markdown(generated_content)

    title, subtitle, rest_content, keywords = split_markdown_sections(clean_content)
    print("Keywords:", keywords)
    parsed_date = datetime.strptime(ca_created_date, "%d-%m-%Y")

    # Take today's current time
    current_time = datetime.now().time()
    created_at = parsed_date.replace(hour=16, minute=7, second=7)

    ca_entry = CurrentAffair(
        topic=title,
        content=subtitle,
        theme=theme,
        theme_topic=theme_topic,
        theme_subtopic=theme_subtopic,
        ca_created_date=parsed_date,
        created_at=created_at,
        source=article_url,
        view_count=random.randint(7, 56)
    )
    db.add(ca_entry)
    db.flush()

    highlight = Highlight(
        id=ca_entry.id,
        current_affair_id=ca_entry.id,
        highlight_text=rest_content,
        created_at=created_at
    )
    db.add(highlight)

    # Build image URL from ca_entry.id
    image_url = f"https://yookicaimages.blob.core.windows.net/thumbnail/{ca_entry.id}.jpg"

    #Store all images in one LONGTEXT column as JSON string
    db.add(Image(
        id=ca_entry.id,
        current_affair_id=ca_entry.id,
        image_url=image_url,
        created_at=created_at
    ))

    # Process keywords
    db.add(Keyword(
        id=ca_entry.id,
        current_affair_id=ca_entry.id,
        kword=keywords,
    ))

    db.commit()

    print(f"[IMAGE] Queueing async image generation for article {ca_entry.id}")

    image_task = asyncio.create_task(
        asyncio.to_thread(
            generate_and_save_image,
            ca_entry.id,
            title,
            subtitle,
        )
    )

    return {
        "id": ca_entry.id,
        "topic": title,
        "markdown": clean_content,
        "images": [image_url]
    }
