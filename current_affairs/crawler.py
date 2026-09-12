import re
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode, urlparse
from bs4 import BeautifulSoup
import httpx

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.content_filter_strategy import BM25ContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator


def preprocess_url(url: str) -> str:
    """
    Standardize URL:
    - Strip trailing slashes.
    - Remove tracking and referral query parameters (?ref=..., &term=..., utm_*, fbclid, etc.).
    """
    if not url:
        return ""
    url = url.strip()
    parts = urlsplit(url)

    tracking_params = {
        "ref", "term", "fbclid", "gclid", "msclkid", "twclid", "yclid",
        "spjobid", "spmailingid", "spuserid", "cmpid", "source", "campaign"
    }

    filtered_query = [
        (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not (k.lower() in tracking_params or k.lower().startswith("utm_") or k.lower().startswith("ref_"))
    ]
    new_query = urlencode(filtered_query)
    path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme, parts.netloc, path, new_query, ""))


def is_pib_url(url: str) -> bool:
    return "pib.gov.in" in url.lower()


def is_bbc_url(url: str) -> bool:
    return "bbc.com" in url.lower() or "bbc.co.uk" in url.lower()


def is_idsa_url(url: str) -> bool:
    return "idsa.in" in url.lower()


def is_indianexpress_url(url: str) -> bool:
    return "indianexpress.com" in url.lower()


def is_thehindu_url(url: str) -> bool:
    return "thehindu.com" in url.lower()


def is_toi_url(url: str) -> bool:
    return "timesofindia.indiatimes.com" in url.lower()


def extract_toi_direct(url: str) -> str:
    """
    Extract full article text directly from TOI server-side JSON-LD without headless Chromium.
    Returns cleaned article text if articleBody >= 300 chars, else empty string.
    """
    import json
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://www.google.com/"
    }
    try:
        resp = httpx.get(url, headers=headers, follow_redirects=True, timeout=15)
        if resp.status_code != 200:
            return ""
        soup = BeautifulSoup(resp.text, "html.parser")
        for s in soup.find_all("script", type="application/ld+json"):
            if not s.string:
                continue
            try:
                data = json.loads(s.string)
                if isinstance(data, list):
                    data = data[0]
                body = data.get("articleBody")
                if not body and "@graph" in data:
                    for item in data["@graph"]:
                        if "articleBody" in item:
                            body = item["articleBody"]
                            break
                if body and len(body.strip()) >= 300:
                    headline = data.get("headline", "")
                    clean_text = f"{headline}\n\n{body.strip()}" if headline else body.strip()
                    return clean_text
            except Exception:
                continue
    except Exception:
        pass
    return ""


def _is_blocked_response(text: str) -> bool:
    """Detect if fetched content is an error or Cloudflare/WAF block page."""
    lowered = text.lower()
    block_indicators = [
        "403 forbidden",
        "http 403",
        "access denied",
        "request blocked",
        "attention required",
        "security check",
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


def _clean_html_with_bs4(html_content: str) -> str:
    """Extract readable text from HTML by removing noise tags."""
    soup = BeautifulSoup(html_content, 'lxml')
    for tag in soup.find_all([
        'script', 'style', 'nav', 'footer', 'header', 'aside', 'form',
        'iframe', 'noscript', 'meta', 'link'
    ]):
        tag.decompose()

    paragraphs = []
    for elem in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li']):
        text = elem.get_text(" ", strip=True)
        if text and len(text) > 20:
            paragraphs.append(text)

    cleaned_text = '\n\n'.join(paragraphs)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text


def _extract_pib_text(html_content: str) -> str:
    """Custom extractor for Press Information Bureau releases."""
    soup = BeautifulSoup(html_content, 'lxml')
    release_div = (
        soup.find('div', class_='innner-page-main-about-us-content-right-part')
        or soup.find('div', id='form1')
        or soup.find('div', class_='content-area')
    )
    if not release_div:
        return ""

    for tag in release_div.find_all(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form']):
        tag.decompose()

    paragraphs = []
    for p in release_div.find_all('p'):
        text = p.get_text(" ", strip=True)
        text = re.sub(r'[\r\n\t]+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        if text and len(text) > 30 and 'PIB' not in text[:10]:
            paragraphs.append(text)

    return "\n\n".join(paragraphs)


def extract_best_content(result, url: str = "") -> str:
    """Extract cleaned text using site-specific selectors or markdown fallbacks."""
    if url and is_bbc_url(url):
        try:
            html_content = getattr(result, 'cleaned_html', '') or ""
            if html_content:
                soup = BeautifulSoup(html_content, 'lxml')
                article = soup.find('article')
                if article:
                    paras = [p.get_text(" ", strip=True) for p in article.find_all('p') if len(p.get_text(" ", strip=True)) > 40]
                    if paras:
                        return "\n\n".join(paras)
        except Exception:
            pass

    if url and is_idsa_url(url):
        try:
            html_content = getattr(result, 'cleaned_html', '') or ""
            if html_content:
                soup = BeautifulSoup(html_content, 'lxml')
                clean_parts = []
                title = soup.find('h1', id='posttitle')
                if title and title.get_text(" ", strip=True):
                    clean_parts.append(title.get_text(" ", strip=True))

                article_container = soup.find('div', class_='footnote') or soup.find('div', class_='inner-content-area')
                if article_container:
                    paragraphs = []
                    for p in article_container.find_all('p'):
                        for a in p.find_all('a'):
                            a.decompose()
                        text = re.sub(r'\[\d+\]', '', p.get_text(" ", strip=True)).strip()
                        if text and len(text) > 40:
                            paragraphs.append(text)
                    clean_parts.extend(paragraphs)

                cleaned = "\n\n".join(clean_parts).strip()
                if cleaned and len(cleaned) > 100:
                    return cleaned
        except Exception:
            pass

    if url and is_indianexpress_url(url):
        try:
            html_content = getattr(result, 'cleaned_html', '') or ""
            if html_content:
                soup = BeautifulSoup(html_content, 'lxml')
                clean_parts = []
                title = soup.find('h1', id='main-heading-article')
                if title and title.get_text(" ", strip=True):
                    clean_parts.append(title.get_text(" ", strip=True))
                subtitle = soup.find('h2', class_='synopsis')
                if subtitle and subtitle.get_text(" ", strip=True):
                    clean_parts.append(subtitle.get_text(" ", strip=True))

                article_container = soup.find('div', id='pcl-full-content')
                if article_container:
                    for tag in article_container.find_all(['script', 'style', 'aside', 'figure', 'img', 'button']):
                        tag.decompose()
                    paragraphs = []
                    for p in article_container.find_all('p'):
                        text = re.sub(r'\s+', ' ', p.get_text(" ", strip=True)).strip()
                        if text and len(text) > 40:
                            paragraphs.append(text)
                    clean_parts.extend(paragraphs)

                cleaned = "\n\n".join(clean_parts).strip()
                if cleaned and len(cleaned) > 150:
                    return cleaned
        except Exception:
            pass

    if url and is_thehindu_url(url):
        try:
            html_content = getattr(result, 'cleaned_html', '') or ""
            if html_content:
                soup = BeautifulSoup(html_content, 'lxml')
                clean_parts = []
                title = soup.find('h1')
                if title and title.get_text(" ", strip=True):
                    clean_parts.append(title.get_text(" ", strip=True))

                article_container = (
                    soup.select_one('div.articlebodycontent')
                    or soup.select_one('div.storyline')
                    or soup.select_one('div.story-content')
                    or soup.find('article')
                )
                if article_container:
                    for tag in article_container.find_all(['script', 'style', 'aside', 'button', 'nav', 'header', 'footer', 'figure', 'img']):
                        tag.decompose()
                    paragraphs = []
                    for p in article_container.find_all('p'):
                        text = re.sub(r'\s+', ' ', p.get_text(" ", strip=True)).strip()
                        if text and len(text) > 40 and 'customersupport@thehindu' not in text:
                            paragraphs.append(text)
                    clean_parts.extend(paragraphs)

                cleaned = "\n\n".join(clean_parts).strip()
                if cleaned and len(cleaned) > 150:
                    return cleaned
        except Exception:
            pass

    # Generic fallbacks
    for attr in ['markdown_v2', 'markdown', 'cleaned_html', 'html']:
        if hasattr(result, attr):
            val = getattr(result, attr)
            if val:
                raw_text = val.raw_markdown if hasattr(val, 'raw_markdown') else (val if isinstance(val, str) else "")
                if raw_text and len(raw_text.strip()) > 100:
                    return raw_text.strip()

    return ""


async def fetch_with_fallback(url: str) -> str:
    """Fetch raw page HTML via Wayback Machine archive endpoint when direct fetch fails."""
    async with httpx.AsyncClient(
        timeout=25.0,
        follow_redirects=True,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
    ) as client:
        try:
            wayback_api = f"https://archive.org/wayback/available?url={url}"
            api_resp = await client.get(wayback_api)
            if api_resp.status_code == 200:
                data = api_resp.json()
                closest = data.get("archived_snapshots", {}).get("closest")
                if closest and closest.get("available") and closest.get("url"):
                    page_resp = await client.get(closest["url"])
                    if page_resp.status_code == 200 and page_resp.text and len(page_resp.text) > 500:
                        return page_resp.text
        except Exception:
            pass
    return ""


def get_default_browser_config() -> BrowserConfig:
    """Construct optimized local browser configuration for Crawl4AI."""
    return BrowserConfig(
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


async def fetch_article(url: str, prompt: str = "", crawler: AsyncWebCrawler | None = None) -> str:
    """
    Fetch and extract article text using a shared or ephemeral AsyncWebCrawler instance.
    Directly extracts standard TOI articles via JSON-LD without launching browser.
    """
    if is_toi_url(url):
        toi_text = extract_toi_direct(url)
        if toi_text and len(toi_text) >= 300:
            print(f"[SCRAPE] Extracted {len(toi_text)} chars from TOI directly via JSON-LD", flush=True)
            return toi_text

    browser_config = get_default_browser_config()

    async def _execute_crawl(cfg):
        if crawler is not None:
            return await crawler.arun(url=url, config=cfg)
        else:
            async with AsyncWebCrawler(config=browser_config) as local_crawler:
                return await local_crawler.arun(url=url, config=cfg)

    try:
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
                excluded_tags=["nav", "footer", "header", "script", "style", "aside", "form", "ads"],
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
                excluded_tags=["nav", "footer", "header", "script", "style", "aside", "form", "ads"],
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
                delay_before_return_html=4,
                wait_for="body"
            )

        result = await _execute_crawl(crawler_config)
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
                delay_before_return_html=8,
                js_code=[
                    "window.scrollTo(0, document.body.scrollHeight);",
                    "await new Promise(resolve => setTimeout(resolve, 3500));"
                ]
            )
            result = await _execute_crawl(fallback_config)
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
                    delay_before_return_html=10,
                    js_code=[
                        "window.scrollTo(0, document.body.scrollHeight);",
                        "await new Promise(resolve => setTimeout(resolve, 4500));"
                    ]
                )
                result = await _execute_crawl(ultimate_config)
                content = extract_best_content(result, url)
                if content and len(content.strip()) >= 30:
                    return content.strip()
                else:
                    raise ValueError("No content found in ultimate fallback")
        except Exception as fallback_error:
            if "indianexpress.com" in url or "thehindu.com" in url:
                cached_html = await fetch_with_fallback(url)
                if cached_html:
                    extracted = _clean_html_with_bs4(cached_html)
                    if len(extracted) >= 300 and not _is_blocked_response(extracted):
                        print(f"[FALLBACK] Cache-based extraction succeeded for {url}")
                        return extracted

            raise ValueError(f"All crawling attempts failed. Original: {str(e)}, Ultimate fallback: {str(fallback_error)}")
