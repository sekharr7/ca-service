import re
import time
from typing import Tuple
from openai import OpenAI

from current_affairs.config import PPLEX_API_KEY, LLM_TIMEOUT, DEFAULT_MAX_RETRIES


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
[Use bullet points; focus on 2-3 essential highlights]  
##### Historical Context & Relevant Data  
[Provide bullet points on constitutional context, committee reports, historical background, statutory provisions, or key statistics]  
##### Analysis & Implications  
[Substantive paragraphs analyzing the issue, structural bottlenecks, economic or legal dimensions, and broad policy consequences]  
##### Way Forward  
[Balanced, actionable suggestions or reform measures in bullet points]  
##### Keywords: [Comma-separated relevant keywords, 3 to 7 high-value terms for UPSC answer writing]
---

**Tone, Content & Style Rules:**
1. Formal, authoritative, balanced, and objective coaching tone.
2. Ground all points in credible factual context (constitutional articles, ministries, statutory bodies, judicial verdicts).
3. Do not include introductory or concluding conversational filler.
4. Output strict level-5 markdown headers (#####).
5. Highlight critical phrases using bold or italics.
""".strip()


def generate_content(article_text: str, system_prompt: str) -> str:
    """Generate structured educational UPSC notes using Perplexity sonar-pro/sonar-reasoning."""
    t_llm = time.time()
    client = OpenAI(
        api_key=PPLEX_API_KEY,
        base_url="https://api.perplexity.ai",
        timeout=LLM_TIMEOUT,
        max_retries=DEFAULT_MAX_RETRIES,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"""
ARTICLE TO ANALYZE:
{article_text}

Generate comprehensive educational content following the exact template structure provided above. Use your knowledge base to add historical context, relevant data, and background information that enhances the educational value for UPSC aspirants.
"""
        }
    ]

    primary_err = None
    try:
        response = client.chat.completions.create(
            model="sonar-pro",
            messages=messages,
            temperature=0.3,
            max_tokens=3000,
        )
        if response.choices and response.choices[0].message.content:
            duration = round(time.time() - t_llm, 1)
            print(f"[LLM] Perplexity notes generated with sonar-pro in {duration}s", flush=True)
            return response.choices[0].message.content
    except Exception as e1:
        primary_err = e1
        print(f"[LLM] sonar-pro attempt failed ({e1}) after {round(time.time() - t_llm, 1)}s, trying sonar-reasoning...", flush=True)

    try:
        response = client.chat.completions.create(
            model="sonar-reasoning",
            messages=messages,
            temperature=0.3,
            max_tokens=3000,
        )
        if response.choices and response.choices[0].message.content:
            duration = round(time.time() - t_llm, 1)
            print(f"[LLM] Perplexity notes generated with sonar-reasoning in {duration}s", flush=True)
            return response.choices[0].message.content
    except Exception as e2:
        raise ValueError(f"Perplexity API failed on both models. sonar-pro: {primary_err} | sonar-reasoning: {e2}")

    return ""


def generate_content_from_restricted_url(url: str, headline: str = "", system_prompt: str = "") -> str:
    """
    Synthesize educational UPSC notes for paywalled/restricted articles (like TOI+)
    using gpt-5.6-luna.
    """
    from openai import AzureOpenAI
    from current_affairs.config import (
        CA_GEN_AZURE_OPENAI_API_KEY,
        CA_GEN_AZURE_OPENAI_ENDPOINT,
        AZURE_OPENAI_API_VERSION,
        MODEL_GENERATOR,
    )

    t_start = time.time()
    client = AzureOpenAI(
        api_key=CA_GEN_AZURE_OPENAI_API_KEY,
        azure_endpoint=CA_GEN_AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION,
    )

    if not headline:
        # Infer headline from URL slug if not provided
        slug = url.strip().rstrip("/").split("/")[-2] if "articleshow" in url else url.strip().rstrip("/").split("/")[-1]
        headline = slug.replace("-", " ")

    user_prompt = f"""The source news link is paywalled or restricted.
HEADLINE / EVENT TOPIC: {headline}
SOURCE URL: {url}

Synthesize comprehensive educational UPSC notes on this event following the exact required template structure.
Cover the essential facts, national/international implications, background context, and perspectives from authoritative bodies."""

    resp = client.chat.completions.create(
        model=MODEL_GENERATOR,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_completion_tokens=2500,
    )
    duration = round(time.time() - t_start, 1)
    print(f"[LLM] Notes generated via gpt-5.6-luna for paywalled URL in {duration}s", flush=True)
    return resp.choices[0].message.content or ""


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


def split_markdown_sections(md: str) -> Tuple[str, str, str, str]:
    """
    Split markdown into (title, subtitle, rest_content, keywords).
    Keywords line is stripped from rest_content.
    """
    cleaned_md = clean_and_format_markdown(md)

    kw_match = re.search(r'#####\s*Keywords:\s*(.*)', cleaned_md, re.IGNORECASE)
    if kw_match:
        keywords = kw_match.group(1).strip()
        cleaned_md = cleaned_md[:kw_match.start()] + cleaned_md[kw_match.end():]
    else:
        keywords = ""

    blocks = re.split(r'(^#####\s+.*$)', cleaned_md, flags=re.MULTILINE)
    blocks = [b.strip() for b in blocks if b.strip()]

    headers = []
    rest_parts = []
    idx = 0
    while idx < len(blocks):
        item = blocks[idx]
        if item.startswith("#####"):
            if len(headers) < 2:
                headers.append(item.replace("#####", "").strip())
                idx += 1
            else:
                body = blocks[idx + 1] if idx + 1 < len(blocks) and not blocks[idx + 1].startswith("#####") else ""
                rest_parts.append(f"{item}\n{body}")
                idx += 2 if body else 1
        else:
            rest_parts.append(item)
            idx += 1

    title = headers[0] if len(headers) > 0 else "Untitled Topic"
    subtitle = headers[1] if len(headers) > 1 else ""
    rest_content = "\n\n".join(rest_parts).strip()

    title = re.sub(r'^\[|\]$', '', title).strip()
    subtitle = re.sub(r'^\[|\]$', '', subtitle).strip()

    return title, subtitle, rest_content, keywords
