import re
import time
from typing import Tuple
from openai import AzureOpenAI, OpenAI

from current_affairs.config import (
    SYNTHESIS_PRIMARY_PROVIDER,
    AZURE_SYNTHESIS_MODEL,
    CA_GEN_AZURE_OPENAI_API_KEY,
    CA_GEN_AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    PRIMARY_SYNTHESIS_MODEL,
    FALLBACK_SYNTHESIS_MODEL,
    PPLEX_API_KEY,
    PPLEX_BASE_URL,
    LLM_TIMEOUT,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    DEFAULT_MAX_RETRIES,
)


def build_system_prompt() -> str:
    return """
You are an advanced content generator creating markdown material for UPSC aspirants, specifically tailored for database storage and frontend markdown rendering. Your outputs will be referenced by serious exam candidates, so maintain a formal, informative, and concise tone throughout, as if writing official coaching notes or GS exam preparation material. 

Use Bold, underline and Italics as appropriate to highlight important areas (most important don't skip this)

**CRITICAL LANGUAGE CONSTRAINT:**
All outputs must strictly and exclusively be in English. Under NO circumstances should any part of the output (Topic Name, Description, Subtitle, Section Titles, Body Text, Analysis, or Keywords) be generated in Hindi or any other vernacular language, even if the source article contains Hindi text, vernacular quotes, or auto-translation disclaimer banners. Translate all concepts into professional English.

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
    """
    Generate structured educational UPSC notes.
    Defaults to Azure OpenAI (gpt-6-luna) as primary, with Perplexity (sonar-pro/sonar-reasoning) as fallback.
    """
    t_llm = time.time()
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

    # Primary: Azure OpenAI (gpt-6-luna)
    if SYNTHESIS_PRIMARY_PROVIDER == "azure" and CA_GEN_AZURE_OPENAI_API_KEY and CA_GEN_AZURE_OPENAI_ENDPOINT:
        try:
            azure_client = AzureOpenAI(
                api_key=CA_GEN_AZURE_OPENAI_API_KEY,
                azure_endpoint=CA_GEN_AZURE_OPENAI_ENDPOINT,
                api_version=AZURE_OPENAI_API_VERSION,
                timeout=LLM_TIMEOUT,
                max_retries=DEFAULT_MAX_RETRIES,
            )
            resp = azure_client.chat.completions.create(
                model=AZURE_SYNTHESIS_MODEL,
                messages=messages,
                max_completion_tokens=LLM_MAX_TOKENS,
            )
            if resp.choices and resp.choices[0].message.content:
                duration = round(time.time() - t_llm, 1)
                print(f"[LLM] UPSC notes generated via Azure OpenAI ({AZURE_SYNTHESIS_MODEL}) in {duration}s", flush=True)
                return resp.choices[0].message.content
        except Exception as azure_err:
            print(f"[LLM] Azure OpenAI ({AZURE_SYNTHESIS_MODEL}) failed ({azure_err}) after {round(time.time() - t_llm, 1)}s, falling back to Perplexity...", flush=True)

    # Fallback / Secondary: Perplexity AI
    pplex_client = OpenAI(
        api_key=PPLEX_API_KEY,
        base_url=PPLEX_BASE_URL,
        timeout=LLM_TIMEOUT,
        max_retries=DEFAULT_MAX_RETRIES,
    )

    try:
        response = pplex_client.chat.completions.create(
            model=PRIMARY_SYNTHESIS_MODEL,
            messages=messages,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
        if response.choices and response.choices[0].message.content:
            duration = round(time.time() - t_llm, 1)
            print(f"[LLM] UPSC notes generated with Perplexity ({PRIMARY_SYNTHESIS_MODEL}) in {duration}s", flush=True)
            return response.choices[0].message.content
    except Exception as e1:
        print(f"[LLM] Perplexity ({PRIMARY_SYNTHESIS_MODEL}) failed ({e1}) after {round(time.time() - t_llm, 1)}s, trying {FALLBACK_SYNTHESIS_MODEL}...", flush=True)

    try:
        response = pplex_client.chat.completions.create(
            model=FALLBACK_SYNTHESIS_MODEL,
            messages=messages,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
        if response.choices and response.choices[0].message.content:
            duration = round(time.time() - t_llm, 1)
            print(f"[LLM] UPSC notes generated with Perplexity ({FALLBACK_SYNTHESIS_MODEL}) in {duration}s", flush=True)
            return response.choices[0].message.content
    except Exception as e2:
        raise ValueError(f"Synthesis failed across all providers: Azure ({AZURE_SYNTHESIS_MODEL}) and Perplexity ({FALLBACK_SYNTHESIS_MODEL}: {e2})")

    return ""


def generate_content_from_restricted_url(url: str, headline: str = "", system_prompt: str = "") -> str:
    """
    Synthesize educational UPSC notes for paywalled/restricted articles (like TOI+)
    using gpt-6-luna.
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
        timeout=120.0,
        max_retries=3,
    )

    if not headline:
        # Extract headline from path slug
        path_part = url.split("?")[0].rstrip("/")
        slug = path_part.split("/")[-2] if "article" in path_part.split("/")[-1] and not path_part.split("/")[-1].replace("article", "").isdigit() else path_part.split("/")[-1]
        # Clean up article numbers and extensions
        slug = re.sub(r'article\d+\.ece$', '', slug).strip('-_/')
        slug = re.sub(r'\.cms$', '', slug)
        headline = slug.replace("-", " ").strip()

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
    print(f"[LLM] Notes generated via gpt-6-luna for paywalled URL in {duration}s", flush=True)
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
    Robustly handles various LLM output patterns for title, subtitle, and badge.
    """
    cleaned_md = clean_and_format_markdown(md)

    keywords = ""
    kw_match = re.search(r'(?i)(?:^|\n)#####\s*(?:\*\*)?Keywords\s*:?\s*(?:\*\*)?\s*(.*)', cleaned_md)
    if not kw_match:
        kw_match = re.search(r'(?i)(?:^|\n)(?:\*\*)?Keywords\s*:?\s*(?:\*\*)?\s*(.*)', cleaned_md)
    if kw_match:
        keywords = kw_match.group(1).strip()
        cleaned_md = cleaned_md[:kw_match.start()] + cleaned_md[kw_match.end():]

    cleaned_md = re.sub(r'(\n---\s*)+$', '', cleaned_md).strip()

    span_tag = ""
    span_match = re.search(r'(<span class="evaluation-result-card-but">.*?</span>)', cleaned_md, flags=re.DOTALL)
    if span_match:
        span_tag = span_match.group(1).strip()
        cleaned_md = cleaned_md[:span_match.start()] + "\n" + cleaned_md[span_match.end():]

    # Find the start of the first major content section
    section_split = re.search(
        r'(?i)(^#####\s+(?:Key Takeaways|Takeaways|Historical Context|Analysis|Way Forward|Highlights|Overview).*$)',
        cleaned_md,
        flags=re.MULTILINE
    )

    if section_split:
        preamble = cleaned_md[:section_split.start()].strip()
        body = cleaned_md[section_split.start():].strip()
    else:
        headers = [m.start() for m in re.finditer(r'^#####\s+', cleaned_md, flags=re.MULTILINE)]
        if len(headers) >= 3:
            preamble = cleaned_md[:headers[2]].strip()
            body = cleaned_md[headers[2]:].strip()
        elif len(headers) >= 2:
            preamble = cleaned_md[:headers[1]].strip()
            body = cleaned_md[headers[1]:].strip()
        else:
            preamble = cleaned_md
            body = ""

    p_lines = [l.strip() for l in preamble.splitlines() if l.strip()]
    title = ""
    subtitle = ""
    for line in p_lines:
        clean_line = re.sub(r'^#####\s*', '', line).strip()
        clean_line = re.sub(r'^[\[\*\_]+|[\]\*\_]+$', '', clean_line).strip()
        clean_line = clean_line.replace('*', '').strip()
        if not title:
            title = clean_line
        elif not subtitle:
            subtitle = clean_line
            break

    if subtitle.lower().startswith(('key takeaway', 'historical context', 'analysis', 'way forward')):
        subtitle = ""

    # If subtitle is still empty, fallback: derive from first takeaway in body
    if not subtitle and body:
        first_bullet = re.search(r'[-*]\s+(.*)', body)
        if first_bullet:
            candidate = first_bullet.group(1).replace('*', '').strip()
            candidate = candidate.split('.')[0].strip()
            if len(candidate) > 20:
                subtitle = candidate

    title = re.sub(r'^[\[\*\_]+|[\]\*\_]+$', '', title).strip()
    subtitle = re.sub(r'^[\[\*\_]+|[\]\*\_]+$', '', subtitle).strip()

    rest_parts = []
    if span_tag:
        rest_parts.append(span_tag)
    if body:
        rest_parts.append(body)
    rest_content = "\n\n".join(rest_parts).strip()

    return title, subtitle, rest_content, keywords
