#!/usr/bin/env python3
"""
Test script demonstrating fallback strategy for TOI and paywalled links:
1. Standard TOI: JSON-LD direct extraction (instant, no headless browser needed)
2. Alternative Open Sources: Google/Bing/News search
3. Archive Fallbacks: Wayback Machine / Archive.today
4. LLM Web-Grounding: Perplexity sonar-pro / Luna web-grounded synthesis
"""

import sys
import json
import httpx
from bs4 import BeautifulSoup
from openai import OpenAI
from current_affairs.config import PPLEX_API_KEY

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.google.com/"
}

def extract_toi_json_ld(url: str):
    """Attempt direct extraction from TOI JSON-LD schema without headless browser."""
    print(f"\n[Attempt 0] Direct JSON-LD fetch for: {url}")
    try:
        resp = httpx.get(url, headers=HEADERS, follow_redirects=True, timeout=12)
        if resp.status_code != 200:
            return None, f"HTTP {resp.status_code}"
        
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
                
                if body and len(body.strip()) > 300:
                    headline = data.get("headline", soup.title.string if soup.title else "")
                    return {
                        "headline": headline,
                        "article_body": body.strip(),
                        "length": len(body.strip()),
                        "source": "Direct TOI JSON-LD"
                    }, None
            except Exception:
                continue
        return None, "articleBody too short or paywalled"
    except Exception as e:
        return None, str(e)


def fetch_wayback_fallback(url: str):
    """Check Wayback Machine availability for paywalled links."""
    print(f"[Attempt 2] Checking Wayback Machine for: {url}")
    api_url = f"https://archive.org/wayback/available?url={url}"
    try:
        resp = httpx.get(api_url, headers=HEADERS, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            snapshot = data.get("archived_snapshots", {}).get("closest", {})
            if snapshot.get("available"):
                return snapshot.get("url")
    except Exception as e:
        print(f"Wayback error: {e}")
    return None


def fetch_via_perplexity(headline: str, original_url: str):
    """Use Perplexity (sonar-pro) with live web grounding to synthesize full event context."""
    print(f"[Attempt 3] Web-grounded synthesis via Perplexity for: '{headline}'")
    client = OpenAI(api_key=PPLEX_API_KEY, base_url="https://api.perplexity.ai")
    
    prompt = f"""Synthesize comprehensive current affairs notes based on this news event:
Headline: {headline}
Original Reference URL: {original_url}

Focus on key facts, context, international/national implications, and perspectives from authoritative sources (PIB, UN Press, The Hindu, Indian Express).
Provide a structured synthesis suitable for UPSC / competitive exam preparation."""

    resp = client.chat.completions.create(
        model="sonar-pro",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return resp.choices[0].message.content


if __name__ == "__main__":
    row4_url = "https://timesofindia.indiatimes.com/india/strategic-autonomy-in-motion-as-india-hosts-18th-brics-summit/articleshow/133908484.cms"
    row5_url = "https://timesofindia.indiatimes.com/toi-plus/international/make-africa-bigger-shrink-north-america-why-india-backed-un-resolution-with-a-caveat/articleshow/133879847.cms"

    print("==================================================")
    print("TEST 1: Standard TOI Link (Row 4)")
    print("==================================================")
    data4, err4 = extract_toi_json_ld(row4_url)
    if data4:
        print(f"SUCCESS! Extracted {data4['length']} chars via {data4['source']}")
        print("Headline:", data4["headline"])
        print("Preview :", data4["article_body"][:250], "...\n")
    else:
        print("Failed direct extraction:", err4)

    print("==================================================")
    print("TEST 2: TOI+ Paywalled Link (Row 5)")
    print("==================================================")
    data5, err5 = extract_toi_json_ld(row5_url)
    if not data5:
        print(f"Direct fetch correctly failed as expected: {err5}")
        
        # Step 2: Wayback
        wb_url = fetch_wayback_fallback(row5_url)
        if wb_url:
            print(f"Found Wayback snapshot: {wb_url}")
        else:
            print("Wayback snapshot not found or rate-limited.")
            
        # Step 3: Perplexity Grounded Fallback
        headline5 = "Make Africa bigger shrink North America why India backed UN resolution with a caveat"
        summary = fetch_via_perplexity(headline5, row5_url)
        print("\n--- Perplexity Web-Grounded Result ---")
        print(summary[:600], "...\n")
