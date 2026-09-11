import json
import re
from openai import AzureOpenAI, OpenAI
from sqlalchemy import orm, text

from current_affairs.config import (
    CA_GEN_AZURE_OPENAI_API_KEY,
    CA_GEN_AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    MODEL_GENERATOR,
    PPLEX_API_KEY,
    TAXONOMY_TIMEOUT,
    DEFAULT_MAX_RETRIES,
)


def classify_topic_subtopic(
    db: orm.Session,
    theme: str,
    title: str,
    subtitle: str,
    keywords: str,
    existing_topic: str | None = None,
) -> tuple[str, str]:
    """
    Classify missing topic/subtopic by querying mtr_subjects, mtr_topics, and mtr_subtopics.
    Uses synthesized article notes (title, subtitle, keywords) for high accuracy and minimal token usage.
    Powered primarily by gpt-5.6-luna on Azure OpenAI with fallback to sonar-pro.
    """
    try:
        theme_lower = (theme or "").lower()
        mapping = {
            "ir": ["International Relations", "Internal Security"],
            "security": ["Internal Security", "International Relations"],
            "polity": ["Polity", "Governance"],
            "governance": ["Governance", "Polity"],
            "economy": ["Economy"],
            "geography": ["Geography"],
            "environment": ["Environment/Disaster Management"],
            "disaster": ["Environment/Disaster Management"],
            "science": ["Science & Tech"],
            "tech": ["Science & Tech"],
            "society": ["Society/Social Justice"],
            "social": ["Society/Social Justice"],
            "history": ["Modern History", "Post Independence", "World History"],
            "ethics": ["Ethics"],
            "amac": ["AMAC"],
            "art": ["AMAC"],
            "culture": ["AMAC"],
        }
        matched_subjects = set()
        for k, v in mapping.items():
            if k in theme_lower:
                matched_subjects.update(v)

        rows = []
        if existing_topic and str(existing_topic).strip():
            clean_top = existing_topic.strip()
            # Try exact matches and punctuation variants (hyphen vs en-dash, ampersand, plural)
            variants = [
                clean_top,
                clean_top.replace("-", "–"),
                clean_top.replace("–", "-"),
                clean_top.replace(" & ", "s & "),
                clean_top.replace("s & ", " & "),
                clean_top.replace("&", "and"),
                clean_top.replace("and", "&"),
            ]
            for variant in variants:
                query = """
                    SELECT t.name as topic, GROUP_CONCAT(st.name SEPARATOR ';;;') as subtopics
                    FROM mtr_topics t
                    LEFT JOIN mtr_subtopics st ON st.topic_id = t.id
                    WHERE LOWER(t.name) = LOWER(:topic)
                    GROUP BY t.id, t.name
                """
                rows = db.execute(text(query), {"topic": variant}).fetchall()
                if rows and rows[0][1]:
                    break

            # If still not found or topic has no subtopics, try prefix LIKE
            if not rows or not rows[0][1]:
                query = """
                    SELECT t.name as topic, GROUP_CONCAT(st.name SEPARATOR ';;;') as subtopics
                    FROM mtr_topics t
                    LEFT JOIN mtr_subtopics st ON st.topic_id = t.id
                    WHERE t.name LIKE :topic
                    GROUP BY t.id, t.name
                """
                rows = db.execute(text(query), {"topic": f"%{clean_top[:8]}%"}).fetchall()

        # If existing_topic is missing OR was not found in mtr_topics, query the full theme taxonomy
        if not rows or not any(r[1] for r in rows):
            if matched_subjects:
                placeholders = ",".join(f":s{i}" for i in range(len(matched_subjects)))
                params = {f"s{i}": s for i, s in enumerate(matched_subjects)}
                query = f"""
                    SELECT t.name as topic, GROUP_CONCAT(st.name SEPARATOR ';;;') as subtopics
                    FROM mtr_subjects s
                    JOIN mtr_topics t ON t.subject_id = s.id
                    LEFT JOIN mtr_subtopics st ON st.topic_id = t.id
                    WHERE s.name IN ({placeholders})
                    GROUP BY s.id, t.id, t.name
                    ORDER BY s.disp_order, t.disp_order
                """
                rows = db.execute(text(query), params).fetchall()
            else:
                query = """
                    SELECT t.name as topic, GROUP_CONCAT(st.name SEPARATOR ';;;') as subtopics
                    FROM mtr_subjects s
                    JOIN mtr_topics t ON t.subject_id = s.id
                    LEFT JOIN mtr_subtopics st ON st.topic_id = t.id
                    GROUP BY s.id, t.id, t.name
                    ORDER BY s.disp_order, t.disp_order
                """
                rows = db.execute(text(query)).fetchall()


        taxonomy = {}
        for r in rows:
            subs = [st.strip() for st in r[1].split(";;;")] if r[1] else []
            taxonomy[r[0]] = subs

        if not taxonomy:
            return existing_topic or "General", ""

        azure_chat_client = AzureOpenAI(
            api_key=CA_GEN_AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=CA_GEN_AZURE_OPENAI_ENDPOINT,
            timeout=TAXONOMY_TIMEOUT,
            max_retries=DEFAULT_MAX_RETRIES,
        )

        prompt = f"""Given the following processed UPSC current affairs article:
Article Title: {title}
Summary: {subtitle[:400] if subtitle else ''}
Keywords: {keywords}

Select the SINGLE most appropriate Topic and Subtopic from the following allowed taxonomy list:
{json.dumps(taxonomy, indent=2)}

Return ONLY a JSON object in this exact format:
{{"theme_topic": "<Exact Topic Name>", "theme_subtopic": "<Exact Subtopic Name>"}}
"""

        try:
            resp = azure_chat_client.chat.completions.create(
                model=MODEL_GENERATOR,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.choices[0].message.content.strip()
        except Exception as luna_err:
            print(f"[TAXONOMY] {MODEL_GENERATOR} attempt failed ({luna_err}), falling back to sonar-pro...", flush=True)
            pplex_client = OpenAI(
                api_key=PPLEX_API_KEY,
                base_url="https://api.perplexity.ai",
                timeout=TAXONOMY_TIMEOUT,
                max_retries=DEFAULT_MAX_RETRIES,
            )
            resp = pplex_client.chat.completions.create(
                model="sonar-pro",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            raw = resp.choices[0].message.content.strip()

        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            chosen_topic = data.get("theme_topic", "").strip()
            chosen_subtopic = data.get("theme_subtopic", "").strip()

            matched_top = None
            for top in taxonomy:
                if top.lower() == chosen_topic.lower():
                    matched_top = top
                    break
            if not matched_top and existing_topic:
                matched_top = existing_topic

            matched_sub = None
            if matched_top and matched_top in taxonomy:
                allowed_subs = taxonomy[matched_top]
                for sub in allowed_subs:
                    if sub.lower() == chosen_subtopic.lower():
                        matched_sub = sub
                        break
            if not matched_sub:
                matched_sub = chosen_subtopic

            return matched_top or chosen_topic, matched_sub or ""
    except Exception as e:
        print(f"[TAXONOMY] Classification warning: {e}", flush=True)

    return existing_topic or "", ""
