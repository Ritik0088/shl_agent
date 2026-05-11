import json
import os
import re
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from groq import Groq

# Load catalog
with open("catalog.json") as f:
    CATALOG = json.load(f)

KEY_TO_CODE = {
    "Ability & Aptitude": "A",
    "Assessment Exercises": "E",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}

def keys_to_codes(keys):
    return ",".join(KEY_TO_CODE.get(k, "?") for k in keys)

def format_languages(langs):
    if not langs:
        return "—"
    if len(langs) <= 4:
        return ", ".join(langs)
    return ", ".join(langs[:4]) + f" _(+{len(langs)-4} more)_"

CATALOG_TEXT = "\n".join([
    f"Name: {a['name']}\n"
    f"URL: {a['link']}\n"
    f"Test Type: {keys_to_codes(a['keys'])} ({', '.join(a['keys'])})\n"
    f"Duration: {a['duration'] or '—'}\n"
    f"Languages: {format_languages(a['languages'])}\n"
    f"Job Levels: {', '.join(a['job_levels']) if a['job_levels'] else 'All levels'}\n"
    f"Remote: {a['remote']} | Adaptive: {a['adaptive']}\n"
    f"Description: {a['description']}\n"
    for a in CATALOG
])

# Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

app = FastAPI(title="SHL Assessment Agent")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str
    keys: str
    duration: str
    languages: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: Optional[List[Recommendation]]
    end_of_conversation: bool


SYSTEM_PROMPT = f"""You are an SHL assessment recommendation agent helping hiring managers select the right SHL assessments.

STRICT RULES:
1. ONLY recommend assessments from the SHL catalog below. Never invent assessments not in the catalog.
2. NEVER give general hiring advice, legal advice, or HR advice. If asked, politely decline.
3. Do NOT recommend on the first turn if the query is too vague — ask clarifying questions first.
4. Recommend between 1 and 10 assessments once you have enough context.
5. When the user refines their request, UPDATE the shortlist accordingly.
6. When comparing assessments, use ONLY catalog data.
7. Every URL must come exactly from the catalog below.
8. For senior/leadership roles, proactively include OPQ32r as the personality component unless the user objects.
9. For cognitive ability, prefer SHL Verify Interactive G+ for graduate/professional levels.
10. Be aware of language constraints — flag when key tests are English-only for non-English roles.
11. Do NOT give legal or compliance advice (e.g. whether a test satisfies a legal requirement).
12. If a specific technology has no dedicated test (e.g. Rust), say so clearly and suggest the closest alternatives.

CLARIFYING QUESTIONS TO ASK when needed:
- What is the role/job title?
- What is the seniority level? (entry-level, graduate, professional, manager, director, executive)
- Is this for selection (hiring) or development?
- What competencies matter most? (cognitive ability, personality, technical skills, leadership, situational judgment)
- Any specific technical skills required? (Java, Python, SQL, etc.)
- What language do candidates need to be assessed in?
- Is there a time constraint on the assessment battery?

TEST TYPE CODES:
A = Ability & Aptitude
B = Biodata & Situational Judgment
C = Competencies
D = Development & 360
E = Assessment Exercises
K = Knowledge & Skills
P = Personality & Behavior
S = Simulations

SHL CATALOG:
{CATALOG_TEXT}

OUTPUT FORMAT — always end your reply with exactly one JSON block:
```json
{{
  "recommendations": [
    {{
      "name": "exact name from catalog",
      "url": "exact url from catalog",
      "test_type": "comma-separated codes e.g. A,S",
      "keys": "e.g. Ability & Aptitude, Simulations",
      "duration": "e.g. 20 minutes or — if not available",
      "languages": "first 4 languages _(+N more)_ if many"
    }}
  ],
  "end_of_conversation": false
}}
```

Use `"recommendations": null` when still gathering context (not an empty array).
Set `end_of_conversation` to true only when the user has confirmed the final shortlist.
"""


def parse_response(text: str):
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        reply_text = text[:json_match.start()].strip()
        try:
            data = json.loads(json_match.group(1))
            raw_recs = data.get("recommendations")
            if raw_recs is None:
                recommendations = None
            else:
                recommendations = [
                    Recommendation(
                        name=r.get("name", ""),
                        url=r.get("url", ""),
                        test_type=r.get("test_type", ""),
                        keys=r.get("keys", ""),
                        duration=r.get("duration", "—"),
                        languages=r.get("languages", "—"),
                    )
                    for r in raw_recs
                ]
            return reply_text, recommendations, data.get("end_of_conversation", False)
        except Exception:
            pass
    return text.strip(), None, False


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in req.messages:
        messages.append({"role": m.role, "content": m.content})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=1500,
        temperature=0.3,
    )
    raw_text = response.choices[0].message.content
    reply, recommendations, end_of_conversation = parse_response(raw_text)

    return ChatResponse(
        reply=reply,
        recommendations=recommendations,
        end_of_conversation=end_of_conversation,
    )
