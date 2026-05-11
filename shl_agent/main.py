import json
import os
import re
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from groq import Groq

# Load catalog
with open("catalog.json") as f:
    CATALOG = json.load(f)

CATALOG_TEXT = "\n".join([
    f"Name: {a['name']}\nURL: {a['url']}\nType: {a['test_type']}\nDescription: {a['description']}\nCategories: {', '.join(a['categories'])}\nJob Levels: {', '.join(a['job_levels'])}\n"
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

class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool

SYSTEM_PROMPT = f"""You are an SHL assessment recommendation agent. Help hiring managers select the right SHL Individual Test Solutions.

STRICT RULES:
1. ONLY recommend assessments from the catalog below. Never invent assessments.
2. NEVER give general hiring advice, legal advice, or HR advice.
3. REFUSE off-topic requests politely.
4. Do NOT recommend on the first turn if the query is vague. Ask clarifying questions first.
5. Recommend between 1 and 10 assessments once you have enough context.
6. When the user refines their request, UPDATE the shortlist.
7. When comparing assessments, use ONLY catalog data.
8. Every URL must come from the catalog below.

ASK THESE when needed:
- What is the role/job title?
- What is the seniority level? (entry, graduate, professional, manager, senior, executive)
- What competencies matter most? (cognitive, personality, technical skills, leadership)
- Any specific technical skills? (Java, Python, SQL, etc.)

SHL CATALOG:
{CATALOG_TEXT}

OUTPUT FORMAT - always end your reply with exactly one JSON block:
```json
{{
  "recommendations": [
    {{"name": "exact name from catalog", "url": "exact url from catalog", "test_type": "letter"}}
  ],
  "end_of_conversation": false
}}
```
Use empty recommendations array when still gathering context.
Set end_of_conversation to true only when task is fully complete.
"""

def parse_response(text: str):
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        reply_text = text[:json_match.start()].strip()
        try:
            data = json.loads(json_match.group(1))
            recommendations = [
                Recommendation(name=r["name"], url=r["url"], test_type=r["test_type"])
                for r in data.get("recommendations", [])
            ]
            return reply_text, recommendations, data.get("end_of_conversation", False)
        except Exception:
            pass
    return text.strip(), [], False

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
        max_tokens=1000,
        temperature=0.3,
    )

    raw_text = response.choices[0].message.content
    reply, recommendations, end_of_conversation = parse_response(raw_text)

    return ChatResponse(
        reply=reply,
        recommendations=recommendations,
        end_of_conversation=end_of_conversation
    )
