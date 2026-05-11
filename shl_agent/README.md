# SHL Assessment Recommendation Agent

A conversational AI agent that helps hiring managers select the right SHL assessments.

## How It Works
1. User describes the role they are hiring for
2. Agent clarifies if needed (role, level, skills required)
3. Agent recommends 1-10 SHL assessments from the catalog
4. User can refine or ask for comparisons

## API Endpoints
- `GET /health` → `{"status": "ok"}`
- `POST /chat` → conversational agent response with recommendations

## Setup Locally

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your Anthropic API key
```bash
export ANTHROPIC_API_KEY=your_key_here
```

### 3. Run the server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 4. Test it
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "I am hiring a Java developer"}]}'
```

## Deploy on Render (Free)

1. Push this folder to a GitHub repo
2. Go to https://render.com → New Web Service
3. Connect your GitHub repo
4. Set:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add Environment Variable: `ANTHROPIC_API_KEY = your_key`
6. Deploy!

## Project Structure
```
shl_agent/
├── main.py          # FastAPI app + agent logic
├── catalog.json     # SHL assessment catalog (31 assessments)
├── requirements.txt # Python dependencies
├── Procfile         # For deployment
└── README.md        # This file
```

## Test Types
- A = Ability / Cognitive
- P = Personality
- K = Knowledge / Skills
- B = Behavior / SJT
- S = Simulation / Coding
