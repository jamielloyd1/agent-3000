# agent-3000
agent-finder/
│
├── README.md
├── .gitignore
├── .env.example
│
├── backend/
│   ├── requirements.txt
│   ├── main.py                 # FastAPI entrypoint
│   ├── api/
│   │   ├── routes/
│   │   │   ├── search.py       # /search endpoint
│   │   │   ├── agents.py       # /agents endpoint (ADK + registry)
│   │   │   └── health.py
│   │   ├── models/
│   │   │   └── agent.py        # Pydantic schemas
│   │   └── services/
│   │       ├── embeddings.py   # Vertex embeddings helper
│   │       └── adk_client.py   # Google ADK integration
│   └── registry/
│       └── local_agents.json   # Local seed registry
│
├── ui/
│   ├── requirements.txt
│   └── streamlit_app.py        # MVP UI
│
├── scripts/
│   └── seed_registry.py        # Optional registry sync script
│
└── infra/
    ├── cloudrun.yaml           # Cloud Run deployment spec
    └── firestore.rules         # Firestore security rules
