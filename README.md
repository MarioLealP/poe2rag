# poe2rag

Path of Exile 2 wiki RAG. Felt's poe2_wiki skill + a small web UI (poe2.clockmost.com, SSO).

Deployed on clockmost: push to `main` -> CI builds `git.clockmost.com/benylin/poe2rag:<sha>` and bumps its tag in the infra repo; the deploy loop health-checks it and auto-reverts on failure (Felt WP-53). The index (`/data`) is mounted on the server, not in this repo.

Until 2026-10-02 this code lived vendored inside the infra repo (`projects/poe2rag/app`).
