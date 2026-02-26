## AuditArmor 🛡️🔍

An Intelligent Regulatory Redline Agent built for the LlamaAgents Builder Contest.

AuditArmor is a high-precision compliance agent designed to solve "Regulatory Drift"—the gap that opens when internal company policies fall out of sync with fast-changing government mandates.

## 🚀 The Technical "Flex": Beyond Basic RAG

Most agents simply summarize text. AuditArmor focuses on Spatial and Quantitative Auditing using the latest LlamaIndex stack:

- Page-Level Granularity (Feb 17 Update): Leveraging the newest LlamaParse features, AuditArmor provides bounding-box citations. It points to the exact coordinates in the document where a violation exists.

- Quantitative Conflict Detection: I configured the agent to identify numerical mismatches (e.g., catching a 72-hour reporting window that violates a 24-hour legal statute).

- Proactive Redlining: For every conflict, AuditArmor generates a "suggested_fix" field—proposing compliant text that can be immediately adopted by legal teams.

## 🛠️ Implementation Details

- Parser: LlamaParse (High-accuracy mode for messy layouts).

- Orchestration: LlamaAgents Builder.

- Extraction: Pydantic-driven LlamaExtract in src/extraction_review/config.py.

🧑‍💻 Author

Akshu Grewal
