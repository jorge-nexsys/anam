# Security & Secrets Management

Secure operations and integration endpoints config.

---

## Secrets Configuration

AnamDB integrates with external LLM endpoints for natural language to logic compilation. The authentication keys (e.g., `ANAM_LLM_API_KEY`) are configured via:
* **Environment Variables:** Loaded from a local `.env` file during development startup.
* **CLI Arguments:** Passed directly using `--llm-api-key`.

---

## Future Access Control Models

As part of the roadmap, AnamDB is planning:
1. **Integration with Enterprise Secrets Managers:** Sourcing credentials directly from AWS Secrets Manager or HashiCorp Vault.
2. **Logic-Based Access Control:** Leveraging the internal Datalog engine to define role-based and attribute-based security parameters directly inside query filters.
