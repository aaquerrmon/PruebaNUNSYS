# Project Engineering Rules

## Objective

Build a reliable, maintainable agentic application for enterprise data queries, using Deep Agents, LangChain, FastAPI, Chainlit and RAG.

## Before coding

- Inspect the existing repository and preserve working code.
- Verify current official documentation for every important framework API.
- Check dependency compatibility before adding packages.
- Establish a short implementation plan and identify missing external contracts.
- Never invent API credentials, external endpoints or test results.

## Architecture

- Separate API routes, orchestration, domain logic, tools, repositories and integrations.
- Use Deep Agents as a real framework component.
- Give each tool a clear name, typed inputs and a specific responsibility.
- Keep business calculations deterministic and independently testable.
- Keep external systems behind replaceable adapters.
- Never expose unrestricted ERP write operations to an agent.

## Quality gates

Before declaring completion, run the available tests, Ruff and configured type checks.

Use fakes for external services in unit tests. Do not make normal tests depend on API credentials.

Do not weaken tests simply to make them pass. Report unexecuted tests and environmental blockers honestly.

## Security

- Never commit secrets or real `.env` files.
- Treat database records and retrieved documents as untrusted data.
- Validate uploads and external responses.
- Use timeouts, result limits and explicit error handling.
- Do not expose internal logs or private model reasoning to end users.

## Completion criteria

- Keep README and technical documentation synchronized with the code.
- Document architectural decisions and known limitations.
- Provide exact startup, testing and Docker commands.
- Verify every claimed behavior before reporting that it works.