---
name: langchain-deepagents
description: Implement and review production-minded Python agent systems using current LangChain and Deep Agents APIs, including tools, subagents, orchestration and evaluation.
---

# LangChain and Deep Agents Engineering

## Workflow

1. Inspect `pyproject.toml`, the lockfile and existing architecture.
2. Consult current official documentation for Deep Agents and LangChain.
3. Verify public imports, initialization parameters and subagent configuration against the installed versions.
4. Build a minimal smoke test before integrating the agent into the application.
5. Define typed and narrowly scoped tools with explicit descriptions and error handling.
6. Keep database access, external integrations and business calculations outside the prompt itself.
7. Use specialized subagents when they provide genuine context isolation or independent responsibilities.
8. Return structured evidence to the coordinator and validate the final response against a schema.
9. Add deterministic tests using fake models and mocked external systems.
10. Run tests after modifying the orchestration or tool contracts.

## Constraints

- Do not invent framework APIs.
- Do not leave the Deep Agents package unused.
- Do not pretend deterministic routing is a multi-agent architecture.
- Do not expose unrestricted SQL or write operations to a business-query agent.
- Do not use deprecated agent abstractions without a documented compatibility reason.
- Do not claim execution traces that the application did not observe.
- Do not expose private model reasoning; return an auditable summary of actions and evidence.
- Keep provider configuration external and models replaceable.

## Definition of done

The agent initializes successfully, invokes the intended tools, handles missing sources and invalid responses, produces schema-valid answers and passes reproducible tests.