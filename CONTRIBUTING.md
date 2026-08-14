# Contributing to py-daroja-libs

Thanks for contributing! This document covers the conventions and workflow for this repo.

## Commit conventions

This repo uses [Conventional Commits](https://www.conventionalcommits.org/) for PR titles and commit messages. **semantic-release** auto-tags releases from `feat:` and `fix:` prefixes only.

### Release prefixes (load-bearing)

If your PR adds a **behavior change** — new env var, new optional dep group, new public API, new config option — use a `feat(...):` prefix in the PR title, even if the change is mostly tests or documentation.

semantic-release only tags `feat:` and `fix:` prefixes. These prefixes will **not** trigger a release:

```
test:    chore:    docs:    refactor:    style:    ci:
```

If a behavior-changing PR is merged with one of those prefixes, the release doesn't fire and downstream consumers that pin to `@latest` won't see the change until someone manually triggers a release.

### Prefix rules

| Prefix | When to use | Triggers release? |
|---|---|---|
| `feat(scope):` | New feature, new API, new env var, new dep group | ✅ Yes |
| `fix(scope):` | Bug fix | ✅ Yes |
| `docs(scope):` | Documentation only | ❌ No |
| `test(scope):` | Test-only changes | ❌ No |
| `chore(scope):` | Maintenance, CI, dependencies | ❌ No |
| `refactor(scope):` | Code restructuring, no behavior change | ❌ No |

### Examples

```bash
# ✅ Correct — triggers release
feat(llm): add Azure OpenAI provider support
fix(db): accept int port in DatabaseManager.__init__

# ❌ Won't trigger release even though it adds new behavior
chore(llm): add Azure OpenAI provider support
test(db): add test for int port handling
```

## Development workflow

1. Fork or create a branch from `main`
2. Make changes, matching existing code style
3. Run the verification suite before opening a PR:

```bash
pytest -x              # unit tests (asyncio_mode = "auto")
ruff check .           # lint
ruff format --check .  # format
```

4. Open a PR with a conventional-commit title
5. Wait for CI to pass and for review

## PR title format

```
<type>(<scope>): <description>
```

Common scopes: `db`, `llm`, `a2a`, `compliance`, `vpc-runner`, `release`, `ci`

## Optional dependency groups

| Group | Install | Description |
|---|---|---|
| `psycopg3` | `pip install py-daroja-libs[psycopg3]` | PostgreSQL via psycopg 3 |
| `pgvector` | `pip install py-daroja-libs[pgvector]` | pgvector support |
| `tracing` | `pip install py-daroja-libs[tracing]` | langfuse + langsmith |
| `openai` | `pip install py-daroja-libs[openai]` | OpenAI SDK (for OpenAI + Azure providers) |
| `dev` | `pip install py-daroja-libs[dev]` | pytest + pytest-asyncio |

## Testing

```bash
# Unit tests (mock-based, no real DB needed)
pytest tests/ -v

# Stress tests (require PostgreSQL — run via CI or locally with Docker)
pytest tests/stress/ -v
```

## License

MIT — see [LICENSE](LICENSE).
