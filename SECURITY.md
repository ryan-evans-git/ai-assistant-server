# Security Policy

## Reporting a vulnerability

Please report security issues privately via GitHub's
**Security Advisories** (Security tab → "Report a vulnerability")
rather than opening a public issue. Reports are acknowledged
within 5 business days.

If you can't use GitHub Advisories, email
`ryan-evans-git` via the address listed on their GitHub profile.
PGP-encrypted reports are accepted on request.

## Supported versions

Only the latest released minor version receives security fixes.

## Scope

In scope:

- Code in this repository (`ai_assistant_server/`).
- Container images built from this repository's `Dockerfile`.
- Auth flow (`auth.py`) — anything that handles upstream API
  credentials or forwarded user credentials.

Out of scope:

- Vulnerabilities in upstream MCP / Starlette / httpx — please
  report those to the respective vendors. Dependabot tracks
  their advisories and we ship updated pins promptly.
- The OpenAPI specs themselves — they're user-supplied
  configuration, not code we ship. Sample specs in `tools/` are
  illustrative only.

## What we run on every commit

- Ruff (lint + style)
- Bandit (Python static security analysis, medium+ severity gates merge)
- pip-audit (Python dependency CVE scan)
- Trivy (filesystem + Dockerfile scan)
- CodeQL (GitHub-native SAST, `security-extended` query pack)
- Gitleaks (committed-secret detection across full history)
- Dependabot (weekly dep PRs; immediate security updates)
- Sample-spec smoke load (catches loader regressions against
  the bundled OpenAPI specs)

A failing security check blocks merge to `main`.
