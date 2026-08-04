# Security policy

## Supported versions

Security fixes are applied to the latest published minor version while the project is in alpha.

## Reporting a vulnerability

Do not open a public issue containing secrets, personal data or an exploitable proof of concept. Contact the project owner through the private security-reporting channel configured in the public repository.

## Data handling

`memnet-agent` stores the text supplied to `ask()` and `learn()` when memory updates are enabled. Applications are responsible for consent, access control, encryption, retention and deletion policies. Model prompts may contain retrieved memory excerpts; do not connect an external model provider unless its data policy is acceptable for the stored information.
