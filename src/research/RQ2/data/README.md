# Curated Alias Groups (Gold Set)

Provide a CSV named `aliases_review.csv` with headers:

canonical_id,phrase
CANON_WEB_DEV,web developer
CANON_WEB_DEV,frontend developer
CANON_WEB_DEV,ui engineer
CANON_DS,data scientist
CANON_DS,ml engineer

- `canonical_id`: your group identifier
- `phrase`: a surface form you consider synonymous within the group

Then run a trainer (centralized or FL contrastive) which will consume this file.

