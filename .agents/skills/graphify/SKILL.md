---
name: graphify
description: Query and maintain this repository's Graphify knowledge graph. Use for /graphify, codebase or architecture questions, cross-file relationships, dependency paths, focused concept explanations, and graphify query/path/explain/update requests when graphify-out exists.
---

# Graphify

Use the repository knowledge graph before broad source searches.

## Existing graph

Resolve `scripts/graphify.mjs` relative to this skill directory. The helper is read-only and runs inside the Codex sandbox.

- Natural-language question: run `node scripts/graphify.mjs query "QUESTION"`.
- Relationship trace: run `node scripts/graphify.mjs path "CONCEPT_A" "CONCEPT_B"`.
- Focused concept: run `node scripts/graphify.mjs explain "CONCEPT"`.
- Vocabulary mismatch: run `node scripts/graphify.mjs vocab`, select up to 12 exact graph terms that match the user's intent, then rerun `query` with those terms.
- Use `--dfs` for chain/dependency questions and `--budget N` to change the default 2,000-token output cap.

Answer only from the returned subgraph. Cite `source_file` and `source_location` when stating a specific fact. If the graph is insufficient, say so before inspecting source files.

## Graph mutations

Use the native `graphify` CLI for builds, updates, clustering, exports, reflection, and save-result operations. If Codex reports `uv trampoline failed to canonicalize script path` or blocks the external Python installation, rerun the same narrowly scoped Graphify command with sandbox escalation. Do not reinstall Graphify for this error.

After modifying repository code, run `graphify update .` as required by `AGENTS.md`. Do not use the read-only helper as a substitute for updates.

If no `graphify-out/graph.json` exists, use the native CLI to build it instead of attempting a fallback query.
