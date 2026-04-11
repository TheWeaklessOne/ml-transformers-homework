# AGENTS.md

This file applies to the repository root and all files under it.

## Purpose of this repository

This repository is for a staged educational project about Transformers and sentiment analysis.
Work is expected to progress incrementally, day by day, with each new step building on the previous one.

## Long-term development rules

### 1. Preserve the learning progression

- Treat the project as a cumulative sequence of milestones.
- Each new day should build on earlier reusable code instead of replacing it without reason.
- Prefer extending existing modules over scattering duplicate logic across many files.
- Do not mark a milestone as complete unless code, documentation, and verification all support that claim.

### 2. Keep the repository presentation clean

- Maintain a simple, understandable history.
- Preferred policy: **one project day = one final commit**.
- Intermediate cleanup or fixup work should be folded before the day is considered complete.
- Keep the repository suitable for demonstration, review, and portfolio use.

### 3. Prefer production-style educational code

- Write code that is both teachable and maintainable.
- Prefer:
  - small functions,
  - clear names,
  - explicit validation,
  - predictable exceptions,
  - docstrings where helpful,
  - reusable helpers over notebook-only one-off logic.
- Separate core logic from CLI/demo/presentation layers.
- Avoid mixing unrelated concerns in the same function when a small abstraction would improve clarity.

### 4. Be conservative with scope

- Only implement what belongs to the current milestone unless expanding scope is clearly useful and coherent.
- Do not invent later-stage features prematurely.
- Keep roadmap/docs aligned with actual implementation.
- If something is planned but not implemented, document it as planned rather than implying it already exists.

### 5. Testing expectations

- Every implemented milestone should have verification appropriate to its scope.
- Prefer deterministic tests for helper logic.
- Keep tests fast and local when possible.
- When external models or heavyweight assets are involved, isolate local logic so it can still be tested without unnecessary network or runtime cost.
- Do not reduce coverage for already implemented behavior without a strong reason.

### 6. Dependency discipline

- Do not add new dependencies unless they clearly support the current milestone.
- Prefer the smallest dependency surface that still keeps the project professional and usable.
- If a standard-library approach is sufficient, prefer it.

### 7. Repository hygiene

Do not commit machine-local or generated artifacts unless they are intentional project deliverables.

Examples of things that should stay untracked:
- virtual environments
- editor caches
- test/tool runtime state
- browser automation artifacts
- packaging byproducts
- local datasets, unless intentionally versioned
- trained model artifacts, unless intentionally versioned

## Documentation rules

- README should describe the project in a way that remains true as the repository grows.
- Day-by-day plan/checklist documents should reflect actual progress.
- Do not encode fragile, temporary, machine-specific details in long-lived documentation unless they are required instructions.

## Code organization guidance

- Put reusable implementation code under `src/`.
- Put tests under `tests/`.
- Keep public entrypoints minimal and delegate to reusable internal functions.
- Prefer interfaces that remain useful for future milestones.

## Change discipline

Before finishing substantial work:
- ensure the implementation matches the stated milestone,
- update the relevant documentation,
- run verification,
- keep the repository history tidy.

## Instruction priority

If there is a conflict:
1. system/developer/user instructions override this file;
2. a deeper `AGENTS.md` overrides this one for files in its subtree.
