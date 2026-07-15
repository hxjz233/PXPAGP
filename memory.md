# AGP Project Implementation Memory

This file is a handoff prompt for a future Codex conversation. The next conversation should use it to implement the agreed engineering plan in this repository. Do not reopen the physics discussion unless a concrete implementation ambiguity makes it necessary.

## Project context

This repository studies deformations of the PXP model using numerical exact-diagonalization-style calculations. The main quantities include AGP norms, spectral functions, level-spacing statistics, typical susceptibility, and finite-size scaling. The repository contains Python source under `src/`, human-facing Windows launchers and Slurm scripts under `scripts/`, generated figures under `fig/`, cached results under `res/`, logs under `log/`, historical artifacts under `archive/`, and literature-related material under `articles/`.

The main production entry point is `src/pxp_agp_scaling.py`, with most CLI logic in `src/pxp_agp_cli.py`. `src/pxp_agp_cli.py` currently contains several calculation modes, cache handling, parameter sweeps, and shard collection. The `src/PXP_infra/` package is treated as an existing block and should normally not be modified.

## Physics background and claims

Use the following as project background, not as facts to expand or reinterpret without literature evidence:

- The PXP model is known to be a non-integrable model containing a nearly integrable subspace.
- Recent research indicates that some perturbations, 你such as PXPZ, can enhance the integrability of the nearly integrable subspace.
- Multiple perturbations can produce non-chaotic level spacing or transport behavior, but the evidence is not yet conclusive.
- The AGP norm is a numerical quantity sensitive to long-range correlations. This project uses it to study how different deformations affect the physics of the PXP model.

When documenting or reviewing physics, distinguish carefully between:

- literature-supported claims;
- numerical evidence reproduced by this repository;
- working hypotheses;
- unresolved questions.

Do not present a finite-size numerical observation as a general integrability result without explicitly recording the relevant system sizes, symmetry sectors, normalization, observable, and limitations.

## Scope of implementation

Implement the following in stages. Keep changes incremental, reviewable, and test-backed. Preserve existing production behavior unless a change is explicitly justified by tests or a clear compatibility plan.

## Environment rule: QuSpin environment

All development, testing, smoke testing, import checks, and local numerical validation for this project must be performed inside the project’s `quspin` environment defined by `environment.yml`.

Before running Python-based checks, verify that the active interpreter belongs to the `quspin` environment and that the required packages, especially QuSpin, NumPy, SciPy-compatible linear algebra, Matplotlib, and optional CuPy/JAX backends, import successfully. Do not silently use the system Python, a different Conda environment, or an unrelated virtual environment.

This applies to:

- unit tests;
- physics exact checks;
- CLI parsing and smoke tests;
- import/type/lint checks that execute Python;
- local reproduction scripts;
- generation or validation of test fixtures;
- Wolfram-generated fixture verification when the resulting Python tests are run.

The environment check should be fast and should report the interpreter path, Python version, environment name, and key dependency versions. If the `quspin` environment is unavailable, stop before running tests and report the environment problem rather than treating the failure as a code failure. Cluster jobs should continue to activate `quspin` explicitly, as already done by the Slurm workflow.

### Stage 1: project guidance and context boundaries

Add a concise root `AGENTS.md` with:

- project purpose and terminology;
- normal commands for tests, smoke tests, and production calculations;
- the rule that behavior-changing code must have tests;
- the rule that bug fixes require regression tests;
- modification boundaries for protected or generated areas;
- logging and metadata expectations;
- review expectations for code-sensitive and physics-sensitive changes.

Add focused guidance files only where useful, for example:

- `src/PXP_infra/AGENTS.md`: existing block; do not modify unless explicitly requested;
- `scripts/AGENTS.md`: production launchers and Slurm conventions;
- `tests/AGENTS.md`: testing conventions.

Add `docs/context_boundaries.md` describing default reading priorities:

- read source, configuration, tests, and metadata first;
- normally do not scan all files in `res/`, `log/`, `fig/`, or `archive/`;
- inspect binary result files only when a task explicitly needs them;
- treat generated outputs as evidence to inspect selectively, not as source code.

Add `docs/physics_conventions.md` or equivalent only for project-local definitions: basis ordering, boundary conditions, symmetry sectors, operator conventions, AGP normalization, spectral observables, and known small-system checks. Keep this document concise and equation-oriented.

### Stage 2: unit tests and smoke tests

Use a standard Python test framework, preferably `pytest` if it fits the existing environment. Organize tests into:

- `tests/unit/`: pure functions and serialization;
- `tests/physics/`: small exact Hamiltonian/operator checks;
- `tests/smoke/`: low-cost end-to-end CLI checks;
- `tests/fixtures/`: small, human-readable or compact oracle data.

Prioritize tests for:

- perturbation and Hamiltonian term construction;
- OBC/PBC behavior;
- cache path construction and parameter completeness;
- cache save/load round trips;
- shard splitting and merging;
- spacing-ratio and spectral-weight functions;
- Hermiticity and basis dimensions;
- CPU/GPU agreement where available;
- small-system AGP consistency against an independent calculation;
- CLI parsing, cache reuse, `--force`, and tiny runs for each mode.

Tests should validate actual implementation rather than duplicating the implementation logic. Do not require large production calculations or a cluster for the default test suite.

Use the following policy: every behavior change needs a relevant test or an explicit reason why no test is applicable; every bug fix needs a regression test; documentation-only changes need normal test execution but not necessarily new tests.

### Stage 3: structured, progressive logging

Replace scattered user-facing `print` calls gradually; do not perform a risky all-at-once logging rewrite.

Use concise human-readable console output plus structured run records. Each meaningful run should produce a summary containing at least:

- run identifier;
- Git commit or equivalent code version;
- command and effective parameters;
- model/operator, system sizes, sector, boundary, and backend;
- environment/version information when practical;
- cache paths and output paths;
- elapsed time;
- final status and warnings.

Prefer a progressive-reading layout such as:

```text
log/runs/<run-id>/summary.json
log/runs/<run-id>/events.jsonl
```

Agents should read `summary.json` first and inspect detailed events only when diagnosing a failure or anomaly. Do not make routine tasks scan all historical logs. Generated logs should remain excluded from normal source context and version control where appropriate.

### Stage 4: protected code and safe hooks

Mark `src/PXP_infra/` as protected legacy/existing code. Add a lightweight safety hook that detects modifications to protected paths and requires explicit user intent or a clear override.

Use hooks only for mechanical, fast, deterministic checks, such as:

- blocking accidental edits to protected directories;
- detecting secrets or suspicious local paths;
- detecting oversized generated files;
- running import checks, formatting checks, or the fast smoke suite;
- checking that result metadata is present.

Do not use hooks to launch full-scale diagonalization, delete caches, alter physical parameters, or make scientific judgments.

### Stage 5: review protocol

Add `docs/review_protocol.md` and define two read-only review roles.

#### Code reviewer

Review implementation diffs, tests, numerical stability, performance, memory use, cache compatibility, deterministic behavior, CLI compatibility, and protected-file violations.

#### Physics reviewer

Trigger only for changes involving Hamiltonians, perturbations, boundary conditions, symmetry sectors, AGP formulas, normalization, spectral observables, finite-size scaling, or claims based on new results.

The physics reviewer must receive an evidence package containing:

- the model and convention used;
- the precise claim being evaluated;
- parameters, system sizes, sectors, and backend;
- relevant tests and independent checks;
- run metadata;
- plots or numerical summaries;
- known limitations and alternative interpretations.

The physics reviewer must distinguish a coding error from an inconclusive physical result. Reviewers are read-only and should report findings rather than silently changing code.

Do not assume that a generic language model has project-specific quantum-physics knowledge. Ground physics review in `docs/physics_conventions.md`, explicit equations, tests, and cited literature.

### Stage 6: literature workflow

Do not load an entire literature archive by default. Introduce a layered literature package:

```text
references.md
references.bib
claims.md
articles/
  papers/
  excerpts/
```

Start with a bibliography and short metadata for each paper. For each important paper, record:

- authors, title, DOI or arXiv link;
- why it matters to this project;
- sections, equations, or figures that matter;
- claims supported by the paper;
- limitations or unresolved points.

Prefer public links and targeted excerpts before full PDFs. Provide full papers only when exact formulas, notation, symmetry-sector details, or figure reproduction require them.

Maintain a claim ledger in `claims.md` with labels such as:

- `literature-supported`;
- `numerically-supported-here`;
- `hypothesis`;
- `unresolved`.

The current physics motivation should be represented using the claims in this file: PXP is non-integrable but contains a nearly integrable subspace; perturbations such as PXPZ may enhance its integrability; non-chaotic level statistics or transport may occur for several perturbations but remain inconclusive; AGP norm is used as a long-range-correlation-sensitive diagnostic of deformations.

### Stage 7: Wolfram/MCP oracle pilot

If a compatible Mathematica, Wolfram|One, or Wolfram Engine installation is available, pilot the official Wolfram Local MCP for small exact calculations. Use it as an independent symbolic/numerical oracle, not as a runtime dependency of ordinary tests.

Store reproducible plain-text Wolfram Language calculations and exported fixtures, for example:

```text
references/wolfram/*.wl
references/wolfram/*.md
tests/fixtures/wolfram/*.json
```

Use it for small system sizes, exact arithmetic, basis construction, matrix elements, traces, commutators, eigenvalues, and AGP checks. Explicitly document basis ordering and the map between Wolfram and QuSpin representations.

Normal Python tests must run without an MCP server or Mathematica license. The Wolfram process should generate or audit fixtures; it should not be required for every test invocation.

### Stage 8: CLI and scripts

Do not immediately introduce Make or a large CLI rewrite. Existing files under `scripts/` are human-facing production and reproduction launchers.

First improve only clear weaknesses:

- remove or centralize hard-coded environment paths where practical;
- avoid duplicated help text drifting from the Python parser;
- preserve existing launcher names for human use;
- keep reproduction scripts as explicit paper-figure entry points;
- separate Slurm resource settings from experiment parameters over time.

The current production CLI should remain focused on producing scientific results. Development actions such as tests, reviews, linting, and oracle generation should live in test tools, hooks, or development scripts, not as unrelated production subcommands.

Only consider mode-specific CLI subcommands after tests stabilize the existing behavior and only if the mode-specific argument space continues to grow.

### Stage 9: custom skill decision

Do not create many project-specific skills immediately. Existing scripts are better for deterministic command execution, and `AGENTS.md` is better for durable project rules.

After the review protocol has been used several times, consider one focused skill such as `physics-validation`. It should help an agent decide which tests, independent oracle checks, metadata, and reviewers are needed for a physics-sensitive change. Do not turn simple execution of an existing batch file into a skill.

## Non-goals

- Do not rewrite the physics model without explicit instruction.
- Do not modify `src/PXP_infra/` by default.
- Do not make large production calculations part of ordinary tests or hooks.
- Do not treat level-spacing behavior alone as proof of integrability.
- Do not require Mathematica, Wolfram MCP, OpenScience, or external services for normal unit tests.
- Do not integrate OpenScience into the main repository workflow yet.
- Do not add unrelated connectors such as Slack, Notion, Drive, Figma, or email unless the project workflow later requires them.

## Recommended implementation order

1. Add `AGENTS.md`, context boundaries, protected-directory guidance, and the minimal docs skeleton.
2. Add pure-function and small-system tests.
3. Add smoke tests for the current production CLI.
4. Add progressive run summaries and structured events.
5. Add lightweight safety hooks.
6. Add the code-review protocol and run a first review.
7. Add physics conventions, claim tracking, and the physics-review protocol.
8. Pilot Wolfram MCP and commit the first independent fixtures.
9. Improve launcher portability and help-text maintenance.
10. Reassess CLI subcommands and a focused `physics-validation` skill only after real usage.

## Handoff rule for the next conversation

Before editing, inspect the current repository state and preserve unrelated user changes. Implement one stage at a time, run the smallest relevant verification after each stage, and report which tests and checks were run. Do not silently expand the scope into new physics research.
