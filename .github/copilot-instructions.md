Follow the rules in `AGENTS.md` at the repo root (architecture invariants, conventions, infrastructure rules, workflow).
Order of work: `docs/TASK_infra.md` (Phase 0, AWS environment) first, then `docs/TASK_scenes.md`, phase by phase. Do not start a phase before the previous one's acceptance checks pass.
Key rules: only `synthrig/render`, `synthrig/assets` and `scenes/` may import `bpy`; AWS code only in `infra/`; the rig is defined only in `configs/rigs/*.yaml`;
all randomness lives in `sample_params`/`sample_sequence`; the dataset format changes additively only; never apply/destroy/submit paid AWS resources unless the user explicitly asks.
