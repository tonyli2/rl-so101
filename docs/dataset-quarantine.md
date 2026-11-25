# Baseline dataset quarantine

The detected robot self-contact episodes are removed from the
active dataset without discarding their original recordings. Both resulting
directories remain native LeRobot v3 datasets, with numeric rows, videos, episode
metadata, statistics and exact MuJoCo archives kept consistent.

| Directory | Purpose | Episodes | Frames |
| --- | --- | ---: | ---: |
| `lerobot-tabletop/recordings/archive/sim-baseline/` | Archived old-scene baseline: 32 retained plus 18 replacements; not current-scene training data | 50 | 25,666 |
| `lerobot-tabletop/recordings/archived-sim-baseline-collision/` | Quarantined contact examples; do not train | 18 | 8,725 |
| `outputs/dataset-backups/sim-baseline-original-50/` | Untouched historical source; evaluation only, not training | 50 | 26,006 |

The migration retained 32 episodes / 17,281 frames. Eighteen clean replacements
have since been appended as active indices 32–49 and independently replay-audited
without detected self-contact. The migration mappings describe the original split,
not the newly appended replacements.

The authoritative audit checks every nonadjacent robot body pair during every
physics substep under the exact original archived model and controls. The
retained 32 episodes passed that rule, with exact end-state replay agreement.
This is evidence for the archived simulation/model; it does not certify changed
collision geometry or physical hardware. The initial 11 shoulder–gripper flags
were expanded to 18 once all self-contact pairs were checked.

Original zero-based episode indices quarantined:
`5, 6, 11, 19, 22, 23, 24, 25, 29, 31, 32, 33, 34, 38, 39, 45, 47, 49`.

Each subset renumbers native episode indices contiguously. Its
`episode_migration.json` preserves the original index → current index mapping,
original episode UUID, frame count and backup path. Simulator archive UUIDs and
NPZ bytes are unchanged. Archive manifest indices are remapped to match the
native dataset. Retained manifests carry
`passed_original_model_substep_audit` provenance;
quarantine manifests carry `policy_eligible=false`, an explicit
reason and a root `QUARANTINED.json` marker.

Original episode **45** is quarantined as episode **15** and keeps UUID
`cafc8bb6d9f2415bad0f4372a784cefd`. Historical checkpoint configurations were
created against the original 50-episode dataset, so reproducible old-checkpoint
evaluation should resolve their reset through the original backup or UUID
mapping, rather than treating reindexed episode numbers as original IDs. That original episode is not available in the retained active subset; active
index 45 now refers to a newly collected replacement.

## Migration implementation and verification

`lerobot-tabletop/learning/quarantine.py` uses LeRobot's existing `split_dataset`
with explicit episode lists. It refuses source datasets with shared MP4 files,
so this migration cannot silently reencode video. Staging occurs before any
source rename. Every copied video is checked by SHA256, and the existing replay
validator checks native action/state alignment and both decoded camera images
for every recorded frame. Scene bundles and array checksums remain valid.

After verification, a separate commit operation renames the untouched source to
its historical backup and installs the prepared subsets. Both paths are on the
same filesystem, with rollback if installation fails. The aggregate migration
report is `recordings/sim-baseline-quarantine-migration.json`.

For a different future source/destination set, inspect the prepare and commit
help from `lerobot-tabletop/`:

```bash
python -m learning.quarantine prepare --help
python -m learning.quarantine commit --help
```

Destinations must be unused; the utility does
not overwrite existing recordings or backups. Keep dataset directories intact
when backing them up. Git does not store these recordings.
