# Share and archive datasets

Keep videos and simulation archives outside Git history. Share a versioned
snapshot as a GitHub Release attachment; commit the instructions and provenance
needed to identify it. A local folder inside this repository can still be ignored
by Git: `recordings/` and `outputs/` are already ignored.

## Publish a snapshot

1. Stop collection or choose a completed dataset that will not change during
   packaging. Run the dataset validation described in [collection.md](collection.md).
2. Include the entire native LeRobot dataset: metadata, data, videos, simulation
   states, scene snapshots and assets. Do not publish only the videos.
3. Record episode/frame counts, scene/configuration version, source commit,
   camera settings, action units, collection settings and any train/eval split.
   Note whether the working tree had uncommitted changes; a commit alone may not
   identify the actual collected scene. Preserve the per-episode scene archives.
4. Package and checksum it, then upload to a dedicated dataset release. Example
   commands from the repository root, with names replaced for your snapshot:

```bash
mkdir -p outputs/dataset-sharing/example-v1
tar -czf outputs/dataset-sharing/example-v1/example-v1.tar.gz \
  -C lerobot-tabletop/recordings example-dataset
(cd outputs/dataset-sharing/example-v1 && sha256sum example-v1.tar.gz > SHA256SUMS)
gh release create dataset-example-v1 \
  outputs/dataset-sharing/example-v1/example-v1.tar.gz \
  outputs/dataset-sharing/example-v1/SHA256SUMS \
  --title 'Dataset example v1' --notes-file /path/to/dataset-release-notes.md
```

Keep published snapshots immutable. Use a new version for additions, corrections
or a changed scene. For an attachment above GitHub's current size limit, split
it into documented parts or use dedicated dataset hosting.

## Download and verify

```bash
gh release download dataset-example-v1 --dir outputs/downloads/example-v1
(cd outputs/downloads/example-v1 && sha256sum -c SHA256SUMS)
```

Inspect archive contents before extracting. Extract only into a fresh destination;
do not merge with an existing collection or overwrite recordings. Preserve the
release notes and checksum with the downloaded snapshot.

## Archive a superseded local dataset

Move the complete dataset directory under `lerobot-tabletop/recordings/archive/`.
Keep its internal files and scene archives unchanged. Update explicit consumers
and documentation; saved training configurations can still contain old absolute
paths and require overrides. Never resume collection into the historical copy.

### Historical baseline

The old-scene clean 50-demo baseline (25,666 frames) is now stored locally at
`lerobot-tabletop/recordings/archive/sim-baseline/`. It predates the current arm
placement, cameras, paper enclosure and measured object positions. It is for
historical reproduction, not the current-scene training dataset. The randomized
recorder still uses episode 0 only as an IK seed, interpreted in its original
scene; that does not incorporate old demonstrations into new recordings.

The existing [sim-baseline-clean-50-v1 release](https://github.com/tonyli2/rl-so101/releases/tag/sim-baseline-clean-50-v1)
remains a historical snapshot. Its archive SHA256 is
`b8ea481b03115decf7c9d4cf7e312e103800794a2180986458468234765d3d34`.
It extracts as `sim-baseline/`; extract under the archive directory, into a fresh
location, if restoring it. Moving the local copy does not change that release.
