# Tabletop tools

From this directory in the `lerobot` Conda environment:

```bash
python launch_tabletop.py                      # Mounted single-cube scene
python launch_tabletop.py --scene upstream     # Original four-cube scene
python record_demonstrations.py --help
python replay_demonstrations.py --help
```

Start with the [project README](../README.md). Detailed instructions live in
[collection and replay](../docs/collection.md), [scene configuration](../docs/simulation.md),
and the [upstream reference](../docs/upstream-tabletop.md).

`recordings/` contains local datasets and is ignored by Git. Leave it in place
when reorganizing code; datasets include their original scene/source snapshots.
