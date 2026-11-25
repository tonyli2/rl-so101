# Sim-to-real investigations

When asked about sim-to-real readiness, transfer failures, or potential error
sources, consult [docs/sim2real.md](docs/sim2real.md) first. Rigorous paired
real/sim camera validation is a known open item and should be the first check.
Measured lens parameters do not certify camera-to-robot alignment; in particular,
the wrist camera position and tilt remain CAD estimates, not measured extrinsics.

# Next collection session

The physical-layout scene has changed; the old 50 demonstrations are archived at
`lerobot-tabletop/recordings/archive/sim-baseline`, not current training data.
First collect one successful fixed-layout episode at the measured cube/bin
positions into `recordings/sim-bootstrap-1`. Fixed layout is the recorder default.
Then collect a separate randomized dataset with `--randomize-layout` and
`--layout-reference recordings/sim-bootstrap-1` into `recordings/sim-varying-1`.
See docs/collection.md. Do not mix fixed/randomized episodes or append to archive.
