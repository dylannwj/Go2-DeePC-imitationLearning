# Imitation-learning dataset

The academically sufficient dataset artifact is the canonical processed file:

```text
expert_imitation_dataset_v1.npz
```

- Accepted episodes: 120 of 120
- Transitions: 240,000
- Student state: 45 features
- Command: 3 features
- Student input: 48 features
- Expert action: 12 features
- SHA-256: `7c6da8cf8987711f099d01976b594b3119322158c3f750f4cdd71158c9c4ac99`
- Size: 246,636,478 bytes

## Distribution decision

Distribute this one processed file as a GitHub Release asset named
`expert_imitation_dataset_v1.npz`. Download that asset into
`data/imitation/`; it should not be committed to ordinary Git history. The
release repository keeps the small episode-disjoint split files under
`data/imitation/splits/`.

The original workspace also contains 120 raw episode files totaling
248,955,375 bytes, plus intermediate manifests and plots. Raw episodes are
not required to reproduce the final BC result once the canonical processed
file and split files are available, so they remain local or in a separate
provenance archive. They can be regenerated with the expert collection
pipeline if the Genesis/MJLab dependencies and licensed expert assets are
available.

## Obtain and verify

Download the release asset into this directory, then verify it before training:

```bash
sha256sum data/imitation/expert_imitation_dataset_v1.npz
```

The output must be:

```text
7c6da8cf8987711f099d01976b594b3119322158c3f750f4cdd71158c9c4ac99  data/imitation/expert_imitation_dataset_v1.npz
```

The final split is episode-disjoint: 96 episodes / 192,000 rows for train,
12 / 24,000 for validation, and 12 / 24,000 for test. The split metadata and
row-index files are normal Git-sized artifacts and must remain paired with
the canonical dataset.

The final student checkpoint records the same dataset SHA-256. Do not
recompute normalization statistics or change the split before claiming a
reproduction of the reported student result.
