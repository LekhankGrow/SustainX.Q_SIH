# Dataset

The training script downloads the `parkinsons.data` table from the official [UCI Parkinsons dataset](https://archive.ics.uci.edu/dataset/174/parkinsons) archive on first run. The file is not committed to this repository.

- Citation: Little, M. (2007). *Parkinsons*. UCI Machine Learning Repository. [https://doi.org/10.24432/C59C74](https://doi.org/10.24432/C59C74)
- License: CC BY 4.0
- Expected columns: 22 numeric engineered voice measures, `name` recording/person identifier, and binary `status` target.
- The downloaded file is checked for schema, target labels, missingness, subject IDs, consistent per-person labels, and a SHA-256 checksum.
- UCI's dataset page describes 195 recordings from 31 people (23 with Parkinson's), while its headline metadata lists 197 instances. The downloaded CSV in this environment has 195 rows from 32 subject IDs, including 8 healthy and 24 Parkinson's participants. The benchmark reports validated file counts and records this discrepancy rather than changing the source data.

The table contains engineered features, not raw audio. It is not compatible with the microphone flow unless a compatible feature-extraction pipeline is independently demonstrated.
