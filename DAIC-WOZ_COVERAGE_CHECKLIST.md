# DAIC-WOZ Coverage Checklist

Source compared:
- `/Users/sukhmanichhabra/Downloads/DAIC-WOZ/extracted`

Target compared:
- `data/daic_woz`

Result:
- Source patient folders found: 140
- Repo `audio/` files: 140
- Repo `transcripts/` files: 140
- Repo dataset now matches the source patient coverage for audio and transcript content.

Notes:
- `379_P ` exists in the source tree with a trailing space in the folder name and was normalized during the sync.
- `labels_full.csv` is the training/evaluation manifest and covers the full patient set.