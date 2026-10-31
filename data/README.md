# Official competition data

Source: Kaggle — **Name That LLM**

- Competition: https://www.kaggle.com/competitions/name-that-llm
- Data page: https://www.kaggle.com/competitions/name-that-llm/data
- License shown by Kaggle: **CC BY-SA 4.0**

Official competition package files:

- `train.csv`
- `test.csv`
- `sample_submission.parquet`

## Reproduce the audited snapshot

Preferred path:

```bash
bash scripts/bootstrap_data.sh /path/to/name-that-llm.zip
```

The bootstrap verifies the official files before use. Recorded SHA-256 values from the uploaded Kaggle package:

```text
train.csv                  40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707
test.csv                   4545aa3a3459f4c0724f97ad5bd06c4caa3506bdfea5fab905e8203184fc0c57
sample_submission.parquet  ccf86796b80fd7f83d8463664603fd1208e17559abcf416f198776b819646188
```

If the local zip and Kaggle CLI are unavailable, the bootstrap may use a public GitHub copy of `train.csv` / `test.csv` only as a transport mirror. Those files were independently checked byte-for-byte against the uploaded official package and are rejected if their SHA-256 differs.

The GitHub Action `Materialize official competition data` performs the same verification before committing a snapshot to this directory.

Attribution: Yao Yan, Martyna Plomecka, and Ashley Oldacre. *Name That LLM*. Kaggle, 2026.
