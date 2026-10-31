# Maniceet Kaggle dataset access check — 2026-10-31

## Result

- `KAGGLE_API_TOKEN` is present in GitHub Actions.
- `kagglehub.whoami(verbose=False)` succeeds: token authentication is valid.
- The same token can access the official `name-that-llm` competition through `kaggle competitions files -c name-that-llm`.
- Kaggle API access to `maniceet/name-that-llm-openrouter-openai-generations` returns HTTP 403 from both:
  - `datasets.DatasetApiService/ListDatasetFiles` via Kaggle CLI;
  - `datasets.DatasetApiService/GetDataset` via `kagglehub.dataset_download`.

Therefore the blocker is **resource-specific dataset API permission / consent**, not a missing or invalid token and not missing competition access.

## Public dataset metadata independently visible

The public Kaggle page currently reports:

- 7,753 passages;
- 10 competition labels;
- row-level provenance fields including API route, provider, served model, generator, source tag, record id and prompt lineage;
- CC BY 4.0;
- prompt filtering against competition train passages and no exact/near train duplicates reported by the publisher.

## Next action

Open the dataset page while logged in and use its Download / consent flow once if Kaggle presents one. After Kaggle grants API access to this resource, rerun `Verify Maniceet Kaggle external dataset` to perform the independent row-level audit and materialize the verified snapshot.

Until then, do not claim the 7,753-row raw file itself has been independently audited locally; only its public metadata has been verified.
