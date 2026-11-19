# NTL Hierarchy + Hard Negative, 2026-11-19 — terminal CPU result

**Decision: REJECT the preregistered fixed C1 Hard-Negative objective.** Not a new official Kaggle result, no promotion. [Training Action #44477189113](https://github.com/kolihhan/name-kaggle/actions/runs/44477189113) completed **SUCCESS**; [full aggregate artifact #14507550021](https://github.com/kolihhan/name-kaggle/actions/runs/44477189113/artifacts/14507550021).

## Scientific contract

- [Preregistered before training](PREREG.md), commit `6465ca9`. Implementation `scripts/ntl_hier_hardneg_20261119.py`; one CPU-only action, no parameter sweeps, no Kaggle GPU or competition submission.
- Frozen official train `2,699` responses, `150` prompts, `18` labels, `5×30` disjoint prompt heldouts, known source SHA and fold manifest SHA checked in GitHub Action. **Same train-only TF-IDF/SVD256 + learnable 128-d encoder**, 120 optimizer updates per fold/model, matching initialization and input order; 15 fits total.
- C0 flat 18-way CE; CH soft family × sibling conditional heads; C1 the same CH architecture/objective plus contrastive term with same-model positives across prompts, weighted same-family negatives and matched-prompt sibling hard negatives.

## Results — reused official TRAIN grouped OOF (not official Public)

| Arm | Macro 18-class BA | Wrong cross-family | Wrong within-family |
| --- | ---: | ---: | ---: |
| C0 Flat | **0.278914** | 947 | 999 |
| CH Hierarchical only | 0.275963 | 918 | 1036 |
| C1 Hierarchical + Hard Negative | **0.273741** | 921 | 1039 |

- Primary **C1 − CH BA = −0.002222**, 95% stratified prompt-cluster CI **[−0.006667, +0.002227]**. **18 rescues, 24 regressions** relative to CH. Within-family targeted: **10 rescues, 15 new regressions**. Cross-family errors **918 → 921**.
- Secondary **C1 − C0 BA = −0.005173**, 95% CI **[−0.017778, +0.007829]**; 130 rescues / 144 regressions. Family errors decrease (947→921) but within-family errors rise (999→1039); aggregate is worse.
- Existing historical plain TF-IDF/LinearSVC hierarchy-only separate experiment on spent Oleg216: +0.009259 flat→hierarchy BA, far below predeclared +0.05 gate. [Original verified Action #43902774559](https://github.com/kolihhan/name-kaggle/actions/runs/43902774559). Not numerically comparable to this official-TRAIN OOF.
- Earlier within-prompt column cross-entropy test on the same fixed fold mapping scored C0 **0.278153**, matched **0.282598**, shuffled **0.290375**, so naïve same-prompt supervision was already a poor mechanism. [Action #44277551557](https://github.com/kolihhan/name-kaggle/actions/runs/44277551557).

## Interpretation / next research

**FACT:** Fixed, preregistered representation-level sibling-hard-negative contrastive objective decreased pooled grouped OOF BA, increased both cross-family and within-family mistakes vs CH, and did not clear +.030 positive gate or the confidence interval/family non-regression gates. The added contrastive loss does **not** provide useful incremental evidence on these prefit char features.

**Limitations:** Only a cheap char-TFIDF→SVD256 representation plus small trainable MLP, 120 update steps. This does **not** disprove hierarchical training with genuinely strong pretrained embeddings or all contrastive objectives. But **no justification for tuning this same objective on the already-reused official folds**. The true scored Apex0.702 prefit stack has no honest cross-validated OOF and was **not evaluated** in this trial. Do not claim CH/C1 can beat or rescue 0.702 based on these numbers.

**REJECT:** C1 v0; no new random seeds, hard-negative weights, temperature sweeps, more official fold recycling or blind Kaggle submissions.

**PIVOT:** Don't resume external-source hunting. Preserve verified official champion **Public BA 0.702 (#58514818)**. The remaining technically distinct path worth assessing is **selective inference on existing scored ensemble disagreements**, contingent on whether unchanged score vectors / gate inputs and safe production runtime actually exist. Unlike this failed training objective, that can target current strong model decisions rather than building a low-BA weak second classifier. No official submission authorized by this report.
