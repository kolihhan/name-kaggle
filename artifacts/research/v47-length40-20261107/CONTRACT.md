# V47 fixed-length SVC experiment

Frozen before evaluating the changed readout or querying its competition result.

Hypothesis. Holding the evidence window at 40 words can improve the transfer of the pooled-external SVC in V47. Raw answer length can reflect label-dependent generation instructions. The tested external pilot proves that shortcut in its own corpus. Membership of that exact pilot in the V47 fitting pool is unverified.

Control. Lightning V47 Kaggle script version 365000313, source SHA-256 0930f6105b42c29d6f4e8a42a5536bd63e39e64b54a78a17c93ed391342dbd6f, historical score 0.666 on submission 58008463. Use the same 30 mounted source IDs and existing fitted states. Apply one prediction change to svc4 in ntl660_readout.py. All other model votes, ensemble weights and postprocessing remain those in V47.

Treatment. The official task describes passages as 40 to 250 words. For each passage with at least 41 words, score ceil(word_count / 40) contiguous, full 40-word windows. Place the first and last windows on the boundaries and space other starts evenly using floor. Average the existing raw SVC margins over the windows before the original log-softmax and weight. Pass shorter text through verbatim. No new supervised fits, thresholds, label subsets or weight searches.

Preflight. Validate compilation, actual fitted SVC inference, mixed lengths and normalization. Repeat the same full 40-word block 1, 2 and 4 times and require the proposed readout to agree to floating point tolerance. Run the complete version on the official public fixture without the shortcut and verify its output schema, row IDs and label vocabulary within Kaggle. Reject any source-version drift or inherited supervised Gaussian refit.

Decision. The user's instruction authorizes one Kaggle comparison for this diagnostic after technical preflight. The inherited final-fit ensemble does not provide leakage-safe grouped OOF, so a score would be a one-shot comparison with V47 and would not promote a new validated model. Compare the observed official score with 0.666. A gain of at least 0.005 merits follow-up, a drop of at least 0.005 closes the variant, and anything between is inconclusive. No LB-driven changes.
