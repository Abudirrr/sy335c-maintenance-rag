# Evaluation and interpretation

The source archive includes a historical `structured_evaluation.json` with five questions. Its summary agrees with the accompanying paper: mean Precision@8 0.225, step coverage 0.200, and 1.600 unmatched generated steps per question.

`app.evaluate_steps` defines relevance as the expected source file within one PDF page of the expected page. Precision is calculated over returned chunks, with page deduplication disabled. It is not document-level precision.

Generated steps are greedily matched to unused expected steps with token-overlap threshold 0.34. The denominator is the smaller token set. A match is not proof of technical correctness: shared generic words can produce false positives, and an incomplete reference list can mark valid steps as unsupported. “Hallucinated” in the saved report means unmatched by this heuristic.

Several archived answers contain formatting placeholders. The published generator has fallback filtering, but the archive does not record an exact source commit, package lock, model revision, or timestamp tying the saved results to that source version. Treat these numbers as historical evidence, not a reproducible performance guarantee.

`app.eval` compares raw hybrid retrieval, UI-filtered retrieval, and a keyword baseline on ten questions, using Hit@1/3/5 and MRR with strict and one-page-tolerant matching. `app.eval_metrics` uses the JSON test set with exact page matching and top-five metrics. These evaluators answer different questions; their output is not interchangeable.

For a defensible next benchmark, version the manuals and models, pin dependencies, separate tuning and test questions, annotate multiple relevant pages, and have a domain expert judge each generated instruction against the cited evidence.
