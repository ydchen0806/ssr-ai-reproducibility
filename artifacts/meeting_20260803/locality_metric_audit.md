# Locality metric audit

## Matched custom FT/SSR pipeline

`BioCsLLMEditor.evaluate_edit` generates an answer for every supplied locality prompt. A
locality item is scored one when any normalized ground-truth string is a case-insensitive
substring of the generated continuation. The edit-level score is the mean over all valid
locality items; an edit with no valid item receives zero. `run_sequential_editing` then averages
these edit-level scores and reports percentages.

This definition is used consistently for plain FT and every componentized SSR recipe, so those
conditions can be compared with paired seeds and identical edit order.

The evaluator was extracted from the locked implementation at commit `adb8d66`. Its source blob,
the exact `evaluate_edit` block, the normalized protocol hash and the three dataset hashes are
recorded in `locality_legacy_equivalence.json`. All 13,761 valid locality items across the three
hashed KnowEdit files use the legacy-compatible list-of-items shape; no incompatible item was
found. Formal launchers now reject other input shapes and pair results only when evaluator name,
version, protocol hash, model hash and pairing-protocol hash are identical.

## EasyEdit context pipeline

The EasyEdit wrapper passes one extracted neighborhood item per edit to `BaseEditor.edit`, then
reads values from each method's `post.locality` object. Depending on the EasyEdit method and
version, these values may represent token-level preservation, exact-match accuracy or a nested
list. The wrapper currently takes the first scalar/list entry and averages available entries.

## Decision

The two pipelines are not metric-equivalent. ROME, MEMIT and AlphaEdit remain method-level
context only. Direct SSR attribution uses custom plain/stabilized/SSR conditions evaluated by
the same function. Empty-locality handling and all-item versus first-item aggregation must be
versioned before any historical EasyEdit values are moved into a matched table.
