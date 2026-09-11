# Vendored sources

- `PLOP_SSR/`: actual SSR low-rank decoder source tree at
  `64dba29a0d5c9b84e98f0487a71d663d5bc0e647`, derived from
  https://github.com/arthurdouillard/CVPR2021_PLOP . Its MIT LICENSE is retained.
- `plop_apex_compat/`: compatibility overlay used by the completed VOC runs.
- `EasyEdit_manuscript.patch`: uncommitted AlphaEdit/SPHERE extensions from the
  run checkout over EasyEdit `14cea8245f06715684592ab55184939b99d70784`.
  Apply with `git apply` inside that pinned EasyEdit checkout. EasyEdit must be
  obtained separately from https://github.com/zjunlp/EasyEdit with its license.

No pretrained weights or dataset images are part of these source snapshots.
