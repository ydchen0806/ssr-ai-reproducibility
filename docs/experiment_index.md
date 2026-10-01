# Manuscript-to-release index

Paths are relative to this repository. Figure numbers refer to the manuscript version recorded in `MANUSCRIPT_IDENTITY.json`. `Data_S1/MANIFEST.csv` inventories the delivered numerical files and their SHA-256 hashes.

| Manuscript item | Current source data | Scope |
|---|---|---|
| Fig. 4A | `plot_data/figure4_artwork/` | Conceptual artwork |
| Fig. 4B,E; Table S2 | `Data_S1/figure4/Fig4B_displayed_seed_level.csv`, `Fig4E_displayed_seed_level.csv`, `TableS2_original_recipe_summary.csv` | Five paired seeds per dataset |
| Fig. 4C,D,F | `Data_S1/figure4/Fig4C_complete_kernel_history.csv`, `Fig4D_displayed_seed_level.csv`, `Fig4F_all_development_and_confirmation.csv` | Ten pairs per comparison; the F artwork shows seven arms and its source table retains the shared-coefficient factorial |
| Fig. 4G; Table S5 | `Data_S1/figure4/Fig4G_displayed_transfer.csv` | Full FT+Anchor+Spectral+SSR recipe, one pair per setting; recipe-specific learning rates |
| Fig. 4H; Table S7 | `Data_S1/figure4/Fig4H_history_rerun/` | Ten paired AlphaEdit orders; both endpoints retest all 250 historical items |
| Fig. 4I | `Data_S1/figure4/Fig4I_displayed_geometry_seed_level.csv`, `Fig4I_relative_cosine_seed_level.csv` | Four dense-editing pairs per dataset |
| Fig. 5C; Table S10 | `Data_S1/figure5_rehearsal/` | Separate eight-target rehearsal sweep; ten paired 100-edit orders per rank; C shows immediate and final-history efficacy. Immediate locality remains in Table S10. |
| Fig. 5A,B,D,E,F; matched-endpoint SI table | `Data_S1/figure5_dual_followup/` | 32-target rehearsal; D shows rank-8/32 immediate and final-history locality, while A/B/E/F use rank 32. Ten pairs per rank use complete-phrase matching; rank-8 locality intervals cross zero. Geometry and trajectories are derived from the rank-32 records. |
| Fig. 5D rank 16; Table S20 | `Data_S1/figure5_rank1664_followup/` | Rank 16 uses a 32-target window and appears in D. |
| Fig. 6A--C | `Data_S1/figure6/Fig6A_displayed_seed_level.csv`, `Fig6B_displayed_stagewise_seed_level.csv`, `qualitative_case_2010_003362/` | Five paired VOC seeds and an illustrative case |
| Fig. 6D--F; visual SI | `Data_S1/figure6/Fig6D_displayed_seed_level.csv`, `Fig6D_validation_main5_summary.csv`, `Fig6E_rank16_displayed_seed_level.csv`, `Fig6E_rank32_displayed_seed_level.csv`, `Fig6F_cifar100_displayed_seed_level.csv`, `Fig6F_supplementary_nine_strengths.csv` | Five paired seeds per comparison; nine strength settings in the supplementary figure |
| Fig. 6G | `Data_S1/figure6/response_maps_rank32_seed8411/` | Matched scalar response arrays and illustrative image |

Earlier protocols remain in Git history, `plot_data/` and the linked checkpoint archive where indexed. Their G/H or Figure 5 endpoints are separate experiments and are reported separately from the current panels. The frozen training entrypoints for the current Figure 5 are `scripts/run_lowrank_ke_rehearsal.py` and `scripts/run_fig5_dual_endpoints.py`; follow-up campaigns record their frozen plans and selections in their Data S1 directories.
