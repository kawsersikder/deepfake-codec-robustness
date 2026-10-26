# Results

Every number in the top-level README is read from the files in this folder. Model weights (`*.pt`, `*.pth`) and cached features (`*.npy`) are not included.

| Folder | Experiment | Notebook |
|---|---|---|
| `499a_cnn/phase07_robustness` | EfficientNet-B0 corruption robustness with bootstrap intervals | CSE 499A |
| `499a_cnn/phase11_baseline` | EfficientNet-B0, leave-one-dataset-out | CSE 499A |
| `499a_cnn/phase11_dualstream` | RGB + DCT dual-stream EfficientNet | CSE 499A |
| `499a_cnn/phase11_cam` | Attention-supervised (CAM) variant | CSE 499A |
| `499a_cnn/phase11_dann` | Domain-adversarial (DANN) variant | CSE 499A |
| `499a_cnn/phase12_ldm_probe` | Probe on latent-diffusion faces | CSE 499A |
| `phase13_foundation_lora` | CLIP, DINOv2 and EVA-02: frozen probe and LoRA | `01` |
| `phase13b_followups` | q,k,v,o variant, seeds, CNNs re-scored on the same frames | `02` |
| `phase14_codec_mask_pilot` | JPEG-table masking pilot (CNN, DFDC fold) | `03` |
| `phase14b_codec_mask_cnn` | Codec masking on the CNN: 3 folds × 8 arms | `04` |
| `gate0_compression_shift` | Compression shift of all models, shortcut audit | `05` |
| `d1_session1`, `d1_session2` | Codec-aware training of EVA-02 + LoRA | `06` |
| `d1_session3_strength` | Strength-matched masks | `07` |
| `d1_merged_analysis` | Sessions 1 + 2 combined (`analysis/d1_merge.py`) | |
| `d1_strength_analysis` | Pre-registered 2 × 2 (`analysis/d1_strength_analysis.py`) | |

## File conventions

- **`pred_<arm>_<fold>_<condition>.csv`:** one row per test video, with columns `dataset, video_id, group_id, label, probability` (label 1 = fake). `<fold>` names the held-out dataset. `<condition>` is one of `clean`, `resize224`, `jpeg50`, `jpeg20`, `h264_crf23`, `h264_crf40`.
- **`<arm>/<fold>/done.json`:** the unit's status, AUC per condition, training summary (best epoch, source-validation AUC), timings and the configuration signature used for reuse.
- **`auc*.csv`, `paired_deltas*.csv`:** AUC tables and paired cluster-bootstrap differences (`delta = AUC(first) − AUC(second)` on identical videos, 1,000 resamples).
- **`consistency.csv`:** the unchanged Phase 13 checkpoint re-scored in each session; it must reproduce the Gate 0 AUCs.
- **`strength_calibration.csv`:** per fold, every mask table's scale, mean drop probability, coefficients dropped per image, removed energy at native size and after the 224 px resize, and PSNR.
- **`probe/bank_eva02_l14.csv.gz`:** the frame list (path, split, labels) that defines a session's data sample. Kept for session 1 (also used by Phase 13 and session 2) and for session 3.
