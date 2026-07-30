# GCALF-Net Glossary

| Term | Meaning in this repository |
|---|---|
| bpMRI | Biparametric MRI input composed of T2W, ADC, and high-b-value DWI. |
| CAF | Bidirectional windowed Q/K/V cross-attention between aligned CNN and Swin features. It does not mean BiFusion. |
| GCALF-Net | Adapted PDHD-Net using both LFF and CAF while preserving the nnDetection decoder and multi-task heads. |
| GGG | Gleason Grade Group, equivalent here to ISUP grades 1 through 5. |
| Instance class | A 0-indexed **foreground** class id in nnDetection's `labelsTr/<case>.json` `instances` dict. GGG *k* is stored as class *k−1*; there is no background class and `classifier_classes = 5`. |
| Negative case | A case with no lesion: an all-zero instance volume and `"instances": {}`. Benign PI-CAI cases (ISUP 0) are negatives, never "class 0". |
| LFF | Learnable Frequency Filter using a grouped low-resolution **real** gain grid, defined on centered frequency coordinates and interpolated to a 3D rFFT spectrum. |
| PDHD-Net baseline | The released encoder behavior: wavelet modules at stages 1, 3, and 4 plus channel-light CNN/Swin fusion at stages 2 and 5. |
| BiFusion | TransFuse gating/Hadamard fusion reference. It is not Q/K/V cross-attention and is only a named fallback. |
| Prepared dataset | The immutable nnDetection raw/preprocessed task, splits, preprocessing config, and manifest used by training. |
| Dataset manifest | SHA-256 inventory that binds data files, modality order, preprocessing config, and split file to one dataset version. |
| Run ID | Unique identity for one config, fold, seed, timestamp, and Git revision. |
| Primary matrix | The required 4 configurations x 5 official folds x 1 primary seed, totaling 20 runs. Reducible only via the pre-committed ladder in `SPEC.md §12`. |
| Durable checkpoint | A locally valid checkpoint and run state successfully synchronized to object storage. |
| S3-compatible | An object store usable through the AWS CLI S3 command contract, optionally with a custom endpoint URL. |
