# Processed UAVDT exploratory check

This directory records the exact evidence boundary used for Table 6 of the manuscript.

## Dataset

- Source archive: `UAVDT.zip`, uploaded by Yaosheng Hou to Zenodo on 2024-12-30.
- Persistent identifier: <https://doi.org/10.5281/zenodo.14575517>
- Archive split: 1266 train, 271 validation, and 272 test images.
- Classes: car, truck, and bus.
- The archive contains YOLO-format labels and is a processed derivative, not the official sequence-disjoint evaluation package.

## Split audit

- One validation image was byte-identical to a training image: train `dhi_05500301_1.jpg` and validation `dhi_05500251_1.jpg`.
- The duplicate was removed from validation.
- The resulting manifest contains 270 images and 7014 boxes.
- Manifest: `val_clean_no_train_overlap.txt`.
- Original server-manifest SHA256: `ca394660c404f6bae798576df3b578a7df06f8cd62aed6ec0121f868f83716d9`.
- The released manifest replaces server-specific absolute prefixes with portable `./val/images/` paths.
- Released portable-manifest SHA256: `5804889885477f50547c5ef0134c01f616eb4aa1b29081562de45e5c40c68e82`.
- A separate train-test duplicate was also observed, but the test split was not used.
- Different frames with the same source-sequence names still occur across subsets. The cleaned validation set is therefore not an official sequence-disjoint UAVDT benchmark.

## Evaluation used in the paper

- Hardware: RTX 4090D.
- Training: 100 epochs, seed 0, image size 640, training batch 16, CIoU, Ultralytics 8.4.51.
- Checkpoint: final-epoch `last.pt` for all three models.
- Validation set: the deduplicated 270-image manifest.
- Reported metrics are rounded to three decimals in `uavdt_clean_final_epoch_results.csv`.
- The use of `last.pt` avoids choosing an epoch with the original contaminated validation split.

These results are reported only as an exploratory stress test. They do not support claims of official UAVDT performance or independent cross-dataset generalization.
