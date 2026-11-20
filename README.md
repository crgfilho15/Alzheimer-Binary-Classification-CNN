# Alzheimer's MRI Classification: Non-Demented vs Very Mild Demented

Comparison of four CNN setups for binary classification of brain MRI slices (Non-Demented, ND, versus Very Mild Demented, VMD), with recall on the VMD class as the main metric. This is a research and learning project, not a clinical tool.

Author: Carlos Roberto Garcia (MSc Data Science and Engineering, UTAD, Portugal)

## Why recall

In screening, a missed case (false negative) costs more than a false alarm. All models are therefore compared on VMD recall first and accuracy second.

## Data

- Source: [Alzheimer's Classification Dataset (Kaggle)](https://www.kaggle.com/datasets/kanaadlimaye/alzheimers-classification-dataset), downloaded with `kagglehub`.
- Only the ND and VMD classes are used. Fixed split from the dataset: 7,574 train, 1,092 validation, 549 test images (230 VMD in the test set).
- Images are resized to 224x224 and replicated to 3 channels.
- The dataset has no patient identifiers, so a patient-level split cannot be verified. See "Limitations".

## Experiments

| Setup | Description |
| :--- | :--- |
| 1. Baseline CNN | 2 convolutional blocks, dense head |
| 2. Baseline + augmentation | Same network with rotation (20 degrees), shifts, shear, zoom and horizontal flip |
| 3. Deep CNN | 4 convolutional blocks (32 to 256 filters), batch normalization, dropout 0.5, learning rate 1e-4 |
| 4. VGG16, frozen | ImageNet weights frozen, flatten and dense head trained, no fine-tuning |

Early stopping on validation loss with best weights restored (patience 5; patience 6 for the deep CNN), seed 42, one run per setup.

## Results (test set, one run)

| Setup | Accuracy | VMD recall |
| :--- | :--- | :--- |
| 1. Baseline CNN | 0.97 | 0.94 |
| 2. Baseline + augmentation | 0.68 | 0.48 |
| 3. Deep CNN | 0.98 | 0.97 |
| 4. VGG16, frozen | 0.84 | 0.86 |

The deep CNN missed about 7 of 230 VMD cases (recall 0.97, 95% Wilson interval 0.94 to 0.985). The baseline recall of 0.94 has an interval of 0.90 to 0.96. The intervals overlap, so with one run and one seed these results do not establish that the deep CNN is better than the baseline.

These scores should be read together with the data audit in the next section: a simple nearest-neighbour baseline on tiny thumbnails matches them, so they do not demonstrate generalisation to new patients.

## Limitations

- **Possible slice-level leakage.** MRI datasets built from 2D slices often put slices of the same patient in both train and test, which inflates scores. This dataset provides no patient IDs, so it cannot be ruled out. Two observations and the audit below are consistent with it: the baseline reaches 0.97 while training accuracy is near 1.0, and mild augmentation drops accuracy from 0.97 to 0.68, which suggests the baseline relies on exact pixel positions. `leak_audit.py` measures exact duplicates, near duplicates and a 1-nearest-neighbour baseline across splits.
- **Audit result:** no exact duplicates of a training image in validation (0 of 1,092) or test (0 of 549), but 34 exact duplicates inside the training set. Every validation and test image has a near-duplicate in training (perceptual hash distance of at most 20 of 256 bits; median 4 for validation and 5 for test; heuristic threshold). A 1-nearest-neighbour classifier on 32x32 thumbnails, with no training beyond memorising the training set, reaches 99.8% accuracy and 99.6% recall for the VMD class on test (99.3% and 98.4% on validation). Without patient IDs this cannot prove leakage, but it indicates that the test set largely measures similarity to training slices, so the reported scores should not be read as generalisation to new patients. Script: `leak_audit.py`.
- **Unstable validation.** For the deep CNN, validation accuracy varies between roughly 0.80 and 0.98 across epochs even though training accuracy is above 0.99. The model with the best validation loss is restored, but the curves are not "aligned".
- **VGG16 was only tested frozen.** It was trained without fine-tuning and with default settings, so the result says that a frozen ImageNet backbone underperformed here, not that transfer learning does not work for MRI.
- **Single run.** No repeated seeds, no cross-validation, no confidence intervals beyond the one above.
- **Decision threshold fixed at 0.5.** For a recall-first goal the threshold should be tuned on validation data, and precision-recall and ROC curves reported.
- **Not clinical.** Results come from one public dataset and have no external validation.

## Reproduce

1. Open `Alzheimer_Detection_Analysis.ipynb` in Google Colab (GPU recommended).
2. Run all cells. The dataset downloads automatically through `kagglehub`.
3. Run the audit after the download cell:

```bash
python leak_audit.py --root "<dataset_path>/Alzheimers_Detection_dataset"
```

Dependencies: tensorflow, pandas, numpy, scikit-learn, matplotlib, seaborn, kagglehub, pillow.

## Repository

- `Alzheimer_Detection_Analysis.ipynb`: full pipeline and the four experiments
- `leak_audit.py`: split diagnostics
- `images/`: confusion matrices and comparison charts

## Next steps

- Evaluate on a dataset with patient identifiers and split by patient
- Repeat each setup over several seeds and report mean and interval
- Fine-tune the VGG16 backbone and compare again
- Tune the decision threshold for recall and report the precision cost
