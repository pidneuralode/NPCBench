# Results

`main_results.csv` reports 30 models across L1–L6 and O1. The other CSVs give
staging, CarePath, external MRI, and physician-agreement results.

To reproduce the main table without model calls, install the package and
download the dataset's text files:

```sh
python -m pip install . huggingface_hub
hf download pengkaiScience/NPCBench --repo-type dataset --local-dir data \
  --include '*.json' '*.jsonl'
python scripts/reproduce_main_results.py --data data --output reproduced_results.csv
```

The script checks all 327 reported scores. Image archives are not needed for
this calculation.

`final_predictions.jsonl.gz` contains 115,056 parsed predictions with anonymous
question IDs. `criterion_judgments.jsonl.gz` contains 51,780 Boolean O1 judgments
from GPT-OSS-120B. These compact files retain the fields used for main-table
scoring; they omit prompts, clinical narratives, reasoning, and judge explanations.
Empty final predictions remain in the denominator and receive no credit.

L2 averages four capability groups, L4 averages patients, and L5 reports both
decision accuracy and all-turn success. CarePath scores 217 care decisions and
joint success over 100 patients. O1 averages the normalized scores of 103
consultations. T400 combines 100 structure-card and 300 plane-group cases;
N400 uses 400 separate cases. The dataset card and paper describe these inputs.
