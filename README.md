# NPCBench

**A Clinical Apprenticeship Benchmark for Multimodal Care-Pathway Reasoning in Nasopharyngeal Carcinoma**

[Paper](https://openreview.net/forum?id=AVTJEREEah) · [Dataset on Hugging Face](https://huggingface.co/datasets/pengkaiScience/NPCBench)

NPCBench evaluates guideline knowledge, multimodal grounding, case planning,
longitudinal revision, care-path reasoning and specialist consultation.
This repository contains evaluation code and de-identified inference scripts.
Download the benchmark data from Hugging Face.

## Installation

Python 3.10 or later:

```sh
git clone https://github.com/pidneuralode/NPCBench.git
cd NPCBench
pip install .
pip install huggingface_hub
hf download pengkaiScience/NPCBench --repo-type dataset --local-dir data
```

For image-based tasks, extract the data archives:

```sh
tar -xzf data/L3/images_and_inputs.tar.gz -C data/L3
tar -xzf data/L6/carepath.tar.gz -C data/L6
```

## Evaluation

Save predictions as JSONL, with one question ID and model response per row:

```json
{"id": "question_id", "response": "B"}
```

For L1, L2 and L4, use the matching task and question file:

```sh
npcbench-mcq-release --task L2 --parser final_answer_v1 \
  --dataset data/L2/questions.jsonl --predictions outputs/l2.jsonl \
  --output outputs/l2_scores.json
```

L1 uses item accuracy, L2 averages four capability groups, and L4 averages
patient accuracies. Use these task-specific scores rather than a pooled accuracy.

For L3, use `Path`, `Seq`, `Struct` or `T-Stg`:

```sh
npcbench-l3 --subset T-Stg --dataset data/L3/T-Stg.jsonl \
  --predictions outputs/mri.jsonl --output outputs/mri_scores.json
```

For L5:

```sh
npcbench-l5 --dataset data/L5/questions.jsonl \
  --predictions outputs/l5.jsonl --output outputs/l5_scores.json
```

For L6, predictions use `sample_id` instead of `id`:

```sh
npcbench-l6 --dataset-root data/L6/full100_inputs_scored217 \
  --predictions outputs/l6.jsonl --policy final_answer_v1 \
  --output outputs/l6_scores.json
```

L5 reports turn accuracy and all-turn patient correctness. L6 reports care-decision
accuracy and joint correctness of pathology, MRI T stage and all selected care
answers. Full100 contains 100 patients and 217 scored care decisions; six
pending-review treatment questions are excluded from scoring.

For open-ended consultation, `npcbench-o1 --help` provides the commands to
prepare rubric tasks, generate answers, run a judge and aggregate rubric scores.
See [the data card](https://huggingface.co/datasets/pengkaiScience/NPCBench)
for rubric fields, cohorts and data limitations.

## Local model inference

The installed runner supports Qwen-VL, InternVL and Hulu-Med adapters. Install
the model's Transformers/vLLM environment separately. Example:

```sh
CUDA_VISIBLE_DEVICES=0 npcbench-infer \
  --config configs/inference/qwen35_9b.json \
  --model-path /path/to/model \
  --l6-inputs data/L6/full100_inputs_scored217 \
  --output-dir outputs/qwen35_full100
```

Set `tensor_parallel_size` in the configuration when using multiple GPUs.
Configuration files expose the generation budget, context length and sampling
settings. Use your own model paths and environment variables for API credentials;
no server login or API key is included in this repository.

## License and citation

Code is [MIT licensed](LICENSE). Original benchmark data and annotations are
[CC BY 4.0](LICENSE-DATA.md); third-party materials retain their terms.
Please cite the NPCBench paper using [CITATION.cff](CITATION.cff).
