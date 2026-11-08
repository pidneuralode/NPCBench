# NPCBench

Evaluation code for NPCBench. Use it to score L1–L6 choices and structured
answers and to prepare and score O1 rubric judgments.

[Paper](https://openreview.net/forum?id=AVTJEREEah) · [Dataset](https://huggingface.co/datasets/pengkaiScience/NPCBench)

Install with Python 3.10 or later and download the inputs for your task.

```sh
pip install . huggingface_hub
hf download pengkaiScience/NPCBench L1/questions.jsonl O1/questions.jsonl --repo-type dataset --local-dir data
```

## Choice and structured answers

```sh
npcbench choice --task L1 --predictions predictions.jsonl
```

Provide one JSONL record per question, such as `{"id":"question_id","response":"B"}`.
Include all selected letters for questions with multiple correct options. For
MRI tasks, follow the answer format in the prompt. Use `sample_id` for L6.

Task names are `L1`, `L2`, `L3-Path`, `L3-Seq`, `L3-Struct`, `T400`, `N400`,
`L4`, `L5`, and `L6`. See the dataset card for image archives and task inputs.
Use `--output scores.json` to save scores.

## O1 open answers

Provide one JSONL record per consultation with `item_id` and `response`. To
format generation inputs as in the paper, use
`npcbench_eval.prompts.render_generation_messages`.

```sh
npcbench open --responses responses.jsonl --output judge_requests.jsonl
npcbench open --judgments judgments.jsonl --output scores.json
```

Run the judge requests with your judge. Each judgment must include `item_id`,
`rubric_id`, and a Boolean `criteria_met`.

Code is [MIT](LICENSE). See the [dataset license](https://huggingface.co/datasets/pengkaiScience/NPCBench/blob/main/LICENSE-DATA.md)
for data terms. Please cite the NPCBench paper.
