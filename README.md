# NPCBench

NPCBench evaluates guideline knowledge, image interpretation, clinical decisions,
and specialist consultation in nasopharyngeal carcinoma.

[Paper](https://openreview.net/forum?id=AVTJEREEah) · [Dataset](https://huggingface.co/datasets/pengkaiScience/NPCBench) · [Results](results/main_results.csv)

Install with Python 3.10 or later and download the questions for your task.

    pip install . huggingface_hub
    hf download pengkaiScience/NPCBench L1/questions.jsonl --repo-type dataset --local-dir data

## Choice and structured answers

    npcbench choice --task L1 --predictions predictions.jsonl

Provide one JSONL record per question, with its id and response. For example,
{"id":"question_id","response":"B"}. Return all selected letters for a
multiple-select question. Use the structure and staging fields specified in
the MRI prompts. L6 uses sample_id in place of id.

Tasks are L1, L2, L3-Path, L3-Seq, L3-Struct, T400, N400, L4, L5, and L6.
The dataset card describes image downloads. Add --output to save item scores.

## Open answers

For O1, download `O1/questions.jsonl` and use `item_id` to identify each response.
Use `npcbench_eval.prompts.render_generation_messages` to format inputs as in
the paper. Prepare one request per rubric criterion, run those requests with your judge,
and return item_id, rubric_id, and a Boolean criteria_met for each judgment.

    npcbench open --responses responses.jsonl --output judge_requests.jsonl
    npcbench open --judgments judgments.jsonl --output scores.json

For L1 short answers, download supplementary/option_free/answers.jsonl and run
npcbench open --task L1-open --responses responses.jsonl. Use the JSON answer
format in the task prompt. This command reports the rule-based slot score.

See [Results](results/README.md) to reproduce the paper's main table.
Code is [MIT](LICENSE), and original data and annotations are [CC BY 4.0](LICENSE-DATA.md).
Please cite the paper using [CITATION.cff](CITATION.cff).
