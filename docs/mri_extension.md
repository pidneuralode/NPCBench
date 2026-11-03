# MRI staging

L3 evaluates T stage in 400 patients and N stage in a disjoint set of 400 patients,
using AJCC/UICC edition 8. T400 has two image formats: structure cards for 100
patients and aligned sequence groups for 300 patients. N400 uses sequence groups.
The 100-patient anatomical panel also supplies the structure-invasion diagnostics.
Full100 is a separate care-path task and uses AJCC edition 9.

## Images and prompts

An axial group stacks T1, T2 and enhanced T1 from the same physical plane. A coronal
group stacks enhanced fat-suppressed T1 and precontrast non-fat-suppressed T1.
Display tiles are 512×512 pixels; groups are 512×1644 or 512×1096 pixels.
All selected groups enter one request. PDFs are used for clinical review;
inference receives images.

Download and extract the archives as shown in the [README](../README.md).
`L3/requests/T400.jsonl` and `N400.jsonl` provide complete per-patient image paths,
relative to `L3/`. The group-format files below use paths relative to `L3/extensions/`:

- `requests.jsonl`: all 700 group-format patients.
- `requests_gpt41.jsonl`: at most 50 groups per patient.
- `requests_internvl.jsonl`: context-matched InternVL groups.

Sampling is uniform within the axial and coronal groups, with a proportional
allocation and both endpoints of each group retained. T400 results report both
image-format strata as well as their combined accuracy.

## Local models

Profiles for 15 local visual models are in `configs/mri_extension/tn/`.

| Family | Context tokens | Group-image encoding |
|---|---:|---|
| Qwen, Lingshu, HealthGPT | 128,000 | All groups, model-specific processor |
| Hulu-Med-4B | 128,000 | All groups; spatial merge 2, visual budget 98,304 |
| Hulu-Med-7B/32B | 32,768 | All groups; spatial merge 4, visual budget 22,000 |
| InternVL3.5-38B | 40,960 | 33–35 groups; dynamic cropping, at most four patches per group |

Hulu processors receive HWC arrays. All profiles reserve 8,192 output tokens and
use greedy decoding. Model-specific resizing, tiling and spatial merging are
part of the evaluation settings.

```sh
CUDA_VISIBLE_DEVICES=0,1 npcbench-infer \
  --config configs/mri_extension/tn/internvl3_5_38b.json \
  --model-path /path/to/InternVL3_5-38B-HF \
  --requests data/L3/extensions/requests_internvl.jsonl \
  --asset-root data/L3/extensions --output-dir outputs/internvl_tn
```

Adjust tensor parallelism and KV allocation to your hardware. The vLLM runner
allows up to 256 scheduled sequences and bounds CPU image preparation to 48
unfinished patients in chunks of eight. The evaluated environment uses vLLM
0.19.1, Transformers 5.6.0, PyTorch 2.10.0 and qwen-vl-utils 0.0.14.

Full100 profiles are in `configs/mri_extension/full100/`. They use ten MRI pages
per patient. InternVL uses up to 12 patches per image. Hulu pathology uses spatial
merge 1 and a 16,384 visual-token budget, separately from MRI. Use `--l6-inputs`
with the extracted Full100 directory to construct care inputs from that model's
image predictions.

## API models

Install `pip install '.[api]'` and set `NPCBENCH_API_KEY` and
`NPCBENCH_BASE_URL` in your environment. The base URL includes the API prefix,
for example `https://api.openai.com/v1`. `--routes` accepts a JSON mapping from
model display names to service route names.

```sh
npcbench-api-mri --requests data/L3/extensions/requests.jsonl \
  --data-root data/L3/extensions --models GPT-5.1 \
  --concurrency 10 --output outputs/gpt51_tn
```

GPT-5.1 uses `gpt-5.1-2025-11-13`, 8,192 completion tokens and reasoning effort
`none`. Lossless WebP preserves the PNG pixel values while reducing transfer size.
The other services receive PNG. Temperature is zero and top-p is 0.0001, with
top-p omitted for Claude. Output limits are 4,096 for GPT-4.1/Claude, 8,192 for
Gemini Flash and 16,384 for Gemini Pro. Claude concurrency is eight.
For GPT-4.1, use `requests_gpt41.jsonl`.

The reported Claude condition allows up to three retries for empty or unanswered
cases. Incomplete structure outputs use an 8,192-token limit. The recovery command
preserves the first response and every attempt, stopping at the first explicit
stage, including an incorrect or unsupported stage:

```sh
npcbench-claude-recovery --requests data/L3/extensions/requests.jsonl \
  --responses outputs/api_tn/Claude-Sonnet-4.6/responses.jsonl \
  --data-root data/L3/extensions --output outputs/claude_recovery
```

## Scoring

`npcbench-mri-extension score` reports stage accuracy, class recall, balanced
accuracy, invalid counts and a confusion matrix. T400 also reports accuracy by
input format. Unsupported labels, such as N3b, count as incorrect. Supply one
response record for every case, using `null` for a request that remains unanswered
after its declared retry limit. T-stage labels alone do not supply invasion labels
for the 300 group-format T cases.
