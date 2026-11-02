# MRI extension evaluation

The extension adds T300 and N400 to L3. It retains the original T100 results;
the new five-sequence inputs are a separate visual condition. T300 and N400
use AJCC/UICC edition 8. Full100 uses edition 9 and 217 scored care decisions.

## Inputs

Each request contains all selected image groups for one patient. Axial groups
stack T1, T2 and enhanced T1; coronal groups stack enhanced fat-suppressed T1
and precontrast non-fat-suppressed T1. Columns match physical planes within
each group. A display tile is 512 × 512 pixels. The lossless PNG groups are
512 × 1644 or 512 × 1096 pixels and are sent jointly, rather than uploading PDFs.
Native model processors still encode and resize these images.

Use JSONL requests with `id`, `task` (`L3-T300` or `L3-N400`), `prompt`, and
an ordered `images` list of paths relative to the asset directory. Axial file
names start with `axial_`, followed by coronal files starting with `coronal_`.
The task prompt is retained during sampling; only image counts are updated.

The [dataset card](https://huggingface.co/datasets/pengkaiScience/NPCBench)
identifies available downloads. Extension image assets are not yet published;
the following commands apply to the prepared group inputs once available.

## Local models

`configs/mri_extension/tn/` and `configs/mri_extension/full100/` each contain
profiles for all 15 local visual models: seven Qwen models, Lingshu-7B/32B,
Hulu-Med-4B/7B/32B, HealthGPT-Pro-4B/8B and InternVL3.5-38B.

| Family | Total context | T/N image encoding |
|---|---:|---|
| Qwen, Lingshu, HealthGPT | 128,000 | All groups; explicit processor pixel limits |
| Hulu-Med-4B | 128,000 | All groups; HWC arrays, spatial merge 2, visual budget 98,304 |
| Hulu-Med-7B/32B | 32,768 | All groups; HWC arrays, spatial merge 4, visual budget 22,000 |
| InternVL3.5-38B | 40,960 | 33–35 uniformly sampled groups, dynamic cropping up to 4 patches |

All local profiles reserve 8,192 output tokens and use greedy decoding. InternVL
sampling preserves the first and last plane of both anatomical groups. It is a
sampled condition, while Hulu's spatial merging compresses visual features;
neither should be described as identical visual encoding across models.

```sh
npcbench-mri-extension prepare --requests requests.jsonl \
  --profile internvl --model-path /path/to/InternVL3_5-38B-HF \
  --output requests_internvl.jsonl
CUDA_VISIBLE_DEVICES=0,1 npcbench-infer \
  --config configs/mri_extension/tn/internvl3_5_38b.json \
  --model-path /path/to/InternVL3_5-38B-HF \
  --requests requests_internvl.jsonl --asset-root images_root \
  --output-dir outputs/internvl_tn
```

Profiles reflect two 80-GB GPUs. Adjust tensor parallelism and KV allocation to
your hardware. GPU scheduling permits up to 256 sequences; host image preparation
keeps at most 48 unfinished requests and decodes in chunks of eight. This bounds
RAM use without changing image resolution. Hulu image inference is sequential.
The continuous vLLM runner uses `LLM.enqueue` and `llm_engine.step`; install a
vLLM build supporting these APIs. The evaluated environment uses vLLM 0.19.1,
Transformers 5.6.0, PyTorch 2.10.0 and qwen-vl-utils 0.0.14. It does not preload
256 patients.

For the separate Full100 schema condition, download and extract the existing L6
archive, then supply the additional MRI prompt file:

```sh
CUDA_VISIBLE_DEVICES=0,1 npcbench-infer \
  --config configs/mri_extension/full100/hulu_med_7b.json \
  --model-path /path/to/Hulu-Med-7B \
  --l6-inputs data/L6/full100_inputs_scored217 \
  --mri-prompts data/supplementary/full100_schema/mri_samples.jsonl \
  --output-dir outputs/hulu7_full100_schema
```

Full100 retains all ten MRI pages and the original pathology inputs. InternVL
uses up to 12 patches per Full100 image. Hulu pathology uses merge 1 and a
16,384 visual-token budget, independently of its MRI settings. Care prompts are
constructed from the new MRI/pathology predictions; old care answers cannot be
substituted when their actual input changes.

## API models

Install Pillow and set `NPCBENCH_API_KEY` and `NPCBENCH_BASE_URL` in your own
environment. The base URL is the API prefix before `/chat/completions`.
`--routes routes.json` can map the displayed model names to your provider's routes.

```sh
npcbench-api-mri --requests requests.jsonl --data-root images_root \
  --models Claude-Sonnet-4.6 Gemini-2.5-Pro --concurrency 16 \
  --output outputs/api_tn
```

Claude stays at eight concurrent requests; other models adapt up to the supplied
limit. Outputs use 4,096 tokens for Claude/GPT-4.1, 8,192 for GPT-5.1/Flash, and
16,384 for Gemini Pro. Temperature is zero; non-Claude requests also use
top-p 0.0001. Images precede the task text and use high-detail lossless PNG input.
For the GPT-4.1 route limited to 50 images, first use `prepare --profile gpt41`.
This proportionally samples the two groups and retains both groups' endpoints.

Transport failures are retried with the same request. Invalid or truncated
answers remain recorded; they are not repeatedly sampled until correct.

## Stage scoring

Labels contain `id` and `t_stage` or `n_stage`. Predictions contain the same `id`
and the raw `response`. Score T and N separately:

```sh
npcbench-mri-extension score --stage N --labels n_labels.jsonl \
  --predictions n_predictions.jsonl --output n_scores.json
```

Stage accuracy includes invalid answers in the denominator as incorrect. Missing
network responses must be completed before scoring. The output includes invalid
counts and a confusion matrix. T-stage scoring does not imply that new T300
structure labels have been clinically annotated.
