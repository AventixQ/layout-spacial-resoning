# Cluster experiment jobs

These scripts run prompt-based local Hugging Face models through the same
`experiments/run_full_experiment.py` pipeline as the API providers.

They assume the project lives at:

```bash
/work/s504297/layout-spacial-resoning
```

and write SLURM logs to:

```bash
/home/s504297/jobs/logs
```

## Local model pilot matrix

Run a 20-form validation pilot for Qwen 4B, Qwen 8B, Bielik 11B, and PLLuM 8B:

```bash
cd /work/s504297/layout-spacial-resoning
sbatch cluster/layout_local_hf_matrix_val.sbatch
```

Watch jobs:

```bash
squeue -u s504297
tail -f /home/s504297/jobs/logs/layout_local_hf_matrix_*.out
```

The matrix job defaults to `EXPERIMENT_LIMIT=20`. To run the full validation
split after the pilot looks good:

```bash
sbatch --export=ALL,EXPERIMENT_LIMIT=0 cluster/layout_local_hf_matrix_val.sbatch
```

## One local model

Run one selected model:

```bash
sbatch --export=ALL,LOCAL_HF_LAYOUT_MODEL=Qwen/Qwen3-4B-Instruct-2507,EXPERIMENT_LIMIT=20 cluster/layout_local_hf_val.sbatch
```

Useful overrides:

```bash
FORMS_PATH=data/splits/val.jsonl
EXPERIMENT_LIMIT=20
LOCAL_HF_LAYOUT_MODEL=Qwen/Qwen3-4B-Instruct-2507
LOCAL_HF_MAX_NEW_TOKENS=4096
LOCAL_HF_LOAD_IN_4BIT=true
```

For local Hugging Face models, `single_structured_output` is skipped by default
because plain Transformers generation does not enforce a JSON schema. Set
`LOCAL_HF_INCLUDE_STRUCTURED_OUTPUT=true` only if guided/constrained decoding is
added later.

## vLLM OpenAI-compatible API

Run a model as a local OpenAI-compatible API inside one SLURM job, then execute
the validation experiment against `localhost`:

```bash
cd /work/s504297/layout-spacial-resoning
sbatch --export=ALL,VLLM_MODEL=Qwen/Qwen3-4B-Instruct-2507,EXPERIMENT_LIMIT=20 cluster/layout_vllm_api_val.sbatch
```

The job uses `/home/s504297/vllm901.sif` by default. Override with:

```bash
VLLM_IMAGE=/path/to/vllm.sif
```

By default this mode runs `zero_shot`, `few_shot`, `cot`, `structured_output`,
`multi_agent`, and `hybrid`.

`structured_output` uses vLLM guided decoding through `guided_json` by default:

```bash
OPENAI_RESPONSE_FORMAT_MODE=vllm_guided_json
```

For newer vLLM versions that removed `guided_json`, use:

```bash
OPENAI_RESPONSE_FORMAT_MODE=vllm_structured_outputs
```

For plain prompt-only generation without constrained decoding:

```bash
OPENAI_RESPONSE_FORMAT_MODE=disabled
EXPERIMENT_SINGLE_VARIANTS=zero_shot,few_shot,cot
```

Useful examples:

```bash
sbatch --export=ALL,VLLM_MODEL=Qwen/Qwen3-8B,EXPERIMENT_LIMIT=20 cluster/layout_vllm_api_val.sbatch
sbatch --export=ALL,VLLM_MODEL=speakleash/Bielik-11B-v3.0-Instruct,EXPERIMENT_LIMIT=20 cluster/layout_vllm_api_val.sbatch
sbatch --export=ALL,VLLM_MODEL=CYFRAGOVPL/Llama-PLLuM-8B-instruct-2512,EXPERIMENT_LIMIT=20 cluster/layout_vllm_api_val.sbatch
```
