# layout-spacial-resoning

Experimental codebase for the thesis project:
**Comparison of Form Layout Generation Methods: Large Language Model vs Machine Learning Approaches**.

The repository is organized around a reproducible pipeline:

- dataset construction for the Form Layout Benchmark,
- layout generation methods,
- automatic quality metrics and failure taxonomy,
- statistical analysis and rendered qualitative examples.

## Structure

```text
configs/                 Experiment and method configuration.
data/                    Raw, processed, and split dataset files.
src/layout_spatial_reasoning/
  schemas/               Input and output JSON structures.
  dataset/               Dataset building, splitting, and quality checks.
  methods/               Baselines and five compared generation methods.
  embeddings/            Embedding provider and similarity utilities.
  evaluation/            Metrics, failure taxonomy, and statistical tests.
  prompts/               Prompt templates for LLM-based methods.
  rendering/             HTML rendering for qualitative analysis.
experiments/             Scripts for running methods and evaluations.
outputs/                 Generated layouts, metrics, reports, and renders.
tests/                   Unit and smoke tests.
```
