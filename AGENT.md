# AGENT.md — Multimodal QA System (Project 6)

## Project Identity

You are building an **Intelligent Multimodal Question Answering System** from scratch. The goal is both educational (understanding every component deeply) and practical (producing a working, deployable system). The learner is a Python developer studying RAG, LLMs, and supervised learning.

## Architecture at a Glance

```
[Image / Video / Audio Input]
        ↓
[Modality Encoder]       ← ViT, CLIP, HuBERT
        ↓
[Input Projector]        ← Linear, MLP, Q-Former
        ↓
[LLM Backbone]           ← Qwen-VL, LLaVA, DeepSeek-VL
        ↓
[Output Projector]       ← MLP / Tiny Transformer
        ↓
[Modality Generator]     ← Stable Diffusion, AudioLDM (optional)
```

The system has 3 macro-stages:
1. **Multimodal Understanding** — encode non-text inputs, project into LLM token space
2. **LLM Backbone Reasoning** — joint text + vision reasoning
3. **Multimodal Generation** — optionally generate images/audio as answers

## Tech Stack

| Layer | Choice | Rationale |
|---|---|---|
| Vision Encoder | CLIP ViT-L/14 → EVA-CLIP | Industry standard, well-documented |
| Projector | Linear → MLP → Q-Former | Build complexity incrementally |
| LLM | Qwen2-VL-7B / LLaVA-1.5 | Open-source, good docs, Python-friendly |
| Fine-tuning | LoRA via PEFT | Memory-efficient, learnable |
| Serving | vLLM / Ollama (local) | Start local, scale to vLLM |
| UI | OpenWebUI | Docker-based, plug-and-play |
| Experiment tracking | MLflow / Weights & Biases | For fine-tuning runs |
| Notebooks | Jupyter | One notebook per concept |

## Coding Conventions

- **Language:** Python 3.10+
- **Style:** PEP8, type hints on all function signatures
- **Docstrings:** Google-style on every class and public method
- **Config:** YAML files via `OmegaConf` or `dataclasses`, never hardcode paths
- **Secrets:** `.env` file + `python-dotenv`, never committed
- **Logging:** `loguru` or standard `logging`, not bare `print()`
- **Tests:** `pytest`, one test file per module
- **Notebook naming:** `NB-{phase}-{topic}.ipynb` e.g. `NB-01-clip-exploration.ipynb`

## Phase Map

```
Phase 0: Environment & Foundations
Phase 1: Vision Encoders (ViT, CLIP)
Phase 2: Input Projectors (Linear, MLP, Q-Former)
Phase 3: LLM Backbone Integration (Qwen-VL / LLaVA)
Phase 4: End-to-End Inference Pipeline
Phase 5: Fine-tuning with LoRA (domain adaptation)
Phase 6: RAG + Multimodal Retrieval
Phase 7: API & OpenWebUI Deployment
Phase 8: (Stretch) Multimodal Generation output
```

## File Layout

```
multimodal-qa/
├── AGENT.md               ← this file
├── PROGRESS.md            ← phase-by-phase checklist
├── cursor_prompt.md       ← Cursor AI prompt
├── .env.example
├── requirements.txt
├── config/
│   ├── model_config.yaml
│   └── train_config.yaml
├── src/
│   ├── encoders/
│   │   ├── __init__.py
│   │   ├── clip_encoder.py
│   │   └── vit_encoder.py
│   ├── projectors/
│   │   ├── __init__.py
│   │   ├── linear_projector.py
│   │   ├── mlp_projector.py
│   │   └── qformer.py
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── backbone.py
│   │   └── lora_finetune.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── multimodal_qa.py
│   │   └── rag_retriever.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── preprocessing.py
│   └── serving/
│       ├── __init__.py
│       └── api_server.py
├── notebooks/
│   ├── NB-00-environment-setup.ipynb
│   ├── NB-01-vit-exploration.ipynb
│   ├── NB-02-clip-encoder.ipynb
│   ├── NB-03-linear-projector.ipynb
│   ├── NB-04-mlp-projector.ipynb
│   ├── NB-05-qformer-deep-dive.ipynb
│   ├── NB-06-llm-backbone.ipynb
│   ├── NB-07-end-to-end-inference.ipynb
│   ├── NB-08-lora-finetuning.ipynb
│   ├── NB-09-rag-multimodal.ipynb
│   └── NB-10-deployment.ipynb
├── tests/
│   ├── test_encoders.py
│   ├── test_projectors.py
│   └── test_pipeline.py
└── docker/
    └── docker-compose.yml
```

## Agent Decision Rules

1. **Always read PROGRESS.md** before starting any task to know current phase.
2. **Never skip phases** — each phase builds on the last.
3. **Notebook first, then src/** — prototype in notebook, then extract clean code to `src/`.
4. **Small, testable functions** — each function does one thing; if it does two, split it.
5. **When implementing a new component**, always:
   - Add unit test in `tests/`
   - Add config entry in `config/`
   - Update PROGRESS.md checkbox
6. **For model downloads**, use `huggingface_hub` with caching; never download to project root.
7. **GPU memory** — always include `.to(device)` and `torch.no_grad()` in inference paths.
8. **When in doubt on architecture**, prefer the simpler version first (Linear projector before Q-Former).

## Key Learning Checkpoints

After each phase, you should be able to answer:
- Phase 1: What does CLIP's vision encoder output? What is its shape?
- Phase 2: Why do we need a projector between the encoder and LLM?
- Phase 3: How does Qwen-VL merge vision tokens with text tokens?
- Phase 5: What is LoRA and why is it better than full fine-tuning here?
- Phase 6: How does multimodal RAG differ from text-only RAG?

## Common Pitfalls to Avoid

- Don't load the full LLM into GPU until Phase 3 — use CPU or small proxies earlier
- CLIP outputs `[batch, seq_len, hidden]` — check shapes before projecting
- Q-Former has its own learnable queries — they are NOT the image patches
- LoRA rank is a hyperparameter — start with `r=8`, tune later
- OpenWebUI expects OpenAI-compatible API format — wrap your model accordingly
