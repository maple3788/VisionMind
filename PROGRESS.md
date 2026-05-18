# PROGRESS.md — Multimodal QA System

> **How to use:** Check off items as you complete them. Add notes under each item. Never skip a phase.
> Update `Current Phase` and `Last Updated` each session.

**Current Phase:** 4 — End-to-End Inference Pipeline
**Last Updated:** 2026-05-18
**Overall:** `████░░░░░░` 40%

---

## Phase 0 — Environment & Foundations
**Goal:** Working Python env, all dependencies installed, GPU verified, first model loaded.

### Setup
- [x] Create conda/venv environment (`python 3.10+`)
- [x] Install base dependencies (`torch`, `transformers`, `PIL`, `numpy`)
- [x] Install project dependencies (`peft`, `accelerate`, `bitsandbytes`, `loguru`)
- [x] Verify CUDA / MPS / CPU device availability
- [x] Configure `.env` with `HF_TOKEN`, `HF_HOME` cache path
- [x] Clone project repo / initialize git

### Foundations Notebook: `NB-00-environment-setup.ipynb`
- [x] Device detection cell (CUDA / MPS / CPU)
- [x] Hugging Face login and token test
- [x] Download a tiny model (`google/vit-base-patch16-224`) to verify pipeline
- [x] Visualize a sample image with PIL
- [x] Shape inspection: understand `[batch, channels, height, width]` tensor layout

**Phase 0 Complete:** [x]
**Notes:** venv at `.venv/` (Python 3.13). MPS detected on Apple Silicon. ViT output verified `[1, 197, 768]`. Config YAMLs created. Notebooks moved to `notebooks/`.

---

## Phase 1 — Vision Encoders: ViT & CLIP
**Goal:** Understand how raw images become feature vectors. Master ViT and CLIP internals.

### Concepts to Learn
- [x] What is a patch? How does ViT divide an image into patches?
- [x] What is positional encoding in the vision context?
- [x] What is the `[CLS]` token and why it represents the whole image?
- [x] How does CLIP align image and text in a shared embedding space?
- [x] What is contrastive learning? (InfoNCE loss)

### Code: `src/encoders/vit_encoder.py`
- [x] `ViTEncoder` class wrapping `google/vit-base-patch16-224` (configurable)
- [x] `encode(image: PIL.Image) -> torch.Tensor` method
- [x] Output shape documented: `[batch, num_patches+1, hidden_dim]`
- [x] `get_patch_features()` vs `get_cls_feature()` methods
- [x] Unit test in `tests/test_encoders.py`

### Code: `src/encoders/clip_encoder.py`
- [x] `CLIPVisionEncoder` class wrapping `openai/clip-vit-large-patch14`
- [x] `encode_image(image) -> torch.Tensor`
- [x] `encode_text(text: str) -> torch.Tensor`
- [x] `compute_similarity(image, text) -> Tensor` (cosine sim matrix)
- [x] Unit test verifying image-text alignment score

### Notebook: `NB-01-vit-exploration.ipynb`
- [x] Load ViT, run on sample image, inspect output shapes
- [x] Visualize attention maps (which patches does the model focus on?)
- [x] Compare `[CLS]` token vs mean-pooled patch features
- [x] Plot patch grid on original image

### Notebook: `NB-02-clip-encoder.ipynb`
- [x] Zero-shot image classification with CLIP
- [x] Image-text similarity scoring
- [x] Embedding space visualization with UMAP/t-SNE
- [x] Experiment: what happens when you encode an image vs a description of it?

**Phase 1 Complete:** [x]
**Notes:** ViTEncoder uses ViTImageProcessor. CLIP similarity uses projected L2-normalized features. 7 unit tests pass with mocked HF models.

---

## Phase 2 — Input Projectors
**Goal:** Build the bridge between vision encoder output and LLM token space.

### Concepts to Learn
- [x] Why can't we feed CLIP embeddings directly to a language model?
- [x] What does "modality alignment" mean?
- [x] Linear projector: simplest possible bridge
- [x] MLP projector: adds non-linearity, better capacity
- [x] Q-Former: cross-attention based, dynamic queries (from BLIP-2)
- [x] What is the token budget? Why do we need to compress patch tokens?

### Code: `src/projectors/linear_projector.py`
- [x] `LinearProjector(in_dim, out_dim)` class
- [x] `forward(x: Tensor) -> Tensor`
- [x] Shape: `[batch, seq, clip_dim]` → `[batch, seq, llm_dim]`
- [x] Unit test with dummy tensors

### Code: `src/projectors/mlp_projector.py`
- [x] `MLPProjector(in_dim, hidden_dim, out_dim, num_layers=2)`
- [x] GELU activation, LayerNorm
- [x] Dropout option for regularization
- [x] Unit test

### Code: `src/projectors/qformer.py`
- [x] `QFormer(num_queries, encoder_dim, llm_dim, num_heads, num_layers)`
- [x] Learnable query tokens `nn.Parameter`
- [x] Cross-attention to encoder output
- [x] Self-attention among queries
- [x] Output: fixed number of tokens regardless of image resolution
- [x] Unit test verifying output is always `[batch, num_queries, llm_dim]`

### Notebook: `NB-03-linear-projector.ipynb`
- [x] Build and visualize a linear projector
- [x] Check that gradients flow through it
- [x] Compare input/output embedding distributions

### Notebook: `NB-04-mlp-projector.ipynb`
- [x] Build MLP projector, compare to linear
- [x] Plot loss curve for a toy alignment task

### Notebook: `NB-05-qformer-deep-dive.ipynb`
- [x] Step-by-step Q-Former implementation from scratch
- [x] Visualize cross-attention weights (which patches do queries attend to?)
- [x] Compare: 32 queries vs 64 queries — tradeoffs
- [x] Production `src/projectors/qformer.py` integration cell

**Phase 2 Complete:** [x]
**Notes:** 9 projector tests pass. QFormer uses stacked cross/self-attn + FFN blocks. Config adds `qformer` section.

---

## Phase 3 — LLM Backbone Integration
**Goal:** Connect vision encoder + projector to a real LLM. Run your first multimodal inference.

### Concepts to Learn
- [x] How does Qwen-VL merge image tokens into the text sequence?
- [x] What is the `<img>` special token pattern?
- [x] Token budget: how many image tokens fit in 2048 context window?
- [x] Autoregressive generation: how does the LLM generate the answer token by token?
- [x] 4-bit quantization (BitsAndBytes) — why and when

### Code: `src/llm/backbone.py`
- [x] `MultimodalLLM` class (wraps encoder + projector + LLM)
- [x] `__init__(encoder, projector, llm_model_id, device, load_in_4bit=True)`
- [x] `prepare_inputs(image, text_prompt) -> dict` — merges modalities
- [x] `generate(image, prompt, max_new_tokens=512) -> str`
- [x] Device management: keep encoder on CPU if GPU is tight
- [x] Unit test with a mock LLM (avoid downloading 7B for tests)

### Notebook: `NB-06-llm-backbone.ipynb`
- [x] Load Qwen2-VL-2B (smaller, faster) or LLaVA-1.5-7B
- [x] Run your first multimodal QA: image + question → answer
- [x] Inspect the merged input token sequence
- [x] Benchmark: latency, memory usage, tokens/sec
- [x] Try different prompts and observe answer quality

**Phase 3 Complete:** [x]
**Notes:** Native path uses Qwen2-VL processor. Custom path prepends projected CLIP patches via inputs_embeds. Projector out_dim=1536 for Qwen2-VL-2B. 4-bit only on CUDA.

---

## Phase 4 — End-to-End Inference Pipeline
**Goal:** Clean, unified pipeline class. Ready for real use cases.

### Code: `src/pipeline/multimodal_qa.py`
- [ ] `MultimodalQAPipeline` class
- [ ] `answer(image_path_or_url, question, history=None) -> str`
- [ ] Multi-turn conversation support (history)
- [ ] Image input types: PIL, file path, URL, base64
- [ ] Graceful fallback: text-only question (no image)
- [ ] Streaming output support (generator)

### Notebook: `NB-07-end-to-end-inference.ipynb`
- [ ] Full pipeline demo: load → ask → answer
- [ ] Multi-turn QA conversation demo
- [ ] Edge cases: blurry image, wrong language, very long question
- [ ] Side-by-side: Linear projector vs MLP vs Q-Former quality comparison
- [ ] Error handling walkthrough

**Phase 4 Complete:** [ ]
**Notes:**

---

## Phase 5 — Fine-Tuning with LoRA (Domain Adaptation)
**Goal:** Adapt the model to a specific domain (e.g. medical imaging, document QA).

### Concepts to Learn
- [ ] What is LoRA? How does rank decomposition work?
- [ ] Why fine-tune only the projector + LoRA adapters, not the full model?
- [ ] What is instruction tuning? What format does the data need to be in?
- [ ] Evaluation: BLEU, CIDEr, VQA accuracy

### Dataset Preparation: `src/data/dataset.py`
- [ ] `MultimodalQADataset(data_dir, split)` class
- [ ] Data format: `{"image": path, "question": str, "answer": str}`
- [ ] Collate function for batching mixed-length sequences
- [ ] Data augmentation: random crop, color jitter, horizontal flip
- [ ] Train/val/test split logic

### Code: `src/llm/lora_finetune.py`
- [ ] `apply_lora(model, r=8, alpha=16, target_modules=[...]) -> model`
- [ ] Training loop with `accelerate`
- [ ] Gradient checkpointing enabled
- [ ] Checkpoint saving every N steps
- [ ] Evaluation loop with VQA accuracy metric
- [ ] `merge_and_save(model, output_dir)` — merge LoRA weights

### Notebook: `NB-08-lora-finetuning.ipynb`
- [ ] Dataset loading and visualization (sample 10 examples)
- [ ] LoRA config walkthrough: what does each hyperparameter do?
- [ ] Training run on small dataset (50-100 examples to see it works)
- [ ] Loss curve plotting
- [ ] Before/after comparison: base model vs fine-tuned on domain examples
- [ ] Export fine-tuned model

**Phase 5 Complete:** [ ]
**Notes:**

---

## Phase 6 — RAG + Multimodal Retrieval (Extension)
**Goal:** Ground answers in a knowledge base. Combine retrieval with generation.

### Concepts to Learn
- [ ] How does RAG work for text? (query → retrieve → augment → generate)
- [ ] How does multimodal RAG differ? (image query, image+text retrieval)
- [ ] Vector databases: FAISS vs Chroma vs Qdrant
- [ ] CLIP as a retrieval encoder: embed images and text in the same space

### Code: `src/pipeline/rag_retriever.py`
- [ ] `MultimodalRetriever(index_path, encoder)` class
- [ ] `index_documents(docs: List[dict])` — embed and store
- [ ] `retrieve(query_image=None, query_text=None, top_k=3) -> List[dict]`
- [ ] Supports image query, text query, or both
- [ ] FAISS index backend

### Notebook: `NB-09-rag-multimodal.ipynb`
- [ ] Build a small image+text knowledge base (20-50 documents)
- [ ] Embed and index with CLIP
- [ ] Query with a new image → retrieve relevant docs
- [ ] Inject retrieved context into LLM prompt
- [ ] Compare: RAG vs no-RAG answer quality
- [ ] Visualize retrieved documents alongside query

**Phase 6 Complete:** [ ]
**Notes:**

---

## Phase 7 — API & OpenWebUI Deployment
**Goal:** Serve the model via API. Connect to OpenWebUI for a chat interface.

### Code: `src/serving/api_server.py`
- [ ] FastAPI app with `/v1/chat/completions` (OpenAI-compatible)
- [ ] Multipart image upload support
- [ ] Streaming response support (SSE)
- [ ] Health check endpoint `/health`
- [ ] Basic API key auth

### Docker: `docker/docker-compose.yml`
- [ ] Service: `api` — runs FastAPI server
- [ ] Service: `openwebui` — OpenWebUI container
- [ ] Volume mounts for model cache
- [ ] GPU passthrough (`runtime: nvidia`)

### Notebook: `NB-10-deployment.ipynb`
- [ ] OpenWebUI setup walkthrough
- [ ] Connect OpenWebUI to your custom API
- [ ] Test multimodal chat in the UI
- [ ] vLLM deployment (if GPU available): performance comparison
- [ ] Ollama as alternative local serving option

**Phase 7 Complete:** [ ]
**Notes:**

---

## Phase 8 — Multimodal Generation Output (Stretch)
**Goal:** Enable the system to answer with generated images, not just text.

- [ ] Understand Output Projector → Modality Generator pipeline
- [ ] Integrate Stable Diffusion for image generation answers
- [ ] `src/generators/image_generator.py`
- [ ] End-to-end: question → text answer + generated image
- [ ] Notebook: `NB-11-image-generation-output.ipynb`

**Phase 8 Complete:** [ ]
**Notes:**

---

## Session Log

| Date | Phase | What was done | Blockers |
|------|-------|---------------|----------|
| 2026-05-18 | 3 | MultimodalLLM, native+custom paths, NB-06 | — |
| 2026-05-18 | 2 | Linear/MLP/QFormer projectors, tests, NB-03/04/05 | — |
| 2026-05-18 | 1 | ViTEncoder, CLIPVisionEncoder, tests, NB-01/02 | — |
| 2026-05-18 | 0 | Scaffold, configs, venv, deps, ViT smoke test, git init | — |

---

## Key Decisions Made

| Decision | Chosen | Reason |
|----------|--------|--------|
| Vision Encoder | CLIP ViT-L/14 | Best balance of quality and accessibility |
| LLM | Qwen2-VL-2B (start) | Fits on 8GB GPU, good multilingual |
| Projector | MLP (primary) | Better than linear, simpler than Q-Former |
| Fine-tuning | LoRA r=8 | Memory-efficient, fast to iterate |
| Vector DB | FAISS | No server needed, easy to start |
| Serving | FastAPI + OpenWebUI | OpenAI-compatible, plug-and-play UI |
