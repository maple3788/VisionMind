# Cursor Prompt — Multimodal QA System (Project 6)

> **Usage:** Paste the relevant section into Cursor's AI panel (Cmd+K or Cmd+L) when starting each phase. Each prompt is self-contained. Always tell Cursor to read `AGENT.md` and `PROGRESS.md` first.

---

## Master Prompt (Use at Start of Every Session)

```
You are helping me build a Multimodal Question Answering system from scratch for learning purposes.

Before writing any code:
1. Read AGENT.md — this contains the architecture, conventions, and decision rules
2. Read PROGRESS.md — find the current phase and only work on unchecked items

Rules:
- Python 3.10+, type hints everywhere, Google-style docstrings
- Prototype in notebooks first, then extract to src/
- Every new class gets a unit test in tests/
- Never hardcode paths — use config files
- Use loguru for logging, not print()
- When adding a model, always include .to(device) and support 4-bit quantization
- After completing each task, update PROGRESS.md checkbox

Current task: [DESCRIBE WHAT YOU WANT TO BUILD]
```

---

## Phase 0 — Environment Setup

```
Read AGENT.md and PROGRESS.md first.

Create the full project scaffold:

1. Create the directory structure exactly as defined in AGENT.md file layout section
2. Create requirements.txt with these groups:
   # Core ML
   torch>=2.1.0
   torchvision
   transformers>=4.37.0
   accelerate>=0.26.0
   peft>=0.8.0
   bitsandbytes>=0.41.0
   
   # Vision
   Pillow>=10.0.0
   open-clip-torch>=2.24.0
   timm>=0.9.0
   
   # Data
   datasets>=2.16.0
   numpy>=1.24.0
   pandas>=2.0.0
   
   # RAG
   faiss-cpu>=1.7.4
   sentence-transformers>=2.3.0
   
   # Serving
   fastapi>=0.109.0
   uvicorn[standard]>=0.27.0
   python-multipart>=0.0.9
   
   # Utils
   loguru>=0.7.0
   python-dotenv>=1.0.0
   omegaconf>=2.3.0
   huggingface_hub>=0.20.0
   mlflow>=2.10.0
   
   # Notebooks
   jupyter>=1.0.0
   ipywidgets>=8.0.0
   matplotlib>=3.8.0
   seaborn>=0.13.0
   umap-learn>=0.5.0

3. Create .env.example:
   HF_TOKEN=your_token_here
   HF_HOME=~/.cache/huggingface
   MODEL_CACHE_DIR=./model_cache
   DEVICE=auto
   API_KEY=your_api_key_here

4. Create config/model_config.yaml:
   vision_encoder:
     model_id: openai/clip-vit-large-patch14
     output_dim: 1024
     freeze: true
   
   projector:
     type: mlp  # linear | mlp | qformer
     in_dim: 1024
     hidden_dim: 2048
     out_dim: 4096
     num_layers: 2
   
   llm:
     model_id: Qwen/Qwen2-VL-2B-Instruct
     load_in_4bit: true
     max_new_tokens: 512
     temperature: 0.7

5. Create config/train_config.yaml:
   training:
     batch_size: 4
     gradient_accumulation_steps: 4
     learning_rate: 2e-4
     num_epochs: 3
     warmup_ratio: 0.03
     save_steps: 100
     eval_steps: 100
   
   lora:
     r: 8
     alpha: 16
     dropout: 0.05
     target_modules:
       - q_proj
       - v_proj
       - o_proj
       - gate_proj

6. Create NB-00-environment-setup.ipynb with these sections:
   - Device detection (CUDA / MPS / CPU auto-detect)
   - HuggingFace login and token validation
   - Download test: google/vit-base-patch16-224 with feature extractor
   - Load one image from PIL, display it, show tensor shape
   - Memory report (GPU if available)
   - Print all key library versions

Mark Phase 0 items in PROGRESS.md as done after creating these.
```

---

## Phase 1 — Vision Encoders

```
Read AGENT.md and PROGRESS.md. We are on Phase 1.

Task: Build the vision encoder module.

1. Create src/encoders/vit_encoder.py:
   - Class ViTEncoder with __init__(model_id, device, freeze=True)
   - Method encode(images: list[PIL.Image]) -> torch.Tensor
     Returns shape [batch, num_patches+1, hidden_dim]
   - Method get_cls_feature(images) -> torch.Tensor  [batch, hidden_dim]
   - Method get_patch_features(images) -> torch.Tensor  [batch, num_patches, hidden_dim]
   - Proper device handling, no_grad in inference
   - Google docstrings on class and all methods

2. Create src/encoders/clip_encoder.py:
   - Class CLIPVisionEncoder with __init__(model_id, device)
   - Method encode_image(images) -> torch.Tensor
   - Method encode_text(texts: list[str]) -> torch.Tensor
   - Method compute_similarity(images, texts) -> torch.Tensor  (cosine sim matrix)
   - Both image and text encoders frozen by default

3. Create tests/test_encoders.py:
   - Test ViTEncoder output shape is correct
   - Test CLIPVisionEncoder image-text similarity: a dog image vs "a dog" should score > 0.2
   - Use a synthetic 224x224 random PIL image for tests (no downloads in tests)
   - Mock the model download with a tiny model or patch

4. Create NB-01-vit-exploration.ipynb with these cells:
   - Load ViTEncoder with vit-base (smaller for notebook)
   - Run on a real image (download a sample dog or cat image)
   - Print output shape: should be [1, 197, 768] for base-16
   - Visualize: draw the 16x16 patch grid on the original image
   - Plot attention rollout (which patches get most attention from CLS)
   - Compare CLS token vs mean of all patch tokens as image representation
   - Markdown cells explaining: what is a patch, what is CLS, what is positional encoding

5. Create NB-02-clip-encoder.ipynb with these cells:
   - Load CLIPVisionEncoder
   - Zero-shot classification: 5 images, 5 text labels, build similarity matrix
   - Visualize similarity matrix as heatmap
   - UMAP projection: embed 20 images, color by category, show clustering
   - Text-guided retrieval: given a text query, find closest image from a set
   - Key insight cell: explain contrastive learning and InfoNCE loss in plain English with diagram
```

---

## Phase 2 — Input Projectors

```
Read AGENT.md and PROGRESS.md. We are on Phase 2.

Task: Build all three projector types.

1. Create src/projectors/linear_projector.py:
   - Class LinearProjector(nn.Module) with in_dim, out_dim
   - Single nn.Linear layer + optional LayerNorm
   - forward(x: Tensor) -> Tensor, preserves batch and seq dims

2. Create src/projectors/mlp_projector.py:
   - Class MLPProjector(nn.Module) with in_dim, hidden_dim, out_dim, num_layers, dropout
   - Stack of Linear → GELU → LayerNorm layers
   - forward(x: Tensor) -> Tensor
   - num_layers controls depth (default 2)

3. Create src/projectors/qformer.py:
   - Class QFormer(nn.Module)
   - __init__(num_queries, encoder_dim, llm_dim, num_heads=8, num_layers=6, dropout=0.1)
   - self.queries = nn.Parameter(torch.randn(1, num_queries, llm_dim))
   - Cross-attention: queries attend to encoder output
   - Self-attention: queries interact with each other
   - forward(encoder_output: Tensor) -> Tensor  shape [batch, num_queries, llm_dim]
   - Add detailed comments explaining each step

4. Create tests/test_projectors.py:
   - Test each projector with dummy tensors
   - Verify shapes: [2, 196, 1024] → [2, 196, 4096] for Linear/MLP
   - Verify QFormer always outputs [batch, num_queries, llm_dim] regardless of input seq len
   - Test that gradients flow (loss.backward() doesn't error)

5. Create NB-03-linear-projector.ipynb:
   - Build a LinearProjector from scratch using only nn.Linear
   - Toy alignment task: project random CLIP features to match random LLM features
   - Plot loss going down
   - Visualize: input vs output embedding distribution (histogram)
   - Explain: this is essentially a learned "translation dictionary"

6. Create NB-04-mlp-projector.ipynb:
   - Same toy alignment task but with MLP
   - Compare convergence: Linear vs MLP on same task
   - Show that MLP learns non-linear transformations Linear cannot
   - Ablation: 1 layer vs 2 vs 4 — plot all loss curves together

7. Create NB-05-qformer-deep-dive.ipynb:
   - Implement Q-Former step by step from scratch (not importing our src/)
   - Step 1: just cross-attention (queries to encoder)
   - Step 2: add self-attention between queries
   - Step 3: add the feed-forward and LayerNorm
   - Visualize: cross-attention weights as image heatmap
   - Key insight: queries are NOT image patches — they are learnable "questions" the model asks about the image
   - Compare 16 queries vs 32 vs 64: tradeoffs in expressiveness vs token budget
```

---

## Phase 3 — LLM Backbone Integration

```
Read AGENT.md and PROGRESS.md. We are on Phase 3.

Task: Connect encoder + projector to an LLM and run first multimodal inference.

1. Create src/llm/backbone.py:
   - Class MultimodalLLM
   - __init__(encoder, projector, llm_model_id, device="auto", load_in_4bit=True)
   - Load LLM with BitsAndBytesConfig if load_in_4bit
   - Method prepare_inputs(image: PIL.Image, prompt: str) -> dict
     * encode image → project → get visual tokens
     * tokenize prompt
     * merge: [BOS] [visual tokens] [text tokens]
     * return attention_mask, input_ids equivalent
   - Method generate(image, prompt, max_new_tokens=512, stream=False) -> str
   - Method chat(history: list[dict], image=None) -> str  (multi-turn)
   - Logging of token counts and latency

2. Create NB-06-llm-backbone.ipynb:
   - Load Qwen2-VL-2B-Instruct (use the model's own processor first for baseline)
   - Baseline inference: use HuggingFace's built-in pipeline to understand expected behavior
   - Then plug in our custom encoder + MLP projector
   - Side-by-side: HF native vs our pipeline on same image+question
   - Inspect: print the merged token sequence to understand how vision tokens appear
   - Benchmark cell: measure tokens/sec, peak GPU memory, time-to-first-token
   - Multi-turn conversation demo: ask follow-up questions about the same image
   - Edge cases: no image (text only), very large image (test resize handling)
```

---

## Phase 4 — End-to-End Pipeline

```
Read AGENT.md and PROGRESS.md. We are on Phase 4.

Task: Clean unified pipeline ready for real applications.

1. Create src/pipeline/multimodal_qa.py:
   - Class MultimodalQAPipeline
   - __init__(config_path: str) — loads from YAML, initializes all components
   - Method answer(image=None, question: str, history=None, stream=False) -> str
   - Image input handling: accepts PIL.Image, file path (str), URL (str), base64 (str)
   - _load_image(input) -> PIL.Image  — private method handling all input types
   - _build_prompt(question, history) -> str
   - Conversation history as list of {"role": "user"/"assistant", "content": str}
   - Streaming generator variant
   - Graceful error handling with informative messages

2. Create NB-07-end-to-end-inference.ipynb:
   - Load the full pipeline from config
   - Demo 1: Single image QA (load an image from URL, ask a question)
   - Demo 2: Multi-turn conversation (3 rounds of questions about the same image)
   - Demo 3: Text-only question (no image, should still work)
   - Demo 4: Edge cases (corrupted image path, very long question)
   - Projector comparison: run same image+question through Linear vs MLP vs QFormer projectors
   - Create a simple Gradio interface (optional cell at the end)
```

---

## Phase 5 — LoRA Fine-tuning

```
Read AGENT.md and PROGRESS.md. We are on Phase 5.

Task: Fine-tune the model on a domain-specific dataset using LoRA.

1. Create src/data/dataset.py:
   - Class MultimodalQADataset(Dataset)
   - __init__(data_dir, split, tokenizer, image_processor, max_length=512)
   - Data format: JSONL with {"image": "path.jpg", "question": "...", "answer": "..."}
   - __getitem__ returns tokenized inputs ready for training
   - Collate function handle_batch(examples) -> dict
   - Class DatasetSplitter: splits a directory into train/val/test (80/10/10)

2. Create src/data/preprocessing.py:
   - Function build_instruction_prompt(question, answer, system_prompt=None) -> str
     Uses the Qwen chat template format
   - Function augment_image(image: PIL.Image, augment_config: dict) -> PIL.Image
     Random crop, flip, color jitter, controlled by config
   - Function validate_dataset(data_dir) -> dict  — reports missing images, bad JSON

3. Create src/llm/lora_finetune.py:
   - Function apply_lora(model, lora_config: dict) -> model  (using PEFT)
   - Class Trainer:
     * __init__(model, train_dataset, val_dataset, train_config)
     * train() — training loop with gradient accumulation
     * evaluate() — VQA accuracy metric
     * save_checkpoint(path)
     * log_metrics(step, metrics) — MLflow logging
   - Function merge_and_save(model, output_dir) — merge LoRA and save full model

4. Create NB-08-lora-finetuning.ipynb:
   - Section 1: Dataset prep
     * Download a small multimodal dataset (e.g. VQAv2 sample or TextVQA sample)
     * Visualize 10 training examples with image + Q + A
     * Show the instruction prompt format
   - Section 2: LoRA configuration
     * Explain each hyperparameter: r, alpha, target_modules, dropout
     * Show which layers get LoRA adapters (print named modules)
     * Count: how many trainable params vs frozen params?
   - Section 3: Training
     * Train for 1 epoch on 100 examples (fast proof of concept)
     * Plot training loss in real time (using a callback)
   - Section 4: Evaluation
     * Compare base model vs fine-tuned on 20 eval examples
     * Show qualitative improvements on domain examples
   - Section 5: Export
     * Save fine-tuned model
     * Reload and verify it works
```

---

## Phase 6 — Multimodal RAG

```
Read AGENT.md and PROGRESS.md. We are on Phase 6.

Task: Add retrieval-augmented generation with multimodal index.

1. Create src/pipeline/rag_retriever.py:
   - Class MultimodalRetriever
   - __init__(encoder: CLIPVisionEncoder, index_path: str = None)
   - Method index_documents(docs: list[dict]) -> None
     Each doc: {"image": PIL.Image or path, "text": str, "metadata": dict}
     Embed with CLIP (both image and text), store in FAISS
   - Method retrieve(query_image=None, query_text=None, top_k=3) -> list[dict]
     Encode query, search FAISS, return top_k docs with scores
   - Method save_index(path) / load_index(path)
   - Hybrid retrieval: combine image similarity + text similarity scores

2. Update src/pipeline/multimodal_qa.py:
   - Add optional retriever parameter to __init__
   - In answer(), if retriever provided:
     * Retrieve relevant documents
     * Inject retrieved context into prompt
     * Include retrieved images in model input if supported

3. Create NB-09-rag-multimodal.ipynb:
   - Section 1: Build a knowledge base
     * Collect 30-50 image+caption pairs (use COCO or Flickr sample)
     * Embed all with CLIP and index in FAISS
     * Visualize the embedding space with UMAP
   - Section 2: Image retrieval demo
     * Query with a new image → show top-3 retrieved images + captions
     * Query with text → show retrieved images
   - Section 3: RAG-augmented QA
     * Ask a question about a topic
     * Retrieve relevant context
     * Feed retrieved context + question to LLM
     * Compare answer quality: with vs without RAG
   - Section 4: Analysis
     * When does RAG help? When does it hurt?
     * Retrieval latency vs answer quality tradeoff
```

---

## Phase 7 — Deployment

```
Read AGENT.md and PROGRESS.md. We are on Phase 7.

Task: Serve the model via OpenAI-compatible API and connect to OpenWebUI.

1. Create src/serving/api_server.py:
   - FastAPI app
   - POST /v1/chat/completions — OpenAI-compatible
     Accepts messages with image_url content type (base64 or URL)
     Returns streaming SSE or full JSON response
   - GET /health — returns {"status": "ok", "model": model_id}
   - GET /v1/models — returns model list (OpenAI format)
   - API key auth via Authorization header
   - Request/response logging with loguru

2. Create docker/docker-compose.yml:
   services:
     api:
       build: .
       ports: ["8000:8000"]
       volumes: ["./model_cache:/model_cache"]
       environment: [HF_TOKEN, MODEL_ID, API_KEY]
       deploy:
         resources:
           reservations:
             devices: [{driver: nvidia, count: 1, capabilities: [gpu]}]
     
     openwebui:
       image: ghcr.io/open-webui/open-webui:main
       ports: ["3000:8080"]
       environment:
         - OPENAI_API_BASE_URL=http://api:8000/v1
         - OPENAI_API_KEY=${API_KEY}
       depends_on: [api]

3. Create NB-10-deployment.ipynb:
   - Section 1: Test the API locally
     * Start FastAPI with uvicorn
     * Send a request with requests library
     * Show the full request/response cycle
   - Section 2: OpenWebUI setup
     * Docker commands to start the stack
     * Screenshot walkthrough of connecting OpenWebUI to custom API
     * Test multimodal chat in the UI
   - Section 3: Performance optimization
     * Compare: naive serving vs vLLM (if GPU)
     * Compare: float16 vs 4-bit quantization latency
     * Batch size experiments
   - Section 4: Ollama alternative
     * Convert model to GGUF
     * Serve with Ollama
     * Connect to OpenWebUI
```

---

## General Purpose Prompts

### "Explain this component to me"
```
I'm learning to build a multimodal QA system. Explain [COMPONENT NAME] to me:
- What problem does it solve?
- How does it work at a high level (no code yet)?
- What is the input shape and output shape?
- What would happen if we removed it?
- Give me a concrete analogy

Then show me the simplest possible implementation in PyTorch (10-20 lines max).
```

### "Debug this shape error"
```
I'm building a multimodal QA system. I have a shape error:
[PASTE ERROR]

Here is the relevant code:
[PASTE CODE]

The expected flow is:
Image [batch, 3, 224, 224]
→ CLIP encoder → [batch, 197, 1024]
→ Projector → [batch, 197, 4096]
→ LLM input

Diagnose the shape error and fix it. Explain what went wrong.
```

### "Write the unit test for this"
```
Write a pytest unit test for this class/function:
[PASTE CODE]

Requirements:
- No real model downloads (use random tensors or mock)
- Test the happy path
- Test edge cases (empty input, wrong dtype, wrong device)
- Use fixtures where appropriate
- Each test function name should describe what it tests
```

### "Review my notebook"
```
Review this Jupyter notebook cell for a learner building a multimodal QA system:
[PASTE CELL CODE]

Check for:
1. Correctness: is the logic right?
2. Learning value: does this cell teach something clearly?
3. Missing steps: what should be explored but isn't?
4. Code quality: Python best practices, type hints where appropriate

Suggest improvements.
```
