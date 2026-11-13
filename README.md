# llama.goose

**[llama.cpp](https://github.com/ggml-org/llama.cpp) ported to
[Goose](https://aardappel.github.io/goose/).** The tokenizers, the model
graphs, sampling, grammars, chat templates, tool calls and `llama-server`
are rewritten in Goose; ggml, the tensor library under them, stays the C it
is and is built by `goose build`.

It writes the same text as llama.cpp, byte for byte, at about the same
speed. It builds with a tenth of the CPU time, without CMake, into binaries
a third of the size. It is a work in progress: it runs on the CPU only, and
a good part of llama.cpp is [not ported yet](#what-is-missing).

Follows llama.cpp `1b6935b22` (2026-11-02), ggml 0.28.0.

## At a glance

| | llama.goose | llama.cpp |
|---|---|---|
| Generated text | the same bytes, checked on 65 architectures | the reference |
| Building server, completion, perplexity, simple and cli | 151 s, on one core | about 1,600 CPU-seconds |
| `server` binary | 6.0 MB | 17.6 MB |
| Server speed, one request (Qwen3 0.6B) | 66 tokens/s | 67 tokens/s |
| Server speed, four requests at once | 130 tokens/s | 138 tokens/s |
| Build system | `goose build` and one shell script | CMake |
| Backends | CPU | CPU, CUDA, Vulkan, Metal, ... |
| Architectures | 65 | more than a hundred |
| Platforms checked | Windows, MinGW-w64 gcc | Windows, Linux, macOS, Android, ... |

Where the figures come from: [Numbers](#numbers).

## What you get

- **The same output.** Each of the 65 architectures writes, with greedy
  sampling, the bytes `llama-simple` writes; with a seed, `tools/completion`
  writes what `llama-completion` writes; the server gives `llama-server`'s
  responses to 29 requests. llama.cpp's own test cases for grammars, JSON
  schemas, Jinja, chat templates and tool calls are run against what
  llama.cpp makes of them. See [How it is checked](#how-it-is-checked).
- **Short builds.** A program is one C file that gcc compiles on one core:
  73 s for the server, 4 to 36 s for the others. ggml is compiled once and
  cached.
- **Small binaries.** 3 to 6 MB, where llama.cpp's are 6 to 18 MB.
- **The same speed.** The kernels, the weights and the KV cache are ggml's
  on both sides, and the code around them keeps what llama.cpp does per
  token: a thread pool that stays, a graph run again while its shape
  holds, a partial sort for top-k.
- **No CMake.** The Goose compiler and gcc; `scripts/build.sh` builds a
  program, `scripts/test.sh` builds and runs the tests.
- **Source to read.** About 34,000 lines of Goose (`llama/` 12,700,
  `common/` 15,300, `tools/` 5,500). Each file starts with the llama.cpp
  sources it ports, and a script reports every week which of them changed
  upstream.

## What is missing

Compared with llama.cpp at the same commit. ❌ is not ported; 🟡 is ported
in part.

| Area | Missing |
|---|---|
| ❌ GPU | Every backend but the CPU's: CUDA, Vulkan, Metal, SYCL, OpenCL, ... ggml's sources are here, but only its CPU backend is built. |
| 🟡 Platforms | Built and checked on Windows with MinGW-w64 gcc only. `goose build` with MSVC does not find the C headers yet. Linux and macOS have their branches in the code and have not been checked. |
| 🟡 Architectures | 65 are ported. Not the ones with no model of 4 GB or less in Q4 (the large mixtures of experts, the 7B and larger dense models), nor the experts of gemma4, nemotron_h and k2-horizon. [The list](#architectures). |
| ❌ Other kinds of model | Embedding and encoder-only models (the bert family and its WPM tokenizer, t5encoder), diffusion models, audio decoders. |
| ❌ Multimodal | Images and audio (`mtmd`, `--mmproj`). The vision models in the list run as text models. |
| ❌ Adapters and state | LoRA adapters, control vectors, saving and loading a context's state (session and prompt cache files). |
| 🟡 KV cache | One cache unified for all the sequences (llama.cpp's `-kvu`). Not the cache split per sequence, context shift or self-extend. A recurrent model and T5 hold one sequence. |
| 🟡 Sampling | Not adaptive-p or infill; no regex triggers for lazy grammars. |
| 🟡 `llama-server` | Not embeddings, rerank, infill, token probabilities (`n_probs`, `logprobs`), the prompt cache in RAM (`--cache-ram`), the web UI, the Anthropic and Responses APIs, nor the endpoints [Programs](#programs) does not name. |
| 🟡 `llama-completion`, `llama-cli` | Not context shift, prompt cache files or self-extend; the cli has no images, audio or `-hf` downloads. |
| ❌ `common/` | The argument parser (each program reads the flags it supports itself), downloads (`-hf`), fitting the context to the free memory. |
| ❌ Other programs | `llama-quantize`, `llama-bench`, `llama-embedding`, `llama-mtmd-cli`, `llama-tts`, `llama-imatrix` and the rest of llama.cpp's tools and examples. |

## What is ported

| Part | Checked against llama.cpp by |
|---|---|
| ✅ ggml: unchanged C, built by `goose build` | `tests/ggml_smoke` |
| ✅ Tokenizers: SPM, BPE, UGM, RWKV, PLaMo2, PLaMo3; Unicode and the pre-tokenizer splits | the same tokens on llama.cpp's 15 SPM and BPE test vocabs; the others through their models |
| ✅ Models: loading, the graphs of 65 architectures, the KV cache for several sequences, recurrent states, T5's encoder | the same text as `llama-simple`, the same perplexities as `llama-perplexity`, the same batches as `llama-batched` and `llama-parallel` |
| ✅ Sampling: penalties, DRY, top-k, typical, top-p, min-p, XTC, top-n-sigma, temperature (dynamic too), mirostat 1 and 2, greedy, dist; the settings a model recommends | the same text as `llama-completion` with a seed |
| ✅ Grammars: GBNF, lazy grammars, JSON schema to grammar | the 470 cases of llama.cpp's grammar tests and the 83 of its JSON schema test: the same parse, the same grammar text |
| ✅ Chat templates: the 54 built-in ones and the Jinja engine | the 383 cases of `test-jinja.cpp`; 215 checks of prompts against libllama and libcommon |
| ✅ Tool calls and reasoning: the PEG parsers, the differential autoparser, the 22 dedicated handlers | llama.cpp's analysis of its 75 templates; the 517 cases of `test-chat.cpp` |

### Programs

| Program | llama.cpp's | What it does |
|---|---|---|
| `tools/simple` | `llama-simple` | greedy generation from a prompt |
| `tools/simple_chat` | `llama-simple-chat` | a chat with the model's template |
| `tools/completion` | `llama-completion` | text and conversation, interactive, reverse prompts, grammars, JSON schemas, `--jinja`, colors |
| `tools/perplexity` | `llama-perplexity` | its default mode |
| `tools/batched`, `tools/parallel` | `llama-batched`, `llama-parallel` | several sequences in one batch |
| `tools/server` | `llama-server` | `/completion`, `/v1/completions`, `/v1/chat/completions` (tools, reasoning, `response_format`, streaming), `/apply-template`, `/tokenize`, `/detokenize`, `/props`, `/models`, `/health`; 4 slots decoded together, the prompt cache of each slot |
| `tools/cli` | `llama-cli` | the chat over a server it starts, with reasoning shown apart, `/regen`, `/clear`, `/read`, `/glob` |

## Quick start

It needs the Goose compiler, v2026.4 (not the `goose` AI agent that may be
on PATH), and MinGW-w64 gcc.

```bash
scripts/build.sh tools/server.goose        # GOOSE=path/to/goose.exe to pick the compiler
tools/server -m model.gguf --port 8080

curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'
```

The first build compiles ggml (about 1.5 minutes) and caches it. More in
[Building](#building).

## Numbers

Measured on 2026-11-13 on a Ryzen 7 5700U (8 cores, 16 threads) with
Windows 11, MinGW-w64 gcc 15.2 and Goose v2026.4. llama.cpp is a CPU-only
`Release` build with the same gcc.

**Build time**, with ggml already compiled and cached. Each Goose program
is one C file, compiled by one gcc process at `-O1`:

| Program | Lines of C | Build |
|---|---:|---:|
| `tools/server` | 646k | 73 s |
| `tools/completion` | 272k | 36 s |
| `tools/perplexity` | 128k | 19 s |
| `tools/simple` | 128k | 19 s |
| `tools/cli` | 18k | 4 s |
| `scripts/test.sh` (build and run every test) | | 109 s |

Goose writes the server's C in about 5 s; gcc takes the rest (a little over
twice as long at `-O3`, for no gain in speed: see [Building](#building)).
llama.cpp
spent about 1,600 CPU-seconds on the same tools without ggml (`common/`,
`src/`, the server, cpp-httplib, mtmd), spread over 16 cores, in its
`.ninja_log` of 2026-11-05. So the Goose build costs about ten times less
CPU. The comparison flatters Goose: the C++ side has the architectures,
argument parsing, downloads and multimodal that are not ported yet.

**Binary size**, without symbols:

| Program | llama.goose | llama.cpp |
|---|---:|---:|
| server | 6.0 MB | 17.6 MB |
| completion | 3.9 MB | 10.1 MB |
| perplexity | 3.0 MB | 10.1 MB |
| simple | 3.0 MB | 6.0 MB |

Here too part of the difference is what is not ported yet.

**Speed**, Qwen3 0.6B Q4_K_M, 8 threads, 200 tokens per request:

| | llama.goose | llama.cpp |
|---|---:|---:|
| server, one request | 66 tokens/s | 67 tokens/s |
| server, four requests at once, in all | 130 tokens/s | 138 tokens/s |
| completion | 67 tokens/s | 67 tokens/s |

The two programs were run in turns in one session, and each figure is the
median of five or six runs. Read the columns against each other, not as
what the machine can do: single runs vary by about 5%, and the machine's
state moves both programs by 10% from one session to another (earlier the
same day the server gave 73 tokens/s for one request, and llama-server
77). A model this small is where the code around ggml shows; with MiniCPM5
2B, at some 100 ms per token, no difference was measured.

**Memory** is ggml's on both sides: the weights and the KV cache.

## How it is checked

- **Text.** `tools/simple` against `llama-simple` on three prompts for each
  architecture, greedy; `tools/completion` against `llama-completion` with
  seeds, samplers, grammars and conversations; `tools/perplexity`,
  `tools/batched` and `tools/parallel` against theirs.
- **The server.** `scripts/compare-server.py` sends 29 requests to it and
  to `llama-server`, both with their 4 slots, and compares the responses:
  completions with and without streaming, stop strings, grammars, JSON
  schemas, chat with tool calls, errors.
- **llama.cpp's tests.** `tests/` runs the cases of llama.cpp's own tests
  (tokenizers, grammars, JSON schemas, Jinja, chat templates, tool call
  parsers) and compares with what llama.cpp answered, kept in
  `tests/data/` and regenerated from a llama.cpp build by `scripts/gen-*`.

One thing to know when comparing: once a context holds more than 256
tokens, ggml splits the attention of a single token over the threads, and
the last bits of the logits depend on how many there are, in llama.cpp
too. The same text then needs the same `-t`.

The notes on each part are in [docs/verification.md](docs/verification.md).

## Architectures

Each one writes, with greedy sampling, byte for byte the text llama.cpp's
`llama-simple` writes on three prompts (`The capital of France is`,
`def fibonacci(n):`, `Once upon a time, in a small village`), 60 to 150
tokens each. The models are the smallest GGUFs found on Hugging Face; none
over 4 GB in Q4.

<details>
<summary>The 65 architectures and the model each was checked with</summary>

| Architecture | Model checked | Notes |
|---|---|---|
| afmoe | Trinity-Nano-Preview Q4_K_M | gated attention, sliding windows, experts with a selection bias |
| arcee | AFM-4.5B Q4_K_M | YaRN, ReLU² |
| bitnet | bitnet_b1_58-large Q4_K_M | |
| bloom | bloomz-560m Q8_0 | ALiBi |
| chatglm | glm-edge-1.5b-chat, glm-edge-4b-chat Q4_K_M | |
| cohere2 | tiny-aya-global Q4_K_M | sliding windows; attention and feed-forward side by side |
| ernie4_5 | ERNIE-4.5-0.3B-PT Q4_K_M | |
| exaone | EXAONE-3.5-2.4B-Instruct Q4_K_M | |
| exaone4 | EXAONE-4.0-1.2B Q4_K_M | |
| falcon-h1 | Falcon-H1-0.5B-Instruct Q4_K_M | attention and mamba2 side by side |
| gemma | gemma-1.1-2b-it Q4_K_M | |
| gemma2 | gemma-2-2b-it Q4_K_M | attention soft-capping |
| gemma3 | gemma-3-270m-it Q8_0 | |
| gemma3n | gemma-3n-E2B-it Q4_K_M | AltUp, Laurel, per-layer inputs, shared KV |
| gemma4 | gemma-4-E2B-it Q4_K_M | head sizes per layer kind, shared KV |
| glm4 | GLM-OCR Q8_0 (text) | M-RoPE; the NextN layer is not run |
| gpt2 | gpt2 Q8_0 | |
| gptneox | pythia-410m Q8_0 | |
| granite | granite-3.1-2b-instruct Q4_K_M | |
| granitehybrid | granite-4.0-h-350m Q4_K_M | mamba2 and attention |
| granitemoe | granite-3.1-1b-a400m-instruct Q4_K_M | experts |
| hrm_text | DFM-Mimir-v1.5 Q4_K_M | a low and a high stack run in cycles, each pass with its own cache |
| hunyuan-dense | Hunyuan-0.5B-Instruct Q4_K_M | |
| internlm2 | internlm2_5-1_8b-chat Q4_K_M | |
| jais | jais-family-590m-chat Q8_0 | ALiBi |
| jamba | AI21-Jamba-Reasoning-3B Q4_K_M | mamba and attention |
| k2-horizon | K2-Horizon-1B, K2-Horizon-4B Q4_K_M | RMS norms over groups |
| lfm2 | LFM2.5-230M Q8_0/Q4_K_M, LFM2.5-1.2B-Thinking Q4_0 | |
| lfm3 | LFM3-1.7B, LFM3-Omni-2B, LFM3-Audio-2B (text) | |
| limite | limite-1b-violetto Q4_K_M | value embeddings, exclusive self attention, dense connections across layers |
| llama | MiniCPM5 2B, Mellum-4b Q4_K_M | |
| maincoder | Maincoder-1B Q4_K_M | |
| mamba | mamba-130m F16 | |
| mamba2 | mamba2-130m Q8_0 | |
| minicpm | MiniCPM-2B-sft Q4_K_M | granite's graph |
| minicpm3 | MiniCPM3-4B Q4_K_M | low rank Q and KV |
| mistral3 | Ministral-3-3B-Instruct-2512 Q4_K_M | YaRN, attention temperature |
| mpt | replit-code-v1_5-3b Q4_0 | ALiBi |
| nanbeige | Nanbeige4.2-3B Q4_K_M | layers that run twice |
| nemotron | Nemotron-Mini-4B-Instruct Q4_K_M | |
| nemotron_h | Nemotron3-Nano-4B Q4_K_M | mamba2, attention and feed-forward layers |
| olmo | OLMo-1B-0724-hf Q4_K_M | |
| olmo2 | OLMo-2-0425-1B-Instruct Q4_K_M | |
| openelm | OpenELM-270M-Instruct Q8_0 | |
| pangu-embedded | openPangu-Embedded-1B-V1.1 Q4_K_M | |
| phi2 | phi-2 Q4_K_M | |
| phi3 | Phi-3.5-mini-instruct Q4_K_M | long and short rope factors |
| plamo2 | plamo-2-1b Q4_K_M | PLaMo2 tokenizer |
| plamo3 | plamo-3-translate-0.8b Q4_K_M | PLaMo3 tokenizer, sliding windows |
| qwen | Qwen-1_8B-Chat Q4_K_M | |
| qwen2 | Qwen2.5-0.5B-Instruct Q4_K_M | |
| qwen2vl | Qwen2-VL-2B-Instruct Q4_K_M (text) | M-RoPE |
| qwen3 | Qwen3-0.6B Q4_K_M | |
| qwen35 | Agents-A1-4B Q4_K_M, Qwen3.5-4B Q4_0 | gated delta nets |
| qwen3vl | Qwen3-VL-2B-Instruct Q4_K_M (text) | IMROPE, deepstack |
| refact | Refact-1_6-base Q4_K_M | ALiBi |
| rwkv6 | RWKV-v6-World-v2.1-1.6B Q4_K_M | RWKV tokenizer |
| rwkv7 | rwkv7-g1h-1.5b Q4_K_M | |
| smallthinker | SmallThinker-4B-A0.6B-Instruct Q4_0 | experts |
| smollm3 | SmolLM3-3B Q4_K_M | |
| spark2_5 | Spark-X2.5-1.7B Q4_K_M | a gate per head, rope sizes per layer kind |
| stablelm | stablelm-2-zephyr-1_6b Q4_K_M | |
| starcoder | tiny_starcoder_py Q8_0 | |
| starcoder2 | starcoder2-3b Q4_K_M | |
| t5 | flan-t5-small F16 | encoder and decoder, UGM tokenizer |

</details>

<details>
<summary>Not ported yet, and why</summary>

- The ones without a model of 4 GB or less in Q4: lfm2moe (LFM2.5-8B-A1B is
  5.2 GB), olmoe (4.2 GB), beam/bailingmoe3 (Moon-Beam-8B-A1B, 4.9 GB),
  apertus, jais2, maple, mellum, the 7B and larger dense models (baichuan,
  xverse, orion, codeshell, deci, falcon, plamo, chameleon, ...) and the
  large mixtures of experts (deepseek2 and later, qwen2moe, qwen3moe,
  qwen35moe, qwen3next, openai-moe, glm4-moe, llama4, command-r, dbrx, grok,
  arctic, minimax, kimi, step35, seed-oss, flint, laguna, kolibri1, ...).
  Their graphs reuse much of what is ported (experts, MLA-like attention,
  YaRN).
- granite_swa: the only small GGUF (granite-swash-3b-a600m) does not load in
  llama.cpp (its shared expert is not fused). graniteswitch: only a BF16
  preview, and its adapter router is not ported. qwen4exp: only a
  random-weight test file, on which llama.cpp asserts. gptj: llama.cpp has
  the name but no graph.
- plm: llama.cpp itself fails on PLM-1.8B (`GGML_ASSERT(a->ne[0] == b->ne[0])`
  when it writes the 2D value tensor to the cache), so there is nothing to
  check against.
- Embedding and encoder-only models (bert family, t5encoder, the embedding
  gemmas), diffusion models (dream, llada) and audio decoders: they need
  `llama-embedding` or their own programs, not `llama-simple`.
- The experts of gemma4 and nemotron_h, k2-horizon's experts, value experts,
  Q/K norms and output gate, limite with four taps, and the MTP layers some
  models carry; they abort with "not ported yet".

</details>

## Layout

```
UPSTREAM            the llama.cpp commit this port follows
ggml/               ggml, copied unchanged from llama.cpp, as three Goose packages:
  ggml.goose          bindings (extern fn), and gguf.goose
  ggml.build          the base's C, built as CMake builds it (not native)
  cpu/                package ggml_cpu: the CPU backend, built -march=native
  shim/               package ggml_shim: shim.c, the calls Goose cannot make directly
  units/              wrappers for sources whose file names collide
llama/              libllama in Goose (package `llama`):
  unicode.goose       pre-tokenizer splits, NFD/NFC (unicode_data.* generated)
  ecma_regex.goose    a backtracking matcher with std::regex's rules
  vocab.goose         vocabularies and tokenizers (vocab_pre.goose generated)
  model.goose         hyperparameters and weights
  context.goose       KV cache, graphs, decode
  sampler.goose       the sampler chain
  grammar.goose       GBNF grammars
  chat.goose          the built-in chat templates
common/             llama.cpp's common/ (package `common`): the Jinja engine, chat
                    templates and their parsers, PEG, JSON schemas, JSON documents, the console
tools/              the programs, and the packages they share (net/, proc/, stash/)
scripts/            build, test, compare with llama.cpp, weekly sync, generators
tests/              Goose programs that exit non-zero on failure; data/ has llama.cpp's answers
docs/               the notes on each check, and the study written before the port
```

Every Goose file ported from llama.cpp starts with a line naming its
sources, which is how the sync script knows what to convert again:

```goose
// port-of: src/llama-vocab.cpp src/unicode.cpp
```

## Building

Goose's own compiler, v2026.4 (not the `goose` AI agent that may be on
PATH), and MinGW gcc. `goose build` with MSVC does not find the C headers
yet. The server's HTTP packages, goose-http and Plod, are fetched by the
compiler at the versions `tools/server.goose` names, the first time it is
built.

```
scripts/build.sh tests/ggml_smoke.goose     # GOOSE=path/to/goose.exe to pick the compiler
scripts/test.sh                             # build and run everything in tests/
```

`scripts/build.sh` runs goose build's own commands (from `--dry-run`), with
the program's C at `-O1 -march=native` (`OPT=3` for another level). ggml
gets `-O3` from its sidecar. It compiles only the package objects that are
missing, so the program's C is compiled once, not twice as a `goose build`
followed by the optimized compile would. v2026.4 names a package's objects
by a hash, and only a whole `goose build` writes the response files that
list them for the link, so `scripts/build.sh` writes those itself.

ggml's C is compiled once (about 1.5 minutes) and cached under `~/.goose`.
After that each program takes a few seconds to a minute ([Numbers](#numbers)):
well under a second to a few seconds for Goose to write its C, the
rest gcc on that one file.

Goose keeps arrays of up to 256 elements in static data and builds larger
ones element by element when the program starts. gcc took 30 s to optimize
that function for the Unicode tables, so they are one binary file the
program embeds (`embed_file`).

<details>
<summary>Why <code>-O1</code> (measured on 2026-11-05)</summary>

**`-O1` is the default.** Goose code compiled at `-O1` runs as fast as at
`-O3`: the heavy work is in ggml, which stays at `-O3`. At `-O0` the Goose
code that runs per token (parsing, sampling, grammars) slows generation by
a third.

| | `-O3` | `-O1` | `-O0` |
|---|---:|---:|---:|
| server, LFM3 1.7B Q4_K_M, chat, 300 tokens: prompt | 114 t/s | 115 t/s | 114 t/s |
| server, same: generation | 22.5 t/s | 22.5 t/s | 15.0 t/s |
| completion, LFM2.5 230M Q8_0, 300 tokens | 67.3 t/s | 67.7 t/s | |
| perplexity, gemma-3 270M, 4 chunks of 512 | 4.6 s | 4.6 s | |

The output is the same at every level (same text, same perplexity).

</details>

## Weekly sync with upstream

```
scripts/sync-upstream.sh            # report only: sync-report.md
scripts/sync-upstream.sh --apply    # copy ggml, regenerate ggml-version.h, move UPSTREAM
scripts/test.sh
```

The script fetches the fork's `master` into `../llama.cpp` (only a `git
fetch`: no branch or file there changes) and sorts every file changed since
`UPSTREAM` into:

- **ported files to convert again**, each with the Goose files that port it,
- **not ported yet** (`src/`, `common/`, `tools/`, `include/`),
- **ggml**, copied unchanged by `--apply`, which also warns when ggml starts
  building a source that `ggml/ggml.build` does not list,
- **ignored**: docs, CI, other backends, scripts.

`git -C ../llama.cpp diff <old> <new> -- <path>` shows what changed in one
file, and the report prints that command with both commits filled in.

## Background

[docs/feasibility-study.md](docs/feasibility-study.md) is the study written
before the port: where llama.cpp's build time goes, what Goose could and
could not change, and the plan that was followed.
