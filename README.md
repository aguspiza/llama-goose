# llama.goose

A plan to port [llama.cpp](https://github.com/ggml-org/llama.cpp) to
[Goose](https://aardappel.github.io/goose/) and to build it with `goose build`
instead of CMake. The goals are shorter build times, lower memory use, faster
inference and source code that is easier to read.

## Status

Following llama.cpp `1b6935b22` (2026-11-02), ggml 0.28.0.

| Part | State | Checked against llama.cpp |
|---|---|---|
| ggml (C, CPU backend) | built by `goose build`, unchanged | `tests/ggml_smoke` |
| Unicode, pre-tokenizer splits | ported | `tests/unicode_split` |
| SPM and BPE tokenizers | ported | all 15 SPM/BPE test vocabs give identical tokens (`tests/tokenizer_0`) |
| UGM, RWKV, PLaMo2 and PLaMo3 tokenizers | ported | through `tools/simple` with T5, RWKV and PLaMo 2 and 3 models |
| WPM tokenizer | loads, does not tokenize yet | |
| model loading, KV cache, recurrent states, decode; T5's encoder | ported for 65 architectures (see Architectures below) | `tools/simple`, `tools/perplexity` |
| other architectures | not yet (see below) | |
| sampling: penalties, DRY, top-k, typical, top-p, min-p, XTC, top-n-sigma, temperature (dynamic too), mirostat 1 and 2, greedy, dist; the model's recommended settings | ported | `tools/completion` |
| GBNF grammars: the parser, the engine, constrained sampling (rejection sampling, as common_sampler) | ported | `tests/grammar`: the 470 cases of `test-grammar-parser.cpp` and `test-grammar-integration.cpp`; `tools/completion --grammar` |
| JSON schema to grammar | ported | `tests/json_schema`: the 83 cases of `test-json-schema-to-grammar.cpp`, the same grammar text; `tools/completion -j`/`-jf` |
| lazy grammars (word and token triggers) | ported | through the tool call grammars below |
| sampling: adaptive-p, infill; regex grammar triggers | not yet | |
| built-in chat templates (`llama_chat_apply_template`, 54 of them, and their detection from a Jinja source) | ported | `tests/chat_template`, `tools/simple_chat` |
| Jinja engine (`common/jinja`: lexer, parser, runtime, builtins, input marking, caps) | ported | `tests/jinja`: all 383 cases of `test-jinja.cpp`, and its caps, stats and string-parts checks |
| Jinja chat templates (`common_chat_templates_init`/`apply`, the prompt) | ported | `tests/chat_template`, `tools/completion --jinja` |
| PEG parsers (`common/peg-parser`) | ported | `tests/peg`: test-gbnf-generation's cases and parse cases of test-basic |
| tool call and reasoning parsers: the differential autoparser, the 22 dedicated handlers of `common/parsers`, their grammars and triggers, and the output parsed back into a message | ported | `tests/autoparser`: llama.cpp's analysis of all 75 templates of `models/templates`; `tests/chat_peg`: all 517 cases of test-chat.cpp's `test_template_output_peg_parsers` (prompt, grammar, triggers, and the messages of whole and cut outputs) |
| `llama-completion`: text, conversation (`-cnv`, the default with a chat template), interactive, reverse prompts, the console's colors | ported, except context shift, prompt cache files and self-extend | `tools/completion` |
| `llama-server`: `/completion`, `/v1/completions`, `/v1/chat/completions` (tools, reasoning, `response_format`, streaming), `/apply-template`, `/tokenize`, `/detokenize`, `/props`, `/models`, `/health`; the prompt cache | ported with one slot (requests are generated one at a time; `/health` and CORS preflights are answered meanwhile, and a client that leaves stops its generation), except embeddings, rerank, infill, token probabilities, multimodal, the web UI and the other APIs (Anthropic, Responses) | `scripts/compare-server.py`: 29 requests, the same responses with gemma-3 and LFM2.5 |
| `llama-cli`: the chat over a server it starts (or `--server-base`), with reasoning shown apart, `/regen`, `/clear`, `/read`, `/glob`, Ctrl+C to stop a reply | ported, except images, audio and `-hf` downloads; the server runs as a child process, not a thread (Goose threads cannot hold the vocab's and Jinja's slices) | `tools/cli`, by hand with LFM2.5 and LFM3 |
| the rest of `common/` (args, downloads) | not yet | |

`tools/simple` is llama.cpp's `examples/simple`. With gemma-3 270M (Q8_0)
and greedy sampling it writes byte for byte the text llama.cpp's
`llama-simple` writes, over 200 generated tokens on four prompts, at the
same speed (77-93 tokens/s on both, on a Ryzen 7 5700U).

Checked the same way, all byte for byte over 150 generated tokens and three
prompts each, at llama.cpp's speed: gemma-3 270M (Q8_0), LFM2.5 230M (Q8_0
and Q4_K_M; lfm2 mixes attention with short convolutions that keep a
recurrent state) and MiniCPM5 2B (llama architecture, Q4_K_M).

### Architectures

Each one writes, with greedy sampling, byte for byte the text llama.cpp's
`llama-simple` writes on three prompts (`The capital of France is`,
`def fibonacci(n):`, `Once upon a time, in a small village`), 60 to 150
tokens each. The models are the smallest GGUFs found on Hugging Face; none
over 4 GB in Q4.

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

Not ported yet:

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

`tools/completion` gives the same text as `llama-completion` with these
architectures too (checked with qwen3, gemma2, granitehybrid and phi3, text
and conversation), given the same `-c`: without it llama-completion fits the
context to the free memory (`common/fit`, not ported), and Phi-3.5's rope
factors depend on the context's size.

Q4_0 models crashed in ggml's 8x8 repack kernels, in this port and in
llama.cpp's own MinGW build: gcc spills AVX values to the stack with aligned
moves that Win64 does not align for. The CPU backend is now compiled with
`-Wa,-muse-unaligned-vector-move`, as the programs already were.

Two details matter for bit-identical results. The weights read by mul_mat
go to the CPU's repack buffers, as llama.cpp places them. And ggml's base
is compiled without `-march=native`, as CMake compiles it, while its CPU
backend is compiled with it: with FMA, gcc fuses the multiply-adds of the
reference quantizers and their results differ in the last bits.
`tools/completion` is llama.cpp's `llama-completion`. With a fixed seed it
writes the same text as `llama-completion` (eight runs over the three
architectures, with penalties, typical, XTC, dynamic temperature,
top-n-sigma, and the sampling settings a GGUF recommends): the RNG is
`std::mt19937` and the uniform draws follow libstdc++. In conversation mode
it reads the user's turns from stdin; with the built-in templates (the
default, as in `llama-completion`) and with `--jinja`, three-turn chats on
the three models, in English and Spanish, with and without colors, give
the same bytes as `llama-completion` fed the same lines.
`scripts/compare-completion.sh` runs both and compares (llama.cpp writes
stdout in Windows text mode, `\r\n`, which it reads as `\n`).

`tools/perplexity` is llama-perplexity's default mode (one sequence per
batch). Over the first chunks of llama.cpp's README at `-c 512`, with
`-b 512` and `-b 128`, it prints the same perplexities as llama-perplexity
for all three architectures. It scores the logits on a pool of Goose
threads, as llama.cpp does with its own; with gemma3's 262k-token
vocabulary that is as much work as the model. Run time is within 10% of
llama-perplexity's (4.9 s against 5.4 s for gemma-3-270m, 4 chunks).
Both tools default to llama.cpp's thread count (half the logical CPUs
above 4); `-t` sets it.

`tests/chat_template` applies every built-in template, and the template of
every case in llama.cpp's `test-chat-template.cpp`, to its conversation,
and compares with what libllama writes (180 checks); it also renders each
case's Jinja template through `common/chat.goose` and compares with what
libcommon's `common_chat_templates_apply` writes (35 checks). The cases and
llama.cpp's answers are in `tests/data/chat_cases.txt`;
`scripts/gen-chat-cases.sh` regenerates it from a llama.cpp build.

`llama/grammar.goose` is llama.cpp's GBNF grammars: the same parse (rule
ids and elements, compared by `tests/grammar` with what llama.cpp's parser
makes of each test grammar), the same pushdown stacks, and the same
rejection of the candidate tokens, code point by code point. With the
grammars of llama.cpp's `grammars/` (JSON, lists, arithmetic, chess, C,
Japanese) and a conversation with a grammar, `tools/completion --grammar`
writes the same text as `llama-completion` on the three models.
`scripts/gen-grammar-cases.sh` regenerates `tests/data/grammar_cases.txt`.

`common/chat_apply.goose` is `common_chat_templates_apply` with Jinja the
whole way: the conversation and tools as OpenAI JSON, the template's
workarounds, and the output's parser from `common/parsers` (in
`common/chat_parsers.goose`) or from the differential autoparser
(`common/autoparser.goose`), which renders the template with pairs of sample
conversations and reads its markers off the differences. The PEG parsers
(`common/peg.goose`) also write the grammar; their GBNF names carry parser
ids, so the parsers are made in the order GCC evaluates the C++ expressions
(right operand of `+` first), and the grammars come out the same text.
`tests/chat_peg` runs test-chat.cpp's 517 PEG cases against what llama.cpp
makes of them (`scripts/gen-peg-chat-cases.sh`), and `tests/autoparser` the
analysis of every template (`scripts/gen-autoparser-cases.sh`).

`common/json_schema.goose` is `json-schema-to-grammar.cpp` with the schema
model of `json-schema.cpp`: the schema is read with the standard library's
`json` into a table of values, parsed into schema nodes, and each node
becomes a rule. `tests/json_schema` compares the grammar text with what
llama.cpp writes for every case of its test, errors and warnings included
(`scripts/gen-schema-cases.sh` regenerates `tests/data/schema_cases.txt`),
and `tools/completion -j` writes the same text as `llama-completion`.

Goose programs get their arguments in the Windows code page, where
llama.cpp's get UTF-8 from their manifest; `console::utf8_args` reads the
command line as UTF-8 (a Japanese prompt failed before it).

`common/jinja*.goose` is llama.cpp's Jinja engine, the one its chat
templates run on. Its values are indices into arenas (llama.cpp shares
them through `shared_ptr`), strings keep which parts came from the user's
input, and errors are `exception::throw`s with llama.cpp's messages.
`tests/jinja` renders every template case of llama.cpp's `test-jinja.cpp`
and compares with what llama.cpp's engine makes of it (output, error, or
not implemented); `scripts/gen-jinja-cases.sh` regenerates
`tests/data/jinja_cases.txt`. Numbers and dates go through libc
(`common/shim.c`: `snprintf`, `strtod`, `strftime`), as in llama.cpp.
`tools/simple_chat` is examples/simple-chat with a `-s` seed: with the seed
fixed in llama.cpp's too, three-turn chats in English and Spanish give the
same bytes on all three models. Like simple-chat it runs 4 threads
(libllama's default, not common's): ggml's flash attention splits a long
KV cache by thread, so the thread count changes the last bits.

`tools/server` is `llama-server`, over [Plod](https://github.com/kaitk/plod)
and [goose-http](https://github.com/kaitk/goose-http) (HTTP/1.1 with
keep-alive and chunked responses, worker threads) and `common/jdoc` (JSON
written as nlohmann writes it). Goose threads may only use flat globals, and
the model, vocab and templates are not flat, so the program's first thread
loads them and serves the requests one at a time, as the lane Plod's workers
send them to. The workers answer `/health`, CORS preflights, and 503
"Loading model" until the model is loaded; request bodies cross to the main
thread as goose-http shared blobs, and responses come back as mail messages,
the events of a stream as they are generated. Between tokens and prompt
batches the main thread checks whether the client is still there, and stops
when it is not. With one
slot on both, `scripts/compare-server.py` sends 29 requests to it and to
llama.cpp's `llama-server` and compares the responses, less ids, times and
the cached-token counts: completions with and without streaming, stop
strings, grammars and JSON schemas, chat with tool calls (streamed as
deltas), prefilled assistant messages and errors. All 29 are the same with
gemma-3 270M and LFM2.5 230M. The cached-token counts differ because
llama-server reuses an SWA or recurrent cache only from a checkpoint, while
this server keeps every attention cell and reuses more.

`LLAMA_GOOSE_TRACE=1` prints sums of layer 0's tensors to compare with
llama.cpp's eval callback.

## Results so far

Measured on 2026-11-05 on the machine below, with ggml already compiled
and cached on both sides. The C++ figures come from the `.ninja_log` of a
CPU-only `Release` build of llama.cpp (llama-server, -completion,
-perplexity, -simple, -tokenize and test-tokenizer-0, without the web UI).

**Build time.** Each Goose program is one C file, compiled by one gcc
process on one core:

| Program | Lines of C | `-O1` | `-O3` |
|---|---:|---:|---:|
| `tools/server` | 384k | 47 s | 106 s |
| `tools/completion` | 172k | 26 s | 49 s |
| `tools/perplexity` | 37k | 11 s | 15 s |
| `tools/simple` | 36k | 11 s | |
| `tools/cli` | 18k | 4 s | 6 s |
| `scripts/test.sh` (build and run every test) | | 105 s | |

Goose writes the server's 16 MB of C in about 3 s; gcc takes the rest.
llama.cpp spends about 1,600 CPU-seconds on the same tools without ggml
(`common/`, `src/`, the server, cpp-httplib, mtmd), spread over 16 cores.
So the Goose build costs about 15 times less CPU: about 100 s for the five
tools one after another, on one core at a time. The comparison flatters
Goose: the C++ side has more than a hundred architectures, argument
parsing, downloads and multimodal, which are not ported yet.

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

**Binary size**, both static, without symbols:

| Program | Goose | llama.cpp |
|---|---:|---:|
| server | 4.4 MB | 16.8 MB |
| completion | 3.2 MB | 9.6 MB |
| perplexity | 2.4 MB | 9.6 MB |
| simple | 2.4 MB | 5.8 MB |

Here too part of the difference is what is not ported yet.

**Inference speed and memory** are llama.cpp's: the kernels, the weights
and the KV cache are ggml's in both.

The rest of this README is the feasibility study written before the port.

## Measurements

All numbers come from this machine (16 threads, MinGW gcc, Ninja) and from
llama.cpp at `faec40e87` (2026-07-11), built for the CPU only in
`RelWithDebInfo`.

### Where the build time goes

The whole build takes **2,108 CPU-seconds** for 365 targets, taken from `.ninja_log`.

| Component        | CPU time | Share |
|------------------|---------:|------:|
| `common/`        |   777 s  | 36.9% |
| `src/` (libllama)|   499 s  | 23.6% |
| `tools/server`   |   373 s  | 17.7% |
| (other paths)    |   145 s  |  6.9% |
| `tools/mtmd`     |   129 s  |  6.1% |
| `ggml/src`       |   123 s  |  5.8% |
| `cpp-httplib`    |    53 s  |  2.5% |
| linking          |     8 s  |  0.4% |

The slowest files are all C++ glue code. `common/arg.cpp` takes 78 s,
`chat.cpp` 66 s, `peg-parser.cpp` 62 s and `server-task.cpp` 61 s. Eleven
files in `common/` and the server include nlohmann/json.

The compute core (`ggml/src`) takes **less than 6%** of the build time.

### Goose compile speed

Compiling the 428-line [raytracer](https://github.com/aguspiza/raytracing-goose)
took 0.2 s with `goose` (it emits 6,774 lines of C) and 1.9 s with `gcc -O2`.
For comparison, `arg.cpp` (4,137 lines of C++) takes 78 s.

### Source size

| Part                    | Lines (C/C++/CUDA/GLSL) |
|-------------------------|------------------------:|
| `ggml/src` (everything) | 300k |
| of which `ggml-cpu`     |  87k |
| of which `ggml-cuda`    |  38k |
| of which `ggml-vulkan`  |  31k |
| `src/`                  |  74k |
| `common/`               |  35k |
| `tools/`                |  45k |

## What to expect

### Build time: large gain, from porting the glue code

About 78% of the build time is C++ in `common/`, `src/` and the server.
Code generated by Goose is plain C, which compiles one or two orders of
magnitude faster than C++ that uses templates and json.hpp. If those three
parts are ported, a CPU-only build could drop from about 35 CPU-minutes to a
few CPU-minutes (estimate, not measured).

There is a risk. Goose emits **one C file per program**, so gcc cannot split
that file across cores. A 100k-line Goose program becomes one very large C file,
and its compile time must be measured early (see step 1 below).

### The cheap alternative was tried: unity builds

CMake can merge the `.cpp` files of each target into a few large ones
(`-DCMAKE_UNITY_BUILD=ON`), with no change to the sources. On this fork
(branch `mtp-clean`, all 87 executables, tests included):

| Build  | Wall time | CPU time |
|--------|----------:|---------:|
| normal |    445 s  |  7,021 s |
| unity  |   ~328 s  |  3,746 s |

The machine was shared with other jobs, and another process killed the
unity build by mistake once (it was resumed), so these are rough numbers.
It did not work as is:

- Four `static` helpers have the same name in two files each (`chat.cpp`,
  `llama-kv-cache.cpp`, `models/gemma4.cpp`) and `mtmd-helper.cpp` refuses
  mtmd's internal headers. Each needed a `SKIP_UNITY_BUILD_INCLUSION` line,
  and new clashes will appear as upstream adds files.
- The resulting test programs failed to start (`0xc0000139`, a function
  missing from a DLL). Found later: Git's MinGW was first on PATH, with an
  older `libstdc++-6.dll`; the normal build has the same problem there. So
  this one was not unity's fault.

It needs patches to the CMake files and constant care on every upstream
merge, so it was dropped. Precompiled headers were ruled out too, because
they complicate the build.

### Goose and C in one program

Goose mixes with C source code. A sidecar file next to the Goose root
(`llama.build` beside `llama.goose`) lists C files with `c_sources`,
`c_include`, `c_defines` and `c_libs`, and `goose build` compiles and links
them with the program (docs/packages.md §3). Goose calls that C through
`extern fn`. ggml can therefore stay C, built as part of the Goose program,
while the logic above it moves to Goose.

What crosses the boundary: scalars, flat structs, slices and pointers to
them. Returns are a scalar or a flat struct, so ggml objects (`ggml_context`,
`ggml_tensor`, a backend) cross as opaque `i64` handles.

### Memory: no gain, no loss

Memory use in llama.cpp is mostly the model weights (memory-mapped) and the
KV cache. The language does not change either of them.

Goose threads **share no memory**: every worker gets its own copy of the
globals (spec §11.2). If the weights or the KV cache lived in Goose memory,
each thread would need its own copy. With ggml kept in C they stay in C
memory, shared by ggml's own C thread pool, as they are today. The possible
saving is the C++ runtime and the heap churn of `std::string` and json in
the server: megabytes, not gigabytes.

### Inference speed: same as today

- CPU token generation is limited by memory bandwidth and by hand-written
  SIMD kernels (AVX2/AVX-512/NEON quantized dot products). Goose has no
  intrinsics. `simd fn` relies on gcc auto-vectorization and forbids fusing
  multiply-adds unless the baseline does. A Goose rewrite of `ggml-cpu`
  would be slower.
- The CUDA, Vulkan and Metal backends cannot be written in Goose.
- The thread pool in ggml splits each matrix multiply across threads that
  share the same tensors. Goose's no-sharing model cannot express that.

All three stay in C, mixed into the Goose program, so the speed of the
kernels does not change. Goose code calls ggml per graph node (a few
thousand direct C calls per token), never per element, so the boundary
should not show up in tokens per second.

### Readability: real gain where the logic lives

`src/` (model architectures, vocabulary, sampling, KV cache bookkeeping) and
`common/` (argument parsing, chat templates, grammars) are where a reader
gets lost in C++. These parts are also the ones that dominate build time,
so readability and build time point to the same parts of the code.

## Proposed plan

1. **Measure the risk first.** Port one self-contained file from `common/`,
   such as `json-schema-to-grammar.cpp`, to Goose. Compare its compile time,
   its output on the existing tests and the size of the generated C. Then
   generate a synthetic 50k-line Goose program to see how gcc handles one
   large C file.
2. **ggml with `goose build`.** List ggml's C sources (CPU backend first,
   then Vulkan) in the `.build` sidecar, so `goose build` compiles them with
   the program. It is checked by running `test-backend-ops` against the
   CMake build. Vulkan shader generation and CUDA need a step the sidecar
   cannot express yet. Avoid `cmake_target` in the sidecar, since the goal
   is to drop CMake.
3. **Port `common/`, then `src/`**, file by file, behind the existing
   `llama.h` C API (Goose `export fn`). The C++ build stays as the reference:
   every step must give identical tokens on a fixed prompt and seed.
4. **Server last.** It depends on json and HTTP, and Goose's `export fn`
   calls must stay on one thread. Its threading model needs its own design.

Libraries that need porting (for example a json library) get their own
folders next to this one, named `<name>.goose`.

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
tools/              programs: simple.goose
scripts/            build, test, weekly sync, generators
tests/              Goose programs that exit non-zero on failure
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
list them for the link, so `scripts/build.sh` writes those itself. See "Results so far" for why `-O1`.

ggml's C is compiled once (about 1.5 minutes) and cached under `~/.goose`.
After that each program takes a few seconds to a minute ("Results so
far"): well under a second to a few seconds for Goose to write its C, the
rest gcc on that one file.

Goose keeps arrays of up to 256 elements in static data and builds larger
ones element by element when the program starts. gcc took 30 s to optimize
that function for the Unicode tables, so they are one binary file the
program embeds (`embed_file`).

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
