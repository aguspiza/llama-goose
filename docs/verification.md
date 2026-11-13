# How llama.goose is checked against llama.cpp

Every part of the port is compared with what llama.cpp makes of the same
input: the same tokens, the same grammar text, the same bytes of generated
text. This page keeps the notes on each part; the [README](../README.md)
has the summary.

## Status in detail

| Part | State | Checked against llama.cpp |
|---|---|---|
| ggml (C, CPU backend) | built by `goose build`, unchanged | `tests/ggml_smoke` |
| Unicode, pre-tokenizer splits | ported | `tests/unicode_split` |
| SPM and BPE tokenizers | ported | all 15 SPM/BPE test vocabs give identical tokens (`tests/tokenizer_0`) |
| UGM, RWKV, PLaMo2 and PLaMo3 tokenizers | ported | through `tools/simple` with T5, RWKV and PLaMo 2 and 3 models |
| WPM tokenizer | loads, does not tokenize yet | |
| model loading, KV cache (one, unified for several sequences, as `-kvu`), recurrent states, decode; T5's encoder | ported for 65 architectures (see [Architectures](../README.md#architectures)); recurrent models and T5 hold one sequence | `tools/simple`, `tools/perplexity`, `tools/batched`, `tools/parallel` |
| other architectures | not yet (see [Architectures](../README.md#architectures)) | |
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
| `llama-server`: `/completion`, `/v1/completions`, `/v1/chat/completions` (tools, reasoning, `response_format`, streaming), `/apply-template`, `/tokenize`, `/detokenize`, `/props`, `/models`, `/health`; the prompt cache | ported, with llama-server's 4 slots in one KV cache (`-np` for another number; one for a recurrent model): the requests of the slots at work are decoded together, a batch at a time, a client that leaves stops its generation, and `/health` and CORS preflights are answered meanwhile; except the prompt cache in RAM (`--cache-ram`), a cache split per slot, embeddings, rerank, infill, token probabilities, multimodal, the web UI and the other APIs (Anthropic, Responses) | `scripts/compare-server.py`: 29 requests, the same responses, slots included, with Qwen3 and Qwen2.5 |
| `llama-cli`: the chat over a server it starts (or `--server-base`), with reasoning shown apart, `/regen`, `/clear`, `/read`, `/glob`, Ctrl+C to stop a reply | ported, except images, audio and `-hf` downloads; the server runs as a child process, not a thread (Goose threads cannot hold the vocab's and Jinja's slices) | `tools/cli`, by hand with LFM2.5 and LFM3 |
| the rest of `common/` (args, downloads) | not yet | |

## tools/simple

`tools/simple` is llama.cpp's `examples/simple`. With gemma-3 270M (Q8_0)
and greedy sampling it writes byte for byte the text llama.cpp's
`llama-simple` writes, over 200 generated tokens on four prompts, at the
same speed (77-93 tokens/s on both, on a Ryzen 7 5700U).

The same text needs the same number of threads once the cache holds 512
cells or more (a context of more than 256 tokens): ggml then splits the
attention of a single token over the threads and adds their parts, so the
last bits of the logits depend on how many there are, in llama.cpp too
(`llama-completion -t 4` and `-t 8` part ways after some 270 tokens with
Qwen3 0.6B). `llama-simple` runs on ggml's fixed 4 threads and `tools/simple`
on half the logical CPUs, as `llama-completion` and `llama-server` do, which
is faster; so on a machine where that is not 4, the two write the same text
up to 256 tokens of context, and beyond it with `tools/simple` built for 4
threads. `tools/completion` and `tools/server` take their originals' number
of threads, and write the same at any length.

Checked the same way, all byte for byte over 150 generated tokens and three
prompts each, at llama.cpp's speed: gemma-3 270M (Q8_0), LFM2.5 230M (Q8_0
and Q4_K_M; lfm2 mixes attention with short convolutions that keep a
recurrent state) and MiniCPM5 2B (llama architecture, Q4_K_M).

## tools/completion

`tools/completion` gives the same text as `llama-completion` with these
architectures too (checked with qwen3, gemma2, granitehybrid and phi3, text
and conversation), given the same `-c`: without it llama-completion fits the
context to the free memory (`common/fit`, not ported), and Phi-3.5's rope
factors depend on the context's size.

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

## What bit-identical results need from the build

Q4_0 models crashed in ggml's 8x8 repack kernels, in this port and in
llama.cpp's own MinGW build: gcc spills AVX values to the stack with aligned
moves that Win64 does not align for. The CPU backend is now compiled with
`-Wa,-muse-unaligned-vector-move`, as the programs already were.

Two details matter for bit-identical results. The weights read by mul_mat
go to the CPU's repack buffers, as llama.cpp places them. And ggml's base
is compiled without `-march=native`, as CMake compiles it, while its CPU
backend is compiled with it: with FMA, gcc fuses the multiply-adds of the
reference quantizers and their results differ in the last bits.

## tools/perplexity

`tools/perplexity` is llama-perplexity's default mode (one sequence per
batch). Over the first chunks of llama.cpp's README at `-c 512`, with
`-b 512` and `-b 128`, it prints the same perplexities as llama-perplexity
for all three architectures. It scores the logits on a pool of Goose
threads, as llama.cpp does with its own; with gemma3's 262k-token
vocabulary that is as much work as the model. Run time is within 10% of
llama-perplexity's (4.9 s against 5.4 s for gemma-3-270m, 4 chunks).
Both tools default to llama.cpp's thread count (half the logical CPUs
above 4); `-t` sets it.

## tools/batched and tools/parallel

`tools/batched` and `tools/parallel` are examples/batched and
examples/parallel, on one cache unified for all the sequences (llama.cpp's
`-kvu`; the cache split per sequence is not ported). Each cell of the cache
keeps a position and the set of sequences it belongs to, so a shared prompt
is stored once, and cells are taken as `llama_kv_cache::find_slot` takes
them. When a batch has tokens of several sequences, each sequence's queries
attend to the rows of its own cells (`ggml_flash_attn_ext_rows`), as in
llama.cpp. The logits depend on which tokens share a batch, so both sides
must decode the same batches to be compared: `tools/parallel` with 3
clients, 5 requests, a shared system prompt and greedy sampling logs the
same requests and responses as llama-parallel on 23 attention-only models
of the table above (gemma-3's sliding window among them), and on Qwen3 0.6B
also with a seed, without the shared prompt and with `--junk`;
`tools/batched` writes the same 4 sequences as llama-batched.

## Chat templates

`tests/chat_template` applies every built-in template, and the template of
every case in llama.cpp's `test-chat-template.cpp`, to its conversation,
and compares with what libllama writes (180 checks); it also renders each
case's Jinja template through `common/chat.goose` and compares with what
libcommon's `common_chat_templates_apply` writes (35 checks). The cases and
llama.cpp's answers are in `tests/data/chat_cases.txt`;
`scripts/gen-chat-cases.sh` regenerates it from a llama.cpp build.

## Grammars

`llama/grammar.goose` is llama.cpp's GBNF grammars: the same parse (rule
ids and elements, compared by `tests/grammar` with what llama.cpp's parser
makes of each test grammar), the same pushdown stacks, and the same
rejection of the candidate tokens, code point by code point. With the
grammars of llama.cpp's `grammars/` (JSON, lists, arithmetic, chess, C,
Japanese) and a conversation with a grammar, `tools/completion --grammar`
writes the same text as `llama-completion` on the three models.
`scripts/gen-grammar-cases.sh` regenerates `tests/data/grammar_cases.txt`.

## Tool calls and reasoning

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

## JSON schemas

`common/json_schema.goose` is `json-schema-to-grammar.cpp` with the schema
model of `json-schema.cpp`: the schema is read with the standard library's
`json` into a table of values, parsed into schema nodes, and each node
becomes a rule. `tests/json_schema` compares the grammar text with what
llama.cpp writes for every case of its test, errors and warnings included
(`scripts/gen-schema-cases.sh` regenerates `tests/data/schema_cases.txt`),
and `tools/completion -j` writes the same text as `llama-completion`.

## Jinja

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

## tools/server

`tools/server` is `llama-server`, over [Plod](https://github.com/kaitk/plod)
and [goose-http](https://github.com/kaitk/goose-http) (HTTP/1.1 with
keep-alive and chunked responses, worker threads) and `common/jdoc` (JSON
written as nlohmann writes it). Goose threads may only use flat globals, and
the model, vocab and templates are not flat, so the program's first thread
loads them and serves the requests, as the lane Plod's workers send them to.
The workers answer `/health`, CORS preflights, and 503 "Loading model" until
the model is loaded; request bodies cross to the main thread as goose-http
shared blobs, and responses come back as mail messages, the events of a
stream as they are generated.

A request that generates takes a slot, as in llama-server: 4 of them by
default, each a sequence of one KV cache they share, which keeps the slot's
last prompt for a request that starts as it does (the slot is chosen as
llama-server chooses it: the most similar prompt, else an empty slot, else
the one used longest ago). The main thread runs the slots at work together:
each step decodes one batch, with the token each generating slot sampled last
and the next tokens of the prompts, and samples a token for each slot. The
sampler, the grammar, the chat parser and the response so far are globals,
as they were with one slot, so each request's are kept as bytes while the
globals hold another's (`tools/stash`: the server lists them once, and the
list both writes them and reads them back). Requests beyond the slots wait
in their order; the others (`/props`, `/tokenize`, ...) are served between
steps. At each step the main thread checks whether each client is still
there, and frees the slot of one that is not. When the cache is full, a slot
not at work gives up its cells, as in llama-server. A recurrent model's
cache holds one sequence, so it has one slot.

`scripts/compare-server.py` sends 29 requests to it and to llama.cpp's
`llama-server`, both with their 4 slots, and compares the responses, less
ids, times and the cached-token counts: completions with and without
streaming, stop strings, grammars and JSON schemas, chat with tool calls
(streamed as deltas), prefilled assistant messages and errors. All 29 are
the same, the slot each request got included, with Qwen3 0.6B and Qwen2.5
0.5B, also with a context of 256 tokens, where slots are purged; and with
Mamba 130M and one slot. The cached-token counts differ because
llama-server reuses an SWA or recurrent cache only from a checkpoint, while
this server keeps every attention cell and reuses more. Requests served at
the same time are not compared with llama-server's, since which batch a
token falls in depends on when each request arrives, and that changes the
last bits of its logits; in a test build that decoded one slot per batch,
four requests at once gave the responses of the same four served one after
another.

## The console and tracing

Goose programs get their arguments in the Windows code page, where
llama.cpp's get UTF-8 from their manifest; `console::utf8_args` reads the
command line as UTF-8 (a Japanese prompt failed before it).

`LLAMA_GOOSE_TRACE=1` prints sums of layer 0's tensors to compare with
llama.cpp's eval callback.
