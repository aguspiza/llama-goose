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
| model loading, KV cache (unified for several sequences, as `-kvu`, or a stream per sequence), recurrent states, decode; T5's encoder | ported for 65 architectures (see [Architectures](../README.md#architectures)); recurrent models and T5 hold one sequence | `tools/simple`, `tools/perplexity`, `tools/batched`, `tools/parallel` |
| other architectures | not yet (see [Architectures](../README.md#architectures)) | |
| sampling: penalties, DRY, top-k, typical, top-p, min-p, XTC, top-n-sigma, temperature (dynamic too), mirostat 1 and 2, greedy, dist; the model's recommended settings | ported | `tools/completion` |
| GBNF grammars: the parser, the engine, constrained sampling (rejection sampling, as common_sampler) | ported | `tests/grammar`: the 526 cases of `test-grammar-parser.cpp` and `test-grammar-integration.cpp`; `tools/completion --grammar` |
| JSON schema to grammar | ported | `tests/json_schema`: the 87 cases of `test-json-schema-to-grammar.cpp`, the same grammar text; `tools/completion -j`/`-jf` |
| lazy grammars (word and token triggers) | ported | through the tool call grammars below |
| sampling: adaptive-p, infill; regex grammar triggers | not yet | |
| built-in chat templates (`llama_chat_apply_template`, 54 of them, and their detection from a Jinja source) | ported | `tests/chat_template`, `tools/simple_chat` |
| Jinja engine (`common/jinja`: lexer, parser, runtime, builtins, input marking, caps) | ported | `tests/jinja`: all 406 cases of `test-jinja.cpp`, and its caps, stats and string-parts checks |
| Jinja chat templates (`common_chat_templates_init`/`apply`, the prompt) | ported | `tests/chat_template`, `tools/completion --jinja` |
| PEG parsers (`common/peg-parser`) | ported | `tests/peg`: test-gbnf-generation's cases and parse cases of test-basic |
| tool call and reasoning parsers: the differential autoparser, the 22 dedicated handlers of `common/parsers`, their grammars and triggers, and the output parsed back into a message | ported | `tests/autoparser`: llama.cpp's analysis of all 83 templates of `models/templates`; `tests/chat_peg`: all 603 cases of test-chat.cpp's `test_template_output_peg_parsers` (prompt, grammar, triggers, and the messages of whole and cut outputs) |
| `llama-completion`: text, conversation (`-cnv`, the default with a chat template), interactive, reverse prompts, the console's colors | ported, except prompt cache files | `tools/completion` |
| `llama-server`: `/completion`, `/v1/completions`, `/v1/chat/completions` (tools, reasoning, `response_format`, streaming), `/apply-template`, `/tokenize`, `/detokenize`, `/props`, `/models`, `/health`; the prompt cache | ported, with llama-server's 4 slots in one KV cache (`-np` for another number; one for a recurrent model): the requests of the slots at work are decoded together, a batch at a time, a client that leaves stops its generation, and `/health` and CORS preflights are answered meanwhile; except the prompt cache in RAM (`--cache-ram`), a cache split per slot, embeddings, rerank, infill, token probabilities, multimodal, the web UI and the other APIs (Anthropic, Responses) | `scripts/compare-server.py`: 29 requests, the same responses, slots included, with Qwen3 and Qwen2.5 |
| `llama-cli`: the chat over a server it starts (or `--server-base`), with reasoning shown apart, `/regen`, `/clear`, `/read`, `/glob`, Ctrl+C to stop a reply | ported, except images and audio; the server runs as a child process, not a thread (Goose threads cannot hold the vocab's and Jinja's slices) | `tools/cli`, by hand with LFM2.5 and LFM3 |
| images in `llama-server` (`--mmproj`: `image_url`, `multimodal_data`, `--media-path`) and in `llama-cli` (`--image`, `/image`) | ported for SigLIP with the gemma3 and idefics3 projectors; not http URLs, audio, video | [below](#images---mmproj) |
| downloads (`common/download`: `-hf`, `-hff`, `-hft`, `--offline`, into the Hugging Face cache) | ported, through libcurl loaded at run time; not `-mu`, Docker, the cache's lock files | [below](#downloads--hf) |
| `--fit` (`common/fit`): the context left to the model (no `-c`, as llama.cpp's default) cut to leave `-fitt` MiB of the host memory, projected from the weights, the cache and the compute buffer of llama.cpp's worst-case graphs | ported for the host memory | [below](#--fit) |
| loading the weights: `-lm auto\|none\|mmap\|mlock\|mmap+mlock\|dio`, `-lzm on\|auto\|off`, `--repack`/`-nr`, `-ot`, `-cmoe`, `-ncmoe` and their `LLAMA_ARG_*` variables; the tied output's copy (`TENSOR_DUPLICATED`); the unused tensors left out | ported for the CPU (the overrides have only the `CPU` buffer type to go to) | [below](#loading-the-weights) |
| the argument parser (`common/arg`): llama.cpp's options for each program, their `LLAMA_ARG_*` variables, `-h`, `--completion-bash`, the errors | ported; the options of what is not ported fail; no config files or remote presets | [below](#the-argument-parser) |

## The argument parser

`common/arg` is llama.cpp's `common/arg.cpp`: `arg_opts.goose` holds the
options of `common_params_parser_init` (generated once from it, then kept by
hand), each with its names, value hints, environment variable, the programs
that show it and its help, whose defaults are those of the program's
`common_params`. As in llama.cpp, the environment variables are read first,
then the command line overrides them; a `--` name may be written with `_`.
`-h` prints the options by section, and `--completion-bash` the bash
completion script. The programs set their defaults before parsing, as
llama.cpp's do (`perplexity`'s `-c 512`, `batched`'s prompt, ...).

`-h` gives the same bytes as llama.cpp's program for `completion`,
`perplexity`, `batched`, `parallel`, `mtmd_cli` and the server (all their
options, the cli's 583 lines too, for which there is no llama-cli build to
compare with). With `completion`, these give the same message and exit
code as `llama-completion`: an unknown argument, a missing value, `-n abc`
and `-c 99999999999` (`stoi`), `--temp x` (`stof`), `--color maybe`,
`--repeat-penalty 0`, `--dry-penalty-last-n -2`, `-ot foo`, `-ot a=GPU`,
no model, and the variables: `LLAMA_ARG_N_PREDICT=abc`,
`LLAMA_ARG_FLASH_ATTN=bad`, `LLAMA_ARG_CONTEXT_SHIFT=maybe`, a variable the
command line overrides (its warning) and `LLAMA_ARG_NO_CONTEXT_SHIFT`. The
programs give the same output as before through it: completion's text
(short and long prompts, samplers, `--samplers`, `--sampling-seq`, `-r`,
`-f`, `-st`, `--jinja`, variables for `-n` and `-c`) on the first 10
models of the regression list, perplexity's, batched's and parallel's on
Qwen3, mtmd_cli's on SmolVLM and gemma-3 4B, and the server's 29 requests
on Qwen3 and the image cases on SmolVLM.

The differences, all known:

- The options of what is not ported (LoRA, control vectors, speculative
  decoding, devices and RPC, the CPU masks and priorities, the router
  server, ...) are listed in `-h`, but fail when given, as an invalid value
  named "not ported", where llama.cpp takes them.
- The system config files (`config.ini`, read through `common/preset`) and
  the remote presets of `-hf` repositories are not read.
- `--version` prints the port's build, not llama.cpp's.
- llama.cpp writes its warning about an argument given twice through its
  log, with a time stamp and a level before it and on its own thread, so
  after the other warnings; the port writes it plainly, in order.
- llama.cpp writes its text in Windows text mode (`\r\n`).

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

A prompt longer than a ubatch (512 tokens) is decoded in ubatches of 512,
as `llama_decode` splits a batch: one graph over the whole batch gave other
logits (the matrix products sum in another order), and with Qwen3 a
570-token prompt then went on with other text. Greedy, with a short prompt
and a 3000-byte one, the first 10 models of the regression list write the
same text as `llama-completion`.

Past the context, with `--context-shift` (`-c 256`, 600 tokens, greedy, 8
threads), `tools/completion` writes the same text as `llama-completion` on
66 of the 68 models of the regression list: a shift drops half the tokens
after `--keep`, moves the rest back, and rotates their keys by the distance
(the K-shift graph, with each architecture's rope; through f32 for the
models whose K norm's weight comes after the rope). The two others: PLM,
which llama.cpp does not load either, and flan-t5, which `tools/completion`
does not run (its encoder is not called there). Self-extend (`-gan 4 -gaw
128` and `-gan 2 -gaw 256`, `-c 1024`) gives the same text with Qwen3. The
server's context shift, with the default `n_keep` and `n_discard` and with
others, gives the same responses as llama-server's with Qwen3 and gemma-3
270M, whose sliding-window layers can shift only when their cache is as big
as the base one, as in llama.cpp.

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
examples/parallel, on one cache unified for all the sequences with `-kvu`,
else split into a stream per sequence. Each cell of the cache
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

Without `-kvu`, each sequence has a stream of `n_ctx / n_seq_max` cells
(padded to 256) of its own, in the same tensors, one stream after another,
as `llama_kv_cache` with `n_stream > 1`: a batch splits into ubatches of the
same number of tokens of consecutive sequences (`split_equal`, sequential),
each sequence's queries attend to its stream's cells through a view of the
streams the ubatch touches, and copying a sequence to another stream copies
the stream's cells, keys and values. `tools/parallel` without `-kvu` logs
the same requests and responses as llama-parallel with Qwen3 0.6B (3 and 4
clients, 40 to 128 tokens, also with `-pps`) and gemma-3 270M, whose
sliding-window layers have a stream each too; with a context too small for
the prompts, both retry with smaller batches and fail alike. `tools/batched`
without `-kvu` fails as llama-batched does, since its prompt belongs to all
the sequences at once.

## tools/mtmd_cli

`tools/mtmd_cli` is llama-mtmd-cli's single turn: the prompt and its images
through the chat template, the images decoded by stb_image (llama.cpp's
`vendor/`, unchanged), resized as Pillow does in fixed point, sliced, and
encoded by a vision transformer whose embeddings the text model reads in
place of tokens. So far the SigLIP encoder with the projectors of gemma3
(the image fitted in a square, average-pooled, its tokens attending to each
other) and idefics3 (SmolVLM: a grid of slices and an overview, the pixel
shuffle). Its embeddings are the same bytes as `MTMD_DEBUG_EMBEDDINGS` dumps,
and its responses, greedy, are the same as llama-mtmd-cli's with SmolVLM
256M and gemma-3 4B, on the jpeg of llama.cpp's tests and on images of
600x600, 600x400 (padded slices) and 1300x900 (a 3x2 grid).

## --fit

Without `-c`, completion, parallel, mtmd_cli and the server take the
model's trained context (for each stream), as llama.cpp's programs do since
their default became 0, and `common/fit` cuts it when the weights, the
cache and the compute buffer would leave less than `-fitt` MiB (1024) of the
host memory's total. The compute buffer is measured on the port's own
worst-case graphs (a ubatch of 512 tokens over every sequence, and one of a
token per sequence, over the whole cache), allocating nothing. For it to be
llama.cpp's to the byte, the port's input embeddings now build the graph as
llama.cpp's `build_inp_embd` does: the token rows, the embeddings given and
the mixed batch all in it, one computed (`ggml_build_forward_select`), and
gemma's scale of the token rows as an input. The text of every model is
unchanged (tools/simple: 67 of the 68 models, as before, PLM being one llama.cpp
cannot load; mtmd_cli with SmolVLM and
gemma-3 4B).

Qwen3 0.6B, with `-fitt 29000` so that the cut is reached on a machine of
32 GB: llama-completion projects 167 + 4480 + 346.77 MiB (weights, cache,
compute) at 40960 cells and 928 MiB at 4096, and picks 24064; the port
projects the same and picks 24064. Through the servers' `/props`
(`cmpfit.sh`), the same context and the same warning in each case:

| model | flags | n_ctx of a slot |
|---|---|---|
| Qwen3 0.6B | (none) / `-fit off` | 40960 |
| | `-fitt 29000` / `30500` | 26624 / 13056 |
| | `-np 2 -fitt 29000` (streams) | 13312 |
| | `-np 3 -kvu -fitt 29000` | 26624 |
| | `-fitt 30500 -fitc 8192` | 13056 |
| | `-c 32768 -fitt 30500` / `-c 0 -fitt 30500` | 32768 / 40960, with the warning |
| | `-fitt 31900` | 4096, with the warning |
| gemma-3 270M (SWA) | `-fitt 31000` / `-np 2 -fitt 31500` | 32768 / 32768 |
| LFM2.5 230M (hybrid) | `-fitt 31500` | 44544 |
| Jamba 3B (hybrid) | `-fitt 29000` | 262144 |

## Loading the weights

`common/arg` reads the flags in every program but `simple` and
`simple_chat`, and `ggml/cpu/load.c` holds the OS's part
(mapping, prefetch, mlock, unbuffered reads). As llama.cpp does:

- `-lm auto` (the default) and `mmap` map the file: the tensors of the
  plain CPU buffer type point into the mapping (`CPU_Mapped`), and only the
  ones repacked into `CPU_REPACK` are copied. `none` reads them into host
  buffers; `mlock` locks those buffers, `mmap+mlock` the mapping; `dio`
  reads with unbuffered I/O (`FILE_FLAG_NO_BUFFERING`, `O_DIRECT`), at the
  file system's alignment. `--repack`/`-nr` turns `CPU_REPACK` on or off.
- `-ot <pattern>=CPU`, `-cmoe` and `-ncmoe N` are checked (an unknown buffer
  type lists the ones there are and fails), and, with mmap, a match gives
  llama.cpp's warning to use `--load-mode none`. With the CPU alone they
  move nothing.
- `-lzm`: gemma4's per-layer token embeddings stay in the mapping whatever
  the mode (with `auto`, above 4 GiB), out of the prefetch and of mlock, and
  the rows of each batch's tokens are read ahead before it is decoded.
- When the output is the token embeddings, and mul_mat takes another buffer
  type than get_rows, the output is a copy of them in that buffer type
  (llama.cpp's `TENSOR_DUPLICATED`): gemma-3n's `CPU_REPACK` grows from
  861.75 to 1149.75 MiB, as llama.cpp's.
- The tensors no loader asks for (GLM-OCR's MTP layer, 220 MiB) are neither
  placed nor read, with llama.cpp's warning; the server's `/models` counts
  only the ones loaded.

`cmpload.sh` runs llama-completion and tools/completion with the same flags
and compares the text and the loader's lines (the buffers and their sizes,
the warnings of mlock, unbuffered I/O, the overrides and the unused
tensors). Over the 68 models, with the default and with `-lm none`: 134 of
136 the same; the two others are flan-t5, whose text completion did not give
before either (its encoder). Qwen3 0.6B with each mode, `-nr`, `-ot`
(`CPU`, and an unknown type), `-cmoe`, `-ncmoe 2`, and through the
`LLAMA_ARG_*` variables; granite MoE, gemma-3n, mamba, Jamba; gemma-4 with
`-lzm on`, `auto`, `off`, and `on` with `none` and `mmap+mlock`: all the
same. mlock fails on Windows for the 167.90 MiB buffer of Qwen3 with the same
warning as llama.cpp's. tools/simple: 67 of the 68 models as before (PLM is
one llama.cpp cannot load); mtmd_cli with SmolVLM and gemma-3 4B, and
parallel with `-kvu` and with a stream per sequence: the same. The server
(`compare-server.py`, Qwen3), with the default and with `-lm none`: 29 of 29
the same. `--fit` (`cmpfit.sh`): Qwen3 `-fitt 29000` 26624 (with `-lm none`
too) and gemma-3n `-fitt 28000` 32768, as llama-server's.

## Downloads (-hf)

`common/download` asks the Hub for the branch's commit and the file tree,
picks the files as llama.cpp does, and downloads them (a HEAD, then up to
3 attempts that resume the partial file) into huggingface_hub's cache:
the blob named by its hash, linked from the commit's snapshot, or moved
there when Windows refuses the link (llama.cpp's "degraded mode"). The
reference build here has no TLS, so the downloads themselves were checked
by hash, and the choice and the cache through llama.cpp reading what the
port wrote:

- `mtmd_cli -hf ggml-org/SmolVLM-256M-Instruct-GGUF` downloads the model and
  its mmproj, both Q8_0 (the repository has no Q4_K_M); llama-mtmd-cli
  `--offline` on that cache loads the same files and writes the same
  caption as the port.
- A partial file of 50 MB left in `blobs/` is completed, and the model's
  SHA-256 is its oid.
- On fake cached repositories, with `--offline`, both pick the same model
  and mmproj for no tag, `:q8_0`, `:F16` beside a BF16, `:BF16`, a split
  model in a subdirectory with mmprojs above, beside and elsewhere, a repo
  with neither Q4_K_M nor Q8_0, `-hff`, and both refuse an unknown file.
- completion, perplexity, cli (through its server) and the server load a
  model by `-hf`; an unknown repository and a missing quant fail with the
  Hub's message and the list of GGUF files.
- With MSYS2's libcurl (OpenSSL, the Windows certificate store), as the
  Windows release ships it, beside the program and with MinGW off PATH.

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
`tests/chat_peg` runs test-chat.cpp's 603 PEG cases against what llama.cpp
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
default, each a sequence of one KV cache they share; with `-np`, each has a
stream of the cache to itself unless `-kvu` is given (without `-c`, of the
context the model was trained with, as `common_fit_params` sets it when the
memory holds it). The cache keeps the slot's
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
0.5B, also with a context of 256 tokens, where slots are purged; with gemma-4
E2B and gemma-3 270M (also with 256 tokens), whose sliding-window layers'
cells the server saves as context checkpoints, splitting the prompts as
llama-server does to make them, and evaluates a later prompt from; and with
Mamba 130M and one slot. With `-np 2`, a stream per slot, all 29 are the
same too with Qwen3 0.6B and gemma-3 270M; with `-c 1024 -np 4 --context-shift`,
slots of 256 cells, the four context-shift requests of Qwen3 are the same,
shifted as many times. The cached-token counts differ because
llama-server reuses a recurrent cache only from a checkpoint, and this port
does not checkpoint a recurrent state yet. The sliding-window layers
keep a smaller cache of their own, as llama.cpp's `llama_kv_cache_iswa`: with
gemma-3 270M, 1500 tokens of llama-completion and 3 clients of 700 tokens of
llama-parallel, past the windows, where that cache overwrites its cells, are
the same. Requests served at
the same time are not compared with llama-server's, since which batch a
token falls in depends on when each request arrives, and that changes the
last bits of its logits; in a test build that decoded one slot per batch,
four requests at once gave the responses of the same four served one after
another.

### Images (--mmproj)

With an mmproj the server reads images as llama-server does: a chat
message's `image_url` parts (a `data:` URL or bare base64; `file://` under
`--media-path`; not http URLs yet) and `/completion`'s `multimodal_data`
become markers in the prompt, and `mtmd` turns them into the image's
tokens, `TOKEN_NULL` in the slot's tokens with the image beside them, its
id the SHA-256 of its file (llama.cpp's `vendor/hash/sha256`, unchanged).
A slot's cache is reused up to the first image that is not the same; an
image the cache lacks is encoded when its request takes the slot and
decoded on its own when the slot reaches it, before the batch of the step,
as `process_mtmd_chunk`. Context shift is turned off, as llama-server turns
it off. `scripts/compare-server.py` with an image (the third argument, and
`LLAMA_MEDIA_MARKER=<__media__>` for both servers) adds 11 requests: chat
with one image, the same image again from the cache, streamed, two images
in one message, text after them, `multimodal_data`, `/apply-template`, and
the errors of a bad image, a `file://` URL without `--media-path` and more
markers than images. With SmolVLM 256M and gemma-3 4B, on the jpeg of llama.cpp's tests: 38 of the 40 requests the same; the other two are `/models`, where llama-server, built with a video decoder, lists `video` among the input modalities, and this port reads no video.

Two bugs this found. The vision encoder built its graph in the memory of the
text model's, which a context keeps from one decode to the next: the first
graph the text model reused after an image ran on overwritten tensors (the
encoder now has memory of its own). And a step that decodes an image keeps
no checkpoint, but cuts its batch where a checkpoint would go as any other
step does; the port cut it only in the steps with no image, and the
different ubatches changed the last bits of the logits.

## The console and tracing

Goose programs get their arguments in the Windows code page, where
llama.cpp's get UTF-8 from their manifest; `console::utf8_args` reads the
command line as UTF-8 (a Japanese prompt failed before it).

`LLAMA_GOOSE_TRACE=1` prints sums of layer 0's tensors to compare with
llama.cpp's eval callback.
