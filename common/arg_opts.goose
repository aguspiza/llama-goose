// The options of llama.cpp's common_params_parser_init: names, value hints,
// environment variables, the examples that show them, and the help text with
// its defaults. Generated once from common/arg.cpp, then kept by hand.
// port-of: common/arg.cpp (common_params_parser_init)

import std;
import llama from "../llama";
import llama.model;
import .fit;
import .arg;

namespace arg;

enum Opt {
    Help, Version, CacheList, CompletionBash, ServerBase, VerbosePrompt, DisplayPrompt, Color,
    Threads, ThreadsBatch, CpuMask, CpuRange, CpuStrict, Prio, Poll, CpuMaskBatch, CpuRangeBatch,
    CpuStrictBatch, PrioBatch, PollBatch, LookupCacheStatic, LookupCacheDynamic, CtxSize,
    KvUnifiedPerSlot, Predict, BatchSize, UbatchSize, Keep, SwaFull, CtxCheckpoints,
    CheckpointMinStep, CacheRam, KvUnified, CacheIdleSlots, ContextShift, Chunks, FlashAttn, Prompt,
    SystemPrompt, Perf, ShowTimings, File, SystemPromptFile, BinaryFile, Escape, PrintTokenCount,
    PromptCache, PromptCacheAll, PromptCacheRo, ReversePrompt, Special, Conversation, SingleTurn,
    Interactive, InteractiveFirst, MultilineInput, InPrefixBos, InPrefix, InSuffix, Warmup,
    SpmInfill, Samplers, Seed, SamplerSeq, IgnoreEos, Temp, TopK, TopP, MinP, TopNsigma,
    XtcProbability, XtcThreshold, Typical, RepeatLastN, RepeatPenalty, PresencePenalty,
    FrequencyPenalty, DryMultiplier, DryBase, DryAllowedLength, DryPenaltyLastN, DrySequenceBreaker,
    AdaptiveTarget, AdaptiveDecay, DynatempRange, DynatempExp, Mirostat, MirostatLr, MirostatEnt,
    LogitBias, Grammar, GrammarFile, JsonSchema, JsonSchemaFile, BackendSampling, Pooling,
    RopeScaling, RopeScale, RopeFreqBase, RopeFreqScale, YarnOrigCtx, YarnExtFactor, YarnAttnFactor,
    YarnBetaSlow, YarnBetaFast, GrpAttnN, GrpAttnW, KvOffload, Repack, NoHost, CacheTypeK,
    CacheTypeV, Hellaswag, HellaswagTasks, Winogrande, WinograndeTasks, MultipleChoice,
    MultipleChoiceTasks, KlDivergence, SaveAllLogits, PplStride, PplOutputType, Parallel, Parallel2,
    Sequences, ContBatching, Mmproj, MmprojUrl, MmprojAuto, MmprojOffload, MmprojDevice, Image,
    ImageMinTokens, ImageMaxTokens, MtmdBatchMaxTokens, VideoFps, VideoTimestampInterval,
    VideoFfmpegDir, Rpc, LoadMode, LazyMode, Numa, Device, ListDevices, OverrideTensor, CpuMoe,
    NCpuMoe, MoeCacheMib, MoeCacheCpuShare, NCpuFfn, GpuLayers, SplitMode, TensorSplit, MainGpu,
    Fit, FitTarget, FitCtx, CheckTensors, OverrideKv, OpOffload, Lora, LoraScaled, ControlVector,
    ControlVectorScaled, ControlVectorLayerRange, Alias, Tags, Model, Ple, ModelUrl, DockerRepo,
    HfRepo, HfFile, HfToken, Junk, Output, Pps, Tgs, EmbdNormalize, Host, Port, ReusePort, Path,
    CorsOrigins, CorsMethods, CorsHeaders, CorsCredentials, ApiPrefix, UiConfig, UiConfigFile,
    UiMcpProxy, UiMcpProxyAllowPrivateUrls, Tools, ToolsRuntime, AllowUnauthenticatedTools,
    McpServersConfig, McpServersJson, Agent, Ui, Connect, ConnectCode, Embedding, Rerank, ApiKey,
    ApiKeyFile, SslKeyFile, SslCertFile, ChatTemplateKwargs, Timeout, SsePingInterval, ThreadsHttp,
    CachePrompt, CacheReuse, Metrics, Props, Slots, SlotSavePath, MediaPath, MediaAllowPrivateUrls,
    ModelsDir, ModelsPreset, ModelsMax, ModelsMemoryMargin, ModelsAutoload, Jinja, ReasoningFormat,
    Reasoning, ReasoningEffort, ReasoningBudget, ReasoningBudgetMessage, ReasoningPreserve,
    ChatTemplate, ChatTemplateFile, SkipChatParsing, PrefillAssistant, SlotPromptSimilarity,
    LoraInitWithoutApply, SleepIdleSeconds, SimpleIo, LogDisable, LogFile, LogJsonl, LogPromptsDir,
    LogColors, Verbose, Offline, Verbosity, LogPrefix, LogTimestamps, SpecDraftHf, SpecDraftThreads,
    SpecDraftThreadsBatch, SpecDraftCpuMask, SpecDraftCpuRange, SpecDraftCpuStrict, SpecDraftPrio,
    SpecDraftPoll, SpecDraftCpuMaskBatch, SpecDraftCpuStrictBatch, SpecDraftPrioBatch,
    SpecDraftPollBatch, SpecDraftTypeK, SpecDraftTypeV, SpecDraftOverrideTensor, SpecDraftCpuMoe,
    SpecDraftNCpuMoe, SpecDraftNMax, SpecDraftNMin, SpecDraftNUbatch, SpecSynthLen, SpecSynthRates,
    SpecDraftPSplit, SpecDraftPMin, SpecDraftBackendSampling, SpecDraftSampling, SpecDraftDevice,
    SpecDraftNgl, SpecDraftModel, SpecType, SpecNgramModNMin, SpecNgramModNMax, SpecNgramModNMatch,
    SpecNgramSimpleSizeN, SpecNgramSimpleSizeM, SpecNgramSimpleMinHits, SpecNgramMapKSizeN,
    SpecNgramMapKSizeM, SpecNgramMapKMinHits, SpecNgramMapK4vSizeN, SpecNgramMapK4vSizeM,
    SpecNgramMapK4vMinHits, Draft, DraftMin, SpecNgramSizeN, SpecNgramSizeM, SpecNgramMinHits,
    TtsLang, TtsVoice, EmbdGemmaDefault, FimQwen15bDefault, FimQwen3bDefault, FimQwen7bDefault,
    FimQwen7bSpec, FimQwen14bSpec, FimQwen30bDefault, GptOss20bDefault, GptOss120bDefault,
    VisionGemma4bDefault, VisionGemma12bDefault, SpecDefault,
}

let EX_BATCHED: u64 = 1;
let EX_DEBUG: u64 = 2;
let EX_COMMON: u64 = 4;
let EX_SPECULATIVE: u64 = 8;
let EX_COMPLETION: u64 = 16;
let EX_CLI: u64 = 32;
let EX_EMBEDDING: u64 = 64;
let EX_PERPLEXITY: u64 = 128;
let EX_RETRIEVAL: u64 = 256;
let EX_PASSKEY: u64 = 512;
let EX_IMATRIX: u64 = 1024;
let EX_BENCH: u64 = 2048;
let EX_SERVER: u64 = 4096;
let EX_CVECTOR_GENERATOR: u64 = 8192;
let EX_EXPORT_LORA: u64 = 16384;
let EX_MTMD: u64 = 32768;
let EX_LOOKUP: u64 = 65536;
let EX_PARALLEL: u64 = 131072;
let EX_TTS: u64 = 262144;
let EX_DIFFUSION: u64 = 524288;
let EX_FINETUNE: u64 = 1048576;
let EX_FIT_PARAMS: u64 = 2097152;
let EX_RESULTS: u64 = 4194304;
let EX_EXPORT_GRAPH_OPS: u64 = 8388608;
let EX_DOWNLOAD: u64 = 16777216;
let EX_TOKENIZE: u64 = 33554432;

fn add_options(ex: Example) {
    if add(Opt.Help, Kind.Void, "-h --help --usage", "", "", "", "", EX_COMMON | EX_DOWNLOAD, 0, 0) {
        h("print usage and exit");
    }
    if add(Opt.Version, Kind.Void, "--version", "", "", "", "", EX_COMMON, 0, 0) {
        h("show version and build info");
    }
    if add(Opt.CacheList, Kind.Void, "-cl --cache-list", "", "", "", "", EX_COMMON, 0, 0) {
        h("show list of models in cache");
    }
    if add(Opt.CompletionBash, Kind.Void, "--completion-bash", "", "", "", "", EX_COMMON, 0, 0) {
        h("print source-able bash completion script for llama.cpp");
    }
    if add(Opt.ServerBase, Kind.Str, "--server-base", "", "URL", "", "", EX_CLI, 0, 0) {
        h("connect to this server instead of starting a new one, example: 'http://localhost:9931' (default: none)");
    }
    if add(Opt.VerbosePrompt, Kind.Void, "--verbose-prompt", "", "", "", "", EX_COMPLETION | EX_CLI | EX_EMBEDDING | EX_RETRIEVAL, 0, 0) {
        h("print a verbose prompt before generation (default: ");
        h(if params.verbose_prompt { "true" } else { "false" });
        h(")");
    }
    if add(Opt.DisplayPrompt, Kind.Bool, "--display-prompt", "--no-display-prompt", "", "", "", EX_COMPLETION | EX_CLI, 0, 0) {
        h("whether to print prompt at generation (default: ");
        h(if params.display_prompt { "true" } else { "false" });
        h(")");
    }
    if add(Opt.Color, Kind.Str, "-co --color", "", "[on|off|auto]", "", "", EX_COMPLETION | EX_CLI | EX_SPECULATIVE | EX_LOOKUP, 0, 0) {
        h("Colorize output to distinguish prompt and user input from generations ('on', 'off', or 'auto', default: 'auto')\n'auto' enables colors when output is to a terminal");
    }
    if add(Opt.Threads, Kind.Int, "-t --threads", "", "N", "", "LLAMA_ARG_THREADS", EX_COMMON, 0, 0) {
        h("number of CPU threads to use during generation (default: ");
        hi(params.n_threads);
        h(")");
    }
    if add(Opt.ThreadsBatch, Kind.Int, "-tb --threads-batch", "", "N", "", "", EX_COMMON, 0, 0) {
        h("number of threads to use during batch and prompt processing (default: same as --threads)");
    }
    if add(Opt.CpuMask, Kind.Str, "-C --cpu-mask", "", "M", "", "", EX_COMMON, 0, 0) {
        h("CPU affinity mask: arbitrarily long hex. Complements cpu-range (default: \"\")");
    }
    if add(Opt.CpuRange, Kind.Str, "-Cr --cpu-range", "", "lo-hi", "", "", EX_COMMON, 0, 0) {
        h("range of CPUs for affinity. Complements --cpu-mask");
    }
    if add(Opt.CpuStrict, Kind.Str, "--cpu-strict", "", "<0|1>", "", "", EX_COMMON, 0, 0) {
        h("use strict CPU placement (default: ");
        hi(0);
        h(")\n");
    }
    if add(Opt.Prio, Kind.Int, "--prio", "", "N", "", "", EX_COMMON, 0, 0) {
        h("set process/thread priority : low(-1), normal(0), medium(1), high(2), realtime(3) (default: ");
        hi(0);
        h(")\n");
    }
    if add(Opt.Poll, Kind.Str, "--poll", "", "<0...100>", "", "", EX_COMMON, 0, 0) {
        h("use polling level to wait for work (0 - no polling, default: ");
        hi(50);
        h(")\n");
    }
    if add(Opt.CpuMaskBatch, Kind.Str, "-Cb --cpu-mask-batch", "", "M", "", "", EX_COMMON, 0, 0) {
        h("CPU affinity mask: arbitrarily long hex. Complements cpu-range-batch (default: same as --cpu-mask)");
    }
    if add(Opt.CpuRangeBatch, Kind.Str, "-Crb --cpu-range-batch", "", "lo-hi", "", "", EX_COMMON, 0, 0) {
        h("ranges of CPUs for affinity. Complements --cpu-mask-batch");
    }
    if add(Opt.CpuStrictBatch, Kind.Int, "--cpu-strict-batch", "", "<0|1>", "", "", EX_COMMON, 0, 0) {
        h("use strict CPU placement (default: same as --cpu-strict)");
    }
    if add(Opt.PrioBatch, Kind.Int, "--prio-batch", "", "N", "", "", EX_COMMON, 0, 0) {
        h("set process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime (default: ");
        hi(0);
        h(")\n");
    }
    if add(Opt.PollBatch, Kind.Int, "--poll-batch", "", "<0|1>", "", "", EX_COMMON, 0, 0) {
        h("use polling to wait for work (default: same as --poll)");
    }
    if add(Opt.LookupCacheStatic, Kind.Str, "-lcs --lookup-cache-static", "", "FNAME", "", "", EX_LOOKUP | EX_SERVER, 0, 0) {
        h("path to static lookup cache to use for lookup decoding (not updated by generation)");
    }
    if add(Opt.LookupCacheDynamic, Kind.Str, "-lcd --lookup-cache-dynamic", "", "FNAME", "", "", EX_LOOKUP | EX_SERVER, 0, 0) {
        h("path to dynamic lookup cache to use for lookup decoding (updated by generation)");
    }
    if add(Opt.CtxSize, Kind.Int, "-c --ctx-size", "", "N", "", "LLAMA_ARG_CTX_SIZE", EX_COMMON, 0, 0) {
        h("size of the prompt context (default: ");
        hi(params.n_ctx);
        h(", 0 = loaded from model)");
    }
    if add(Opt.KvUnifiedPerSlot, Kind.Int, "--kv-unified-per-slot", "", "N", "", "LLAMA_ARG_KV_UNIFIED_PER_SLOT", EX_SERVER, 0, 0) {
        h("context limit per parallel slot (default: unset, behavior unchanged).\nwhen set without -c/--ctx-size, the shared KV pool is sized to n_parallel*N");
    }
    if add(Opt.Predict, Kind.Int, "-n --predict --n-predict", "", "N", "", "LLAMA_ARG_N_PREDICT", EX_COMMON, 0, 0) {
        h("number of tokens to predict (default: ");
        hi(params.n_predict);
        h(if ex == Example.Completion { ", -1 = infinity, -2 = until context filled)" } else { ", -1 = infinity)" });
    }
    if add(Opt.BatchSize, Kind.Int, "-b --batch-size", "", "N", "", "LLAMA_ARG_BATCH", EX_COMMON, 0, 0) {
        h("logical maximum batch size (default: ");
        hi(params.n_batch);
        h(")");
    }
    if add(Opt.UbatchSize, Kind.Int, "-ub --ubatch-size", "", "N", "", "LLAMA_ARG_UBATCH", EX_COMMON, 0, 0) {
        h("physical maximum batch size (default: ");
        hi(params.n_ubatch);
        h(")");
    }
    if add(Opt.Keep, Kind.Int, "--keep", "", "N", "", "", EX_COMMON, 0, 0) {
        h("number of tokens to keep from the initial prompt (default: ");
        hi(params.n_keep);
        h(", -1 = all)");
    }
    if add(Opt.SwaFull, Kind.Void, "--swa-full", "", "", "", "LLAMA_ARG_SWA_FULL", EX_COMMON, 0, 0) {
        h("use full-size SWA cache (default: ");
        h(if false { "true" } else { "false" });
        h(")\n[(more info)](https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)");
    }
    if add(Opt.CtxCheckpoints, Kind.Int, "-ctxcp --ctx-checkpoints --swa-checkpoints", "", "N", "", "LLAMA_ARG_CTX_CHECKPOINTS", EX_SERVER | EX_CLI, 0, 0) {
        h("max number of context checkpoints to create per slot (default: ");
        hi(params.n_ctx_checkpoints);
        h(")[(more info)](https://github.com/ggml-org/llama.cpp/pull/15293)");
    }
    if add(Opt.CheckpointMinStep, Kind.Int, "-cms --checkpoint-min-step", "", "N", "", "LLAMA_ARG_CHECKPOINT_MIN_SPACING_NT", EX_SERVER, 0, 0) {
        h("minimum spacing between context checkpoints in tokens (default: ");
        hi(8192);
        h(", 0 = no minimum)");
    }
    if add(Opt.CacheRam, Kind.Int, "-cram --cache-ram", "", "N", "", "LLAMA_ARG_CACHE_RAM", EX_SERVER | EX_CLI, 0, 0) {
        h("set the maximum cache size in MiB (default: ");
        hi(8192);
        h(", 0 for decision models, -1 - no limit, 0 - disable)[(more info)](https://github.com/ggml-org/llama.cpp/pull/16391)");
    }
    if add(Opt.KvUnified, Kind.Bool, "-kvu --kv-unified", "-no-kvu --no-kv-unified", "", "", "LLAMA_ARG_KV_UNIFIED", EX_SERVER | EX_PERPLEXITY | EX_BATCHED | EX_BENCH | EX_PARALLEL | EX_SPECULATIVE, 0, 0) {
        h("use single unified KV buffer shared across all sequences (default: enabled if number of slots is auto)");
    }
    if add(Opt.CacheIdleSlots, Kind.Bool, "--cache-idle-slots", "--no-cache-idle-slots", "", "", "LLAMA_ARG_CACHE_IDLE_SLOTS", EX_SERVER, 0, 0) {
        h("save idle slots to the prompt cache on new task, and clear them when using unified KV (default: enabled, requires cache-ram)");
    }
    if add(Opt.ContextShift, Kind.Bool, "--context-shift", "--no-context-shift", "", "", "LLAMA_ARG_CONTEXT_SHIFT", EX_COMPLETION | EX_CLI | EX_SERVER | EX_IMATRIX | EX_PERPLEXITY, 0, 0) {
        h("whether to use context shift on infinite text generation (default: ");
        h(if params.ctx_shift { "enabled" } else { "disabled" });
        h(")");
    }
    if add(Opt.Chunks, Kind.Int, "--chunks", "", "N", "", "", EX_IMATRIX | EX_PERPLEXITY | EX_RETRIEVAL, 0, 0) {
        h("max number of chunks to process (default: ");
        hi(params.n_chunks);
        h(", -1 = all)");
    }
    if add(Opt.FlashAttn, Kind.Str, "-fa --flash-attn", "", "[on|off|auto]", "", "LLAMA_ARG_FLASH_ATTN", EX_COMMON, 0, 0) {
        h("set Flash Attention use ('on', 'off', or 'auto', default: '");
        h("auto");
        h("')");
    }
    if add(Opt.Prompt, Kind.Str, "-p --prompt", "", "PROMPT", "", "", EX_COMMON, EX_SERVER, 0) {
        h("prompt to start generation with; for system message, use -sys");
    }
    if add(Opt.SystemPrompt, Kind.Str, "-sys --system-prompt", "", "PROMPT", "", "", EX_COMPLETION | EX_CLI | EX_DIFFUSION | EX_MTMD, 0, 0) {
        h("system prompt to use with model (if applicable, depending on chat template)");
    }
    if add(Opt.Perf, Kind.Bool, "--perf", "--no-perf", "", "", "LLAMA_ARG_PERF", EX_COMMON, 0, 0) {
        h("whether to enable internal libllama performance timings (default: ");
        h(if false { "true" } else { "false" });
        h(")");
    }
    if add(Opt.ShowTimings, Kind.Bool, "--show-timings", "--no-show-timings", "", "", "LLAMA_ARG_SHOW_TIMINGS", EX_CLI, 0, 0) {
        h("whether to show timing information after each response (default: ");
        h(if params.show_timings { "true" } else { "false" });
        h(")");
    }
    if add(Opt.File, Kind.Str, "-f --file", "", "FNAME", "", "", EX_COMMON, EX_SERVER, 0) {
        h("a file containing the prompt (default: none)");
    }
    if add(Opt.SystemPromptFile, Kind.Str, "-sysf --system-prompt-file", "", "FNAME", "", "", EX_COMPLETION | EX_CLI | EX_DIFFUSION, 0, 0) {
        h("a file containing the system prompt (default: none)");
    }
    if add(Opt.BinaryFile, Kind.Str, "-bf --binary-file", "", "FNAME", "", "", EX_COMMON, EX_SERVER, 0) {
        h("binary file containing the prompt (default: none)");
    }
    if add(Opt.Escape, Kind.Bool, "-e --escape", "--no-escape", "", "", "", EX_COMMON, 0, 0) {
        h("whether to process escapes sequences (\\n, \\r, \\t, \\', \\\", \\\\) (default: ");
        h(if params.escape { "true" } else { "false" });
        h(")");
    }
    if add(Opt.PrintTokenCount, Kind.Int, "-ptc --print-token-count", "", "N", "", "", EX_COMPLETION, 0, 0) {
        h("print token count every N tokens (default: ");
        hi(-1);
        h(")");
    }
    if add(Opt.PromptCache, Kind.Str, "--prompt-cache", "", "FNAME", "", "", EX_COMPLETION, 0, 0) {
        h("file to cache prompt state for faster startup (default: none)");
    }
    if add(Opt.PromptCacheAll, Kind.Void, "--prompt-cache-all", "", "", "", "", EX_COMPLETION, 0, 0) {
        h("if specified, saves user input and generations to cache as well\n");
    }
    if add(Opt.PromptCacheRo, Kind.Void, "--prompt-cache-ro", "", "", "", "", EX_COMPLETION, 0, 0) {
        h("if specified, uses the prompt cache but does not update it");
    }
    if add(Opt.ReversePrompt, Kind.Str, "-r --reverse-prompt", "", "PROMPT", "", "", EX_COMPLETION | EX_CLI | EX_SERVER, 0, 0) {
        h("halt generation at PROMPT, return control in interactive mode\n");
    }
    if add(Opt.Special, Kind.Void, "-sp --special", "", "", "", "", EX_COMPLETION | EX_CLI | EX_SERVER, 0, 0) {
        h("special tokens output enabled (default: ");
        h(if params.special { "true" } else { "false" });
        h(")");
    }
    if add(Opt.Conversation, Kind.Bool, "-cnv --conversation", "-no-cnv --no-conversation", "", "", "", EX_COMPLETION, 0, 0) {
        h("whether to run in conversation mode:\n- does not print special tokens and suffix/prefix\n- interactive mode is also enabled\n(default: auto enabled if chat template is available)");
    }
    if add(Opt.SingleTurn, Kind.Void, "-st --single-turn", "", "", "", "", EX_COMPLETION | EX_CLI, 0, 0) {
        h("run conversation for a single turn only, then exit when done\nwill not be interactive if first turn is predefined with --prompt\n(default: false)");
    }
    if add(Opt.Interactive, Kind.Void, "-i --interactive", "", "", "", "", EX_COMPLETION, 0, 0) {
        h("run in interactive mode (default: ");
        h(if params.interactive { "true" } else { "false" });
        h(")");
    }
    if add(Opt.InteractiveFirst, Kind.Void, "-if --interactive-first", "", "", "", "", EX_COMPLETION, 0, 0) {
        h("run in interactive mode and wait for input right away (default: ");
        h(if params.interactive_first { "true" } else { "false" });
        h(")");
    }
    if add(Opt.MultilineInput, Kind.Void, "-mli --multiline-input", "", "", "", "", EX_COMPLETION | EX_CLI, 0, 0) {
        h("allows you to write or paste multiple lines without ending each in '\\'");
    }
    if add(Opt.InPrefixBos, Kind.Void, "--in-prefix-bos", "", "", "", "", EX_COMPLETION, 0, 0) {
        h("prefix BOS to user inputs, preceding the `--in-prefix` string");
    }
    if add(Opt.InPrefix, Kind.Str, "--in-prefix", "", "STRING", "", "", EX_COMPLETION, 0, 0) {
        h("string to prefix user inputs with (default: empty)");
    }
    if add(Opt.InSuffix, Kind.Str, "--in-suffix", "", "STRING", "", "", EX_COMPLETION, 0, 0) {
        h("string to suffix after user inputs with (default: empty)");
    }
    if add(Opt.Warmup, Kind.Bool, "--warmup", "--no-warmup", "", "", "", EX_COMPLETION | EX_CLI | EX_SERVER | EX_MTMD | EX_EMBEDDING | EX_RETRIEVAL | EX_PERPLEXITY | EX_DEBUG, 0, 0) {
        h("whether to perform warmup with an empty run (default: ");
        h(if true { "enabled" } else { "disabled" });
        h(")");
    }
    if add(Opt.SpmInfill, Kind.Void, "--spm-infill", "", "", "", "", EX_SERVER, 0, 0) {
        h("use Suffix/Prefix/Middle pattern for infill (instead of Prefix/Suffix/Middle) as some models prefer this. (default: ");
        h(if false { "enabled" } else { "disabled" });
        h(")");
    }
    if add(Opt.Samplers, Kind.Str, "--samplers", "", "SAMPLERS", "", "", EX_COMMON, 0, SAMPLING) {
        h("samplers that will be used for generation in the order, separated by ';'\n(default: ");
        h(sampler_names());
        h(")");
    }
    if add(Opt.Seed, Kind.Str, "-s --seed", "", "SEED", "", "", EX_COMMON, 0, SAMPLING) {
        h("RNG seed (default: ");
        hi((params.sampling.seed as i64 << 32) >> 32);
        h(", use random seed for ");
        hi(-1);
        h(")");
    }
    if add(Opt.SamplerSeq, Kind.Str, "--sampler-seq --sampling-seq", "", "SEQUENCE", "", "", EX_COMMON, 0, SAMPLING) {
        h("simplified sequence for samplers that will be used (default: ");
        h(sampler_chars());
        h(")");
    }
    if add(Opt.IgnoreEos, Kind.Void, "--ignore-eos", "", "", "", "", EX_COMMON, 0, SAMPLING) {
        h("ignore end of stream token and continue generating (implies --logit-bias EOS-inf)");
    }
    if add(Opt.Temp, Kind.Str, "--temp --temperature", "", "N", "", "LLAMA_ARG_TEMPERATURE", EX_COMMON, 0, SAMPLING) {
        h("temperature (default: ");
        hf(params.sampling.temp, 2);
        h(")");
    }
    if add(Opt.TopK, Kind.Int, "--top-k", "", "N", "", "LLAMA_ARG_TOP_K", EX_COMMON, 0, SAMPLING) {
        h("top-k sampling (default: ");
        hi(params.sampling.top_k);
        h(", 0 = disabled)");
    }
    if add(Opt.TopP, Kind.Str, "--top-p", "", "N", "", "LLAMA_ARG_TOP_P", EX_COMMON, 0, SAMPLING) {
        h("top-p sampling (default: ");
        hf(params.sampling.top_p, 2);
        h(", 1.0 = disabled)");
    }
    if add(Opt.MinP, Kind.Str, "--min-p", "", "N", "", "LLAMA_ARG_MIN_P", EX_COMMON, 0, SAMPLING) {
        h("min-p sampling (default: ");
        hf(params.sampling.min_p, 2);
        h(", 0.0 = disabled)");
    }
    if add(Opt.TopNsigma, Kind.Str, "--top-nsigma --top-n-sigma", "", "N", "", "", EX_COMMON, 0, SAMPLING) {
        h("top-n-sigma sampling (default: ");
        hf(params.sampling.top_n_sigma, 2);
        h(", -1.0 = disabled)");
    }
    if add(Opt.XtcProbability, Kind.Str, "--xtc-probability", "", "N", "", "", EX_COMMON, 0, SAMPLING) {
        h("xtc probability (default: ");
        hf(params.sampling.xtc_probability, 2);
        h(", 0.0 = disabled)");
    }
    if add(Opt.XtcThreshold, Kind.Str, "--xtc-threshold", "", "N", "", "", EX_COMMON, 0, SAMPLING) {
        h("xtc threshold (default: ");
        hf(params.sampling.xtc_threshold, 2);
        h(", 1.0 = disabled)");
    }
    if add(Opt.Typical, Kind.Str, "--typical --typical-p", "", "N", "", "", EX_COMMON, 0, SAMPLING) {
        h("locally typical sampling, parameter p (default: ");
        hf(params.sampling.typ_p, 2);
        h(", 1.0 = disabled)");
    }
    if add(Opt.RepeatLastN, Kind.Int, "--repeat-last-n", "", "N", "", "", EX_COMMON, 0, SAMPLING) {
        h("last n tokens to consider for penalize (default: ");
        hi(params.sampling.penalty_last_n);
        h(", 0 = disabled)");
    }
    if add(Opt.RepeatPenalty, Kind.Str, "--repeat-penalty", "", "N", "", "LLAMA_ARG_REPEAT_PENALTY", EX_COMMON, 0, SAMPLING) {
        h("penalize repeat sequence of tokens (default: ");
        hf(params.sampling.penalty_repeat, 2);
        h(", 1.0 = disabled)");
    }
    if add(Opt.PresencePenalty, Kind.Str, "--presence-penalty", "", "N", "", "LLAMA_ARG_PRESENCE_PENALTY", EX_COMMON, 0, SAMPLING) {
        h("repeat alpha presence penalty (default: ");
        hf(params.sampling.penalty_present, 2);
        h(", 0.0 = disabled)");
    }
    if add(Opt.FrequencyPenalty, Kind.Str, "--frequency-penalty", "", "N", "", "LLAMA_ARG_FREQUENCY_PENALTY", EX_COMMON, 0, SAMPLING) {
        h("repeat alpha frequency penalty (default: ");
        hf(params.sampling.penalty_freq, 2);
        h(", 0.0 = disabled)");
    }
    if add(Opt.DryMultiplier, Kind.Str, "--dry-multiplier", "", "N", "", "", EX_COMMON, 0, SAMPLING) {
        h("set DRY sampling multiplier (default: ");
        hf(params.sampling.dry_multiplier, 2);
        h(", 0.0 = disabled)");
    }
    if add(Opt.DryBase, Kind.Str, "--dry-base", "", "N", "", "", EX_COMMON, 0, SAMPLING) {
        h("set DRY sampling base value (default: ");
        hf(params.sampling.dry_base, 2);
        h(")");
    }
    if add(Opt.DryAllowedLength, Kind.Int, "--dry-allowed-length", "", "N", "", "", EX_COMMON, 0, SAMPLING) {
        h("set allowed length for DRY sampling (default: ");
        hi(params.sampling.dry_allowed_length);
        h(")");
    }
    if add(Opt.DryPenaltyLastN, Kind.Int, "--dry-penalty-last-n", "", "N", "", "", EX_COMMON, 0, SAMPLING) {
        h("set DRY penalty for the last n tokens (default: ");
        hi(params.sampling.dry_penalty_last_n);
        h(", 0 = disable)");
    }
    if add(Opt.DrySequenceBreaker, Kind.Str, "--dry-sequence-breaker", "", "STRING", "", "", EX_COMMON, 0, SAMPLING) {
        h("add sequence breaker for DRY sampling, clearing out default breakers (");
        h(dry_breakers());
        h(") in the process; use \"none\" to not use any sequence breakers\n");
    }
    if add(Opt.AdaptiveTarget, Kind.Str, "--adaptive-target", "", "N", "", "", EX_COMMON, 0, SAMPLING) {
        h("adaptive-p: select tokens near this probability (valid range 0.0 to 1.0; negative = disabled) (default: ");
        hf(-1.0, 2);
        h(")\n[(more info)](https://github.com/ggml-org/llama.cpp/pull/17927)");
    }
    if add(Opt.AdaptiveDecay, Kind.Str, "--adaptive-decay", "", "N", "", "", EX_COMMON, 0, SAMPLING) {
        h("adaptive-p: decay rate for target adaptation over time. lower values are more reactive, higher values are more stable.\n(valid range 0.0 to 0.99) (default: ");
        hf(0.9, 2);
        h(")");
    }
    if add(Opt.DynatempRange, Kind.Str, "--dynatemp-range", "", "N", "", "", EX_COMMON, 0, SAMPLING) {
        h("dynamic temperature range (default: ");
        hf(params.sampling.dynatemp_range, 2);
        h(", 0.0 = disabled)");
    }
    if add(Opt.DynatempExp, Kind.Str, "--dynatemp-exp", "", "N", "", "", EX_COMMON, 0, SAMPLING) {
        h("dynamic temperature exponent (default: ");
        hf(params.sampling.dynatemp_exponent, 2);
        h(")");
    }
    if add(Opt.Mirostat, Kind.Int, "--mirostat", "", "N", "", "", EX_COMMON, 0, SAMPLING) {
        h("use Mirostat sampling.\nTop K, Nucleus and Locally Typical samplers are ignored if used.\n(default: ");
        hi(params.sampling.mirostat);
        h(", 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)");
    }
    if add(Opt.MirostatLr, Kind.Str, "--mirostat-lr", "", "N", "", "", EX_COMMON, 0, SAMPLING) {
        h("Mirostat learning rate, parameter eta (default: ");
        hf(params.sampling.mirostat_eta, 2);
        h(")");
    }
    if add(Opt.MirostatEnt, Kind.Str, "--mirostat-ent", "", "N", "", "", EX_COMMON, 0, SAMPLING) {
        h("Mirostat target entropy, parameter tau (default: ");
        hf(params.sampling.mirostat_tau, 2);
        h(")");
    }
    if add(Opt.LogitBias, Kind.Str, "-l --logit-bias", "", "TOKEN_ID(+/-)BIAS", "", "", EX_COMMON, 0, SAMPLING) {
        h("modifies the likelihood of token appearing in the completion,\ni.e. `--logit-bias 15043+1` to increase likelihood of token ' Hello',\nor `--logit-bias 15043-1` to decrease likelihood of token ' Hello'");
    }
    if add(Opt.Grammar, Kind.Str, "--grammar", "", "GRAMMAR", "", "", EX_COMMON, 0, SAMPLING) {
        h("BNF-like grammar to constrain generations (see samples in grammars/ dir)");
    }
    if add(Opt.GrammarFile, Kind.Str, "--grammar-file", "", "FNAME", "", "", EX_COMMON, 0, SAMPLING) {
        h("file to read grammar from");
    }
    if add(Opt.JsonSchema, Kind.Str, "-j --json-schema", "", "SCHEMA", "", "", EX_COMMON, 0, SAMPLING) {
        h("JSON schema to constrain generations (https://json-schema.org/), e.g. `{\"type\": \"object\"}` for any JSON object");
    }
    if add(Opt.JsonSchemaFile, Kind.Str, "-jf --json-schema-file", "", "FILE", "", "", EX_COMMON, 0, SAMPLING) {
        h("File containing a JSON schema to constrain generations (https://json-schema.org/), e.g. `{\"type\": \"object\"}` for any JSON object");
    }
    if add(Opt.BackendSampling, Kind.Void, "-bs --backend-sampling", "", "", "", "LLAMA_ARG_BACKEND_SAMPLING", EX_COMMON, 0, SAMPLING) {
        h("enable backend sampling (experimental) (default: disabled)");
    }
    if add(Opt.Pooling, Kind.Str, "--pooling", "", "{none,mean,cls,last,rank}", "", "LLAMA_ARG_POOLING", EX_EMBEDDING | EX_RETRIEVAL | EX_SERVER | EX_DEBUG, 0, 0) {
        h("pooling type for embeddings, use model default if unspecified");
    }
    if add(Opt.RopeScaling, Kind.Str, "--rope-scaling", "", "{none,linear,yarn}", "", "LLAMA_ARG_ROPE_SCALING_TYPE", EX_COMMON, 0, 0) {
        h("RoPE frequency scaling method, defaults to linear unless specified by the model");
    }
    if add(Opt.RopeScale, Kind.Str, "--rope-scale", "", "N", "", "LLAMA_ARG_ROPE_SCALE", EX_COMMON, 0, 0) {
        h("RoPE context scaling factor, expands context by a factor of N");
    }
    if add(Opt.RopeFreqBase, Kind.Str, "--rope-freq-base", "", "N", "", "LLAMA_ARG_ROPE_FREQ_BASE", EX_COMMON, 0, 0) {
        h("RoPE base frequency, used by NTK-aware scaling (default: loaded from model)");
    }
    if add(Opt.RopeFreqScale, Kind.Str, "--rope-freq-scale", "", "N", "", "LLAMA_ARG_ROPE_FREQ_SCALE", EX_COMMON, 0, 0) {
        h("RoPE frequency scaling factor, expands context by a factor of 1/N");
    }
    if add(Opt.YarnOrigCtx, Kind.Int, "--yarn-orig-ctx", "", "N", "", "LLAMA_ARG_YARN_ORIG_CTX", EX_COMMON, 0, 0) {
        h("YaRN: original context size of model (default: ");
        hi(0);
        h(" = model training context size)");
    }
    if add(Opt.YarnExtFactor, Kind.Str, "--yarn-ext-factor", "", "N", "", "LLAMA_ARG_YARN_EXT_FACTOR", EX_COMMON, 0, 0) {
        h("YaRN: extrapolation mix factor (default: ");
        hf(-1.0, 2);
        h(", 0.0 = full interpolation)");
    }
    if add(Opt.YarnAttnFactor, Kind.Str, "--yarn-attn-factor", "", "N", "", "LLAMA_ARG_YARN_ATTN_FACTOR", EX_COMMON, 0, 0) {
        h("YaRN: scale sqrt(t) or attention magnitude (default: ");
        hf(-1.0, 2);
        h(")");
    }
    if add(Opt.YarnBetaSlow, Kind.Str, "--yarn-beta-slow", "", "N", "", "LLAMA_ARG_YARN_BETA_SLOW", EX_COMMON, 0, 0) {
        h("YaRN: high correction dim or alpha (default: ");
        hf(-1.0, 2);
        h(")");
    }
    if add(Opt.YarnBetaFast, Kind.Str, "--yarn-beta-fast", "", "N", "", "LLAMA_ARG_YARN_BETA_FAST", EX_COMMON, 0, 0) {
        h("YaRN: low correction dim or beta (default: ");
        hf(-1.0, 2);
        h(")");
    }
    if add(Opt.GrpAttnN, Kind.Int, "-gan --grp-attn-n", "", "N", "", "LLAMA_ARG_GRP_ATTN_N", EX_COMPLETION | EX_PASSKEY, 0, 0) {
        h("group-attention factor (default: ");
        hi(params.grp_attn_n);
        h(")");
    }
    if add(Opt.GrpAttnW, Kind.Int, "-gaw --grp-attn-w", "", "N", "", "LLAMA_ARG_GRP_ATTN_W", EX_COMPLETION, 0, 0) {
        h("group-attention width (default: ");
        hi(params.grp_attn_w);
        h(")");
    }
    if add(Opt.KvOffload, Kind.Bool, "-kvo --kv-offload", "-nkvo --no-kv-offload", "", "", "LLAMA_ARG_KV_OFFLOAD", EX_COMMON, 0, 0) {
        h("whether to enable KV cache offloading (default: ");
        h(if false { "disabled" } else { "enabled" });
        h(")");
    }
    if add(Opt.Repack, Kind.Bool, "--repack", "-nr --no-repack", "", "", "LLAMA_ARG_REPACK", EX_COMMON, 0, 0) {
        h("whether to enable weight repacking (default: ");
        h(if !model::repack { "disabled" } else { "enabled" });
        h(")");
    }
    if add(Opt.NoHost, Kind.Void, "--no-host", "", "", "", "LLAMA_ARG_NO_HOST", EX_COMMON, 0, 0) {
        h("bypass host buffer allowing extra buffers to be used");
    }
    if add(Opt.CacheTypeK, Kind.Str, "-ctk --cache-type-k", "", "TYPE", "", "LLAMA_ARG_CACHE_TYPE_K", EX_COMMON, 0, 0) {
        h("KV cache data type for K\nallowed values: ");
        h(KV_CACHE_TYPES);
        h("\n(default: ");
        h("f16");
        h(")");
    }
    if add(Opt.CacheTypeV, Kind.Str, "-ctv --cache-type-v", "", "TYPE", "", "LLAMA_ARG_CACHE_TYPE_V", EX_COMMON, 0, 0) {
        h("KV cache data type for V\nallowed values: ");
        h(KV_CACHE_TYPES);
        h("\n(default: ");
        h("f16");
        h(")");
    }
    if add(Opt.Hellaswag, Kind.Void, "--hellaswag", "", "", "", "", EX_PERPLEXITY, 0, 0) {
        h("compute HellaSwag score over random tasks from datafile supplied with -f");
    }
    if add(Opt.HellaswagTasks, Kind.Int, "--hellaswag-tasks", "", "N", "", "", EX_PERPLEXITY, 0, 0) {
        h("number of tasks to use when computing the HellaSwag score (default: ");
        hi(400);
        h(")");
    }
    if add(Opt.Winogrande, Kind.Void, "--winogrande", "", "", "", "", EX_PERPLEXITY, 0, 0) {
        h("compute Winogrande score over random tasks from datafile supplied with -f");
    }
    if add(Opt.WinograndeTasks, Kind.Int, "--winogrande-tasks", "", "N", "", "", EX_PERPLEXITY, 0, 0) {
        h("number of tasks to use when computing the Winogrande score (default: ");
        hi(0);
        h(")");
    }
    if add(Opt.MultipleChoice, Kind.Void, "--multiple-choice", "", "", "", "", EX_PERPLEXITY, 0, 0) {
        h("compute multiple choice score over random tasks from datafile supplied with -f");
    }
    if add(Opt.MultipleChoiceTasks, Kind.Int, "--multiple-choice-tasks", "", "N", "", "", EX_PERPLEXITY, 0, 0) {
        h("number of tasks to use when computing the multiple choice score (default: ");
        hi(0);
        h(")");
    }
    if add(Opt.KlDivergence, Kind.Void, "--kl-divergence", "", "", "", "", EX_PERPLEXITY, 0, 0) {
        h("computes KL-divergence to logits provided via --kl-divergence-base");
    }
    if add(Opt.SaveAllLogits, Kind.Str, "--save-all-logits --kl-divergence-base", "", "FNAME", "", "", EX_PERPLEXITY, 0, 0) {
        h("set logits file");
    }
    if add(Opt.PplStride, Kind.Int, "--ppl-stride", "", "N", "", "", EX_PERPLEXITY, 0, 0) {
        h("stride for perplexity calculation (default: ");
        hi(0);
        h(")");
    }
    if add(Opt.PplOutputType, Kind.Int, "--ppl-output-type", "", "<0|1>", "", "", EX_PERPLEXITY, 0, 0) {
        h("output type for perplexity calculation (default: ");
        hi(0);
        h(")");
    }
    if add(Opt.Parallel, Kind.Int, "-np --parallel", "", "N", "", "LLAMA_ARG_N_PARALLEL", EX_SERVER, 0, 0) {
        h("number of server slots (default: ");
        hi(params.n_parallel);
        h(", -1 = auto)");
    }
    if add(Opt.Parallel2, Kind.Int, "-np --parallel", "", "N", "", "LLAMA_ARG_N_PARALLEL", EX_COMMON, EX_SERVER, 0) {
        h("number of parallel sequences to decode (default: ");
        hi(params.n_parallel);
        h(")");
    }
    if add(Opt.Sequences, Kind.Int, "-ns --sequences", "", "N", "", "", EX_PARALLEL, 0, 0) {
        h("number of sequences to decode (default: ");
        hi(params.n_sequences);
        h(")");
    }
    if add(Opt.ContBatching, Kind.Bool, "-cb --cont-batching", "-nocb --no-cont-batching", "", "", "LLAMA_ARG_CONT_BATCHING", EX_SERVER, 0, 0) {
        h("whether to enable continuous batching (a.k.a dynamic batching) (default: ");
        h(if params.cont_batching { "enabled" } else { "disabled" });
        h(")");
    }
    if add(Opt.Mmproj, Kind.Str, "-mm --mmproj", "", "FILE", "", "LLAMA_ARG_MMPROJ", EX_MTMD | EX_SERVER | EX_CLI | EX_TTS, 0, 0) {
        h("path to a multimodal projector file. see tools/mtmd/README.md\nnote: if -hf is used, this argument can be omitted");
    }
    if add(Opt.MmprojUrl, Kind.Str, "-mmu --mmproj-url", "", "URL", "", "LLAMA_ARG_MMPROJ_URL", EX_MTMD | EX_SERVER | EX_CLI | EX_TTS, 0, 0) {
        h("URL to a multimodal projector file. see tools/mtmd/README.md");
    }
    if add(Opt.MmprojAuto, Kind.Bool, "--mmproj-auto", "--no-mmproj --no-mmproj-auto", "", "", "LLAMA_ARG_MMPROJ_AUTO", EX_MTMD | EX_SERVER | EX_CLI | EX_DOWNLOAD, 0, 0) {
        h("whether to use multimodal projector file (if available), useful when using -hf (default: ");
        h(if params.no_mmproj { "disabled" } else { "enabled" });
        h(")");
    }
    if add(Opt.MmprojOffload, Kind.Bool, "--mmproj-offload", "--no-mmproj-offload", "", "", "LLAMA_ARG_MMPROJ_OFFLOAD", EX_MTMD | EX_SERVER | EX_CLI | EX_TTS, 0, 0) {
        h("whether to enable GPU offloading for multimodal projector (default: ");
        h(if params.mmproj_use_gpu { "enabled" } else { "disabled" });
        h(")");
    }
    if add(Opt.MmprojDevice, Kind.Str, "-mmdev --mmproj-device", "", "DEVICE", "", "MTMD_BACKEND_DEVICE", EX_MTMD | EX_SERVER | EX_CLI | EX_TTS, 0, 0) {
        h("device to use for multimodal projector (none = don't offload, default: follows --device)\nuse --list-devices to see a list of available devices");
    }
    if add(Opt.Image, Kind.Str, "--image --audio --video", "", "FILE", "", "", EX_MTMD | EX_CLI, 0, 0) {
        h("path to an image, audio, or video file. use with multimodal models, use comma-separated values for multiple files\n");
    }
    if add(Opt.ImageMinTokens, Kind.Int, "--image-min-tokens", "", "N", "", "LLAMA_ARG_IMAGE_MIN_TOKENS", EX_MTMD | EX_SERVER | EX_CLI | EX_TTS, 0, 0) {
        h("minimum number of tokens each image can take, only used by vision models with dynamic resolution (default: read from model)");
    }
    if add(Opt.ImageMaxTokens, Kind.Int, "--image-max-tokens", "", "N", "", "LLAMA_ARG_IMAGE_MAX_TOKENS", EX_MTMD | EX_SERVER | EX_CLI | EX_TTS, 0, 0) {
        h("maximum number of tokens each image can take, only used by vision models with dynamic resolution (default: read from model)");
    }
    if add(Opt.MtmdBatchMaxTokens, Kind.Int, "--mtmd-batch-max-tokens", "", "N", "", "LLAMA_ARG_MTMD_BATCH_MAX_TOKENS", EX_SERVER, 0, 0) {
        h("maximum number of image tokens per batch when encoding images (default: ");
        hi(1024);
        h(")");
    }
    if add(Opt.VideoFps, Kind.Str, "--video-fps", "", "N", "", "LLAMA_ARG_VIDEO_FPS", EX_MTMD | EX_SERVER | EX_CLI | EX_TTS, 0, 0) {
        h("target video frame rate (default: ");
        hf(4.0, 1);
        h(")");
    }
    if add(Opt.VideoTimestampInterval, Kind.Int, "--video-timestamp-interval", "", "N", "", "LLAMA_ARG_VIDEO_TIMESTAMP_INTERVAL", EX_MTMD | EX_SERVER | EX_CLI | EX_TTS, 0, 0) {
        h("interval in milliseconds between text timestamps (default: ");
        hi(5000);
        h(")");
    }
    if add(Opt.VideoFfmpegDir, Kind.Str, "--video-ffmpeg-dir", "", "DIR", "", "LLAMA_ARG_VIDEO_FFMPEG_DIR", EX_MTMD | EX_SERVER | EX_CLI | EX_TTS, 0, 0) {
        h("path to the directory containing ffmpeg and ffprobe (default: search in PATH)");
    }
    if add(Opt.Rpc, Kind.Str, "--rpc", "", "SERVERS", "", "LLAMA_ARG_RPC", EX_COMMON, 0, 0) {
        h("comma-separated list of RPC servers (host:port)");
    }
    if add(Opt.LoadMode, Kind.Str, "-lm --load-mode", "", "MODE", "", "LLAMA_ARG_LOAD_MODE", EX_COMMON, 0, 0) {
        h("model loading mode (default: auto)\n- auto: mmap, unless a device does not support it\n- none: no special loading mode\n- mmap: memory-map model (if mmap disabled, slower load but may reduce pageouts if not using mlock)\n- mlock: force system to keep model in RAM rather than swapping or compressing\n- mmap+mlock: mmap + force system to keep model in RAM rather than swapping or compressing\n- dio: use DirectIO if available\n");
    }
    if add(Opt.LazyMode, Kind.Str, "-lzm --lazy-mode", "", "MODE", "", "LLAMA_ARG_LAZY_MODE", EX_COMMON, 0, 0) {
        h("on-demand reading of certain tensors, for example per-layer embeddings (default: auto)\n- on: read the rows of such tensors from disk on demand instead of keeping them resident (requires mmap)\n- auto: on, but only for tensors larger than 4 GiB\n- off: always keep them resident");
    }
    if add(Opt.Numa, Kind.Str, "--numa", "", "TYPE", "", "LLAMA_ARG_NUMA", EX_COMMON, 0, 0) {
        h("attempt optimizations that help on some NUMA systems\n- distribute: spread execution evenly over all nodes\n- isolate: only spawn threads on CPUs on the node that execution started on\n- numactl: use the CPU map provided by numactl\nif run without this previously, it is recommended to drop the system page cache before using this\nsee https://github.com/ggml-org/llama.cpp/issues/1437");
    }
    if add(Opt.Device, Kind.Str, "-dev --device", "", "<dev1,dev2,..>", "", "LLAMA_ARG_DEVICE", EX_COMMON, 0, 0) {
        h("comma-separated list of devices to use for offloading (none = don't offload)\nuse --list-devices to see a list of available devices");
    }
    if add(Opt.ListDevices, Kind.Void, "--list-devices", "", "", "", "", EX_COMMON, 0, 0) {
        h("print list of available devices and exit");
    }
    if add(Opt.OverrideTensor, Kind.Str, "-ot --override-tensor", "", "<tensor name pattern>=<buffer type>,...", "", "LLAMA_ARG_OVERRIDE_TENSOR", EX_COMMON, 0, 0) {
        h("override tensor buffer type");
    }
    if add(Opt.CpuMoe, Kind.Void, "-cmoe --cpu-moe", "", "", "", "LLAMA_ARG_CPU_MOE", EX_COMMON, 0, 0) {
        h("keep all Mixture of Experts (MoE) weights in the CPU");
    }
    if add(Opt.NCpuMoe, Kind.Int, "-ncmoe --n-cpu-moe", "", "N", "", "LLAMA_ARG_N_CPU_MOE", EX_COMMON, 0, 0) {
        h("keep the Mixture of Experts (MoE) weights of the first N layers in the CPU");
    }
    if add(Opt.MoeCacheMib, Kind.Int, "--moe-cache-mib", "", "N", "", "LLAMA_ARG_MOE_CACHE_MIB", EX_COMMON, 0, 0) {
        h("GPU cache size in MiB for the MoE experts kept in the CPU. with multiple GPUs, it is split among them like the layers (--tensor-split) (default: 0, disabled)");
    }
    if add(Opt.MoeCacheCpuShare, Kind.Str, "--moe-cache-cpu-share", "", "F", "", "LLAMA_ARG_MOE_CACHE_CPU_SHARE", EX_COMMON, 0, 0) {
        h("share of the MoE cache misses that the CPU computes from the host experts instead of uploading them to the GPU once the cache is full, from 0 to 1 (default: ");
        hf(0.0, 2);
        h(")");
    }
    if add(Opt.NCpuFfn, Kind.Int, "-ncffn --n-cpu-ffn", "", "N", "", "LLAMA_ARG_N_CPU_FFN", EX_COMMON, 0, 0) {
        h("keep the dense FFN weights of the first N layers in the CPU\n(dense models; for MoE expert weights use --n-cpu-moe)");
    }
    if add(Opt.GpuLayers, Kind.Str, "-ngl --gpu-layers --n-gpu-layers", "", "N", "", "LLAMA_ARG_N_GPU_LAYERS", EX_COMMON, 0, 0) {
        h("max. number of layers to store in VRAM, either an exact number, 'auto', or 'all' (default: ");
        h(if params.n_gpu_layers == -1 { "auto" } else { "all" });
        h(")");
    }
    if add(Opt.SplitMode, Kind.Str, "-sm --split-mode", "", "{none,layer,row,tensor}", "", "LLAMA_ARG_SPLIT_MODE", EX_COMMON, 0, 0) {
        h("how to split the model across multiple GPUs, one of:\n- none: use one GPU only\n- layer (default): split layers and KV across GPUs (pipelined)\n- row: split weight across GPUs by rows (parallelized)\n- tensor: split weights and KV across GPUs (parallelized, EXPERIMENTAL)");
    }
    if add(Opt.TensorSplit, Kind.Str, "-ts --tensor-split", "", "N0,N1,N2,...", "", "LLAMA_ARG_TENSOR_SPLIT", EX_COMMON, 0, 0) {
        h("fraction of the model to offload to each GPU, comma-separated list of proportions, e.g. 3,1");
    }
    if add(Opt.MainGpu, Kind.Int, "-mg --main-gpu", "", "INDEX", "", "LLAMA_ARG_MAIN_GPU", EX_COMMON, 0, 0) {
        h("the GPU to use for the model (with split-mode = none), or for intermediate results and KV (with split-mode = row) (default: ");
        hi(0);
        h(")");
    }
    if add(Opt.Fit, Kind.Str, "-fit --fit", "", "[on|off]", "", "LLAMA_ARG_FIT", EX_COMMON, 0, 0) {
        h("whether to adjust unset arguments to fit in device memory ('on' or 'off', default: '");
        h(if fit::enabled { "on" } else { "off" });
        h("')");
    }
    if add(Opt.FitTarget, Kind.Str, "-fitt --fit-target", "", "MiB0,MiB1,MiB2,...", "", "LLAMA_ARG_FIT_TARGET", EX_COMMON, 0, 0) {
        h("target margin per device for --fit, comma-separated list of values, single value is broadcast across all devices, default: ");
        hi(fit::target_mib);
    }
    if add(Opt.FitCtx, Kind.Int, "-fitc --fit-ctx", "", "N", "", "LLAMA_ARG_FIT_CTX", EX_COMMON, 0, 0) {
        h("minimum ctx size that can be set by --fit option, default: ");
        hi(fit::min_ctx);
    }
    if add(Opt.CheckTensors, Kind.Void, "--check-tensors", "", "", "", "", EX_COMMON, 0, 0) {
        h("check model tensor data for invalid values (default: ");
        h(if false { "true" } else { "false" });
        h(")");
    }
    if add(Opt.OverrideKv, Kind.Str, "--override-kv", "", "KEY=TYPE:VALUE,...", "", "", EX_COMMON, 0, 0) {
        h("advanced option to override model metadata by key. to specify multiple overrides, either use comma-separated values.\ntypes: int, float, bool, str. example: --override-kv tokenizer.ggml.add_bos_token=bool:false,tokenizer.ggml.add_eos_token=bool:false");
    }
    if add(Opt.OpOffload, Kind.Bool, "--op-offload", "--no-op-offload", "", "", "", EX_COMMON, 0, 0) {
        h("whether to offload host tensor operations to device (default: ");
        h(if false { "false" } else { "true" });
        h(")");
    }
    if add(Opt.Lora, Kind.Str, "--lora", "", "FNAME", "", "", EX_COMMON | EX_EXPORT_LORA, 0, 0) {
        h("path to LoRA adapter (use comma-separated values to load multiple adapters)");
    }
    if add(Opt.LoraScaled, Kind.Str, "--lora-scaled", "", "FNAME:SCALE,...", "", "", EX_COMMON | EX_EXPORT_LORA, 0, 0) {
        h("path to LoRA adapter with user defined scaling (format: FNAME:SCALE,...)\nnote: use comma-separated values");
    }
    if add(Opt.ControlVector, Kind.Str, "--control-vector", "", "FNAME", "", "", EX_COMMON, 0, 0) {
        h("add a control vector\nnote: use comma-separated values to add multiple control vectors");
    }
    if add(Opt.ControlVectorScaled, Kind.Str, "--control-vector-scaled", "", "FNAME:SCALE,...", "", "", EX_COMMON, 0, 0) {
        h("add a control vector with user defined scaling SCALE\nnote: use comma-separated values (format: FNAME:SCALE,...)");
    }
    if add(Opt.ControlVectorLayerRange, Kind.StrStr, "--control-vector-layer-range", "", "START", "END", "", EX_COMMON, 0, 0) {
        h("layer range to apply the control vector(s) to, start and end inclusive");
    }
    if add(Opt.Alias, Kind.Str, "-a --alias", "", "STRING", "", "LLAMA_ARG_ALIAS", EX_SERVER, 0, 0) {
        h("set model name aliases, comma-separated (to be used by API)");
    }
    if add(Opt.Tags, Kind.Str, "--tags", "", "STRING", "", "LLAMA_ARG_TAGS", EX_SERVER, 0, 0) {
        h("set model tags, comma-separated (informational, not used for routing)");
    }
    if add(Opt.Model, Kind.Str, "-m --model", "", "FNAME", "", "LLAMA_ARG_MODEL", EX_COMMON | EX_EXPORT_LORA | EX_DOWNLOAD | EX_TOKENIZE, 0, 0) {
        h(if ex == Example.ExportLora { "model path from which to load base model" } else { "model path to load" });
    }
    if add(Opt.Ple, Kind.Str, "--ple --shard", "", "FNAME,...", "", "LLAMA_ARG_SHARD", EX_COMMON, 0, 0) {
        h("extra GGUF shards for the model, comma separated; may be given more than once\npaths may live anywhere, including on different drives; --ple is an alias for --shard");
    }
    if add(Opt.ModelUrl, Kind.Str, "-mu --model-url", "", "MODEL_URL", "", "LLAMA_ARG_MODEL_URL", EX_COMMON | EX_DOWNLOAD | EX_TOKENIZE, 0, 0) {
        h("model download url (default: unused)");
    }
    if add(Opt.DockerRepo, Kind.Str, "-dr --docker-repo", "", "[<repo>/]<model>[:quant]", "", "LLAMA_ARG_DOCKER_REPO", EX_COMMON | EX_DOWNLOAD | EX_TOKENIZE, 0, 0) {
        h("Docker Hub model repository. repo is optional, default to ai/. quant is optional, default to :latest.\nexample: gemma3\n(default: unused)");
    }
    if add(Opt.HfRepo, Kind.Str, "-hf -hfr --hf-repo", "", "<user>/<model>[:quant]", "", "LLAMA_ARG_HF_REPO", EX_COMMON | EX_DOWNLOAD | EX_TOKENIZE, 0, 0) {
        h("Hugging Face model repository; quant is optional, case-insensitive, default to Q4_K_M, or falls back to the first file in the repo if Q4_K_M doesn't exist.\nmmproj is also downloaded automatically if available. to disable, add --no-mmproj\nexample: ggml-org/GLM-4.7-Flash-GGUF:Q4_K_M\n(default: unused)");
    }
    if add(Opt.HfFile, Kind.Str, "-hff --hf-file", "", "FILE", "", "LLAMA_ARG_HF_FILE", EX_COMMON | EX_DOWNLOAD | EX_TOKENIZE, 0, 0) {
        h("Hugging Face model file. If specified, it will override the quant in --hf-repo (default: unused)");
    }
    if add(Opt.HfToken, Kind.Str, "-hft --hf-token", "", "TOKEN", "", "HF_TOKEN", EX_COMMON | EX_DOWNLOAD | EX_TOKENIZE, 0, 0) {
        h("Hugging Face access token (default: value from HF_TOKEN environment variable)");
    }
    if add(Opt.Junk, Kind.Int, "--junk", "", "N", "", "", EX_PASSKEY | EX_PARALLEL, 0, 0) {
        h("number of times to repeat the junk text (default: ");
        hi(params.n_junk);
        h(")");
    }
    if add(Opt.Output, Kind.Str, "-o --output --output-file", "", "FNAME", "", "", EX_IMATRIX | EX_CVECTOR_GENERATOR | EX_EXPORT_LORA | EX_TTS | EX_FINETUNE | EX_RESULTS | EX_EXPORT_GRAPH_OPS | EX_CLI, 0, 0) {
        h("output file (default: '");
        h(params.out_file);
        h("')");
    }
    if add(Opt.Pps, Kind.Void, "-pps", "", "", "", "", EX_BENCH | EX_PARALLEL, 0, 0) {
        h("is the prompt shared across parallel sequences (default: ");
        h(if params.is_pp_shared { "true" } else { "false" });
        h(")");
    }
    if add(Opt.Tgs, Kind.Void, "-tgs", "", "", "", "", EX_BENCH | EX_PARALLEL, 0, 0) {
        h("is the text generation separated across the different sequences (default: ");
        h("false");
        h(")");
    }
    if add(Opt.EmbdNormalize, Kind.Int, "--embd-normalize", "", "N", "", "", EX_EMBEDDING | EX_SERVER | EX_DEBUG, 0, 0) {
        h("normalisation for embeddings (default: ");
        hi(2);
        h(") (-1=none, 0=max absolute int16, 1=taxicab, 2=euclidean, >2=p-norm)");
    }
    if add(Opt.Host, Kind.Str, "--host", "", "HOST", "", "LLAMA_ARG_HOST", EX_SERVER, 0, 0) {
        h("IP addresses to listen on, comma-separated, or UNIX socket paths ending in .sock; with multiple TCP addresses, :: binds IPv6 only; overlapping addresses result in undefined behavior (default: ");
        h(params.hostname);
        h(")");
    }
    if add(Opt.Port, Kind.Int, "--port", "", "PORT", "", "LLAMA_ARG_PORT", EX_SERVER, 0, 0) {
        h("port to listen (default: ");
        hi(params.port);
        h(")");
    }
    if add(Opt.ReusePort, Kind.Void, "--reuse-port", "", "", "", "LLAMA_ARG_REUSE_PORT", EX_SERVER, 0, 0) {
        h("allow multiple sockets to bind to the same port (default: ");
        h(if false { "enabled" } else { "disabled" });
        h(")");
    }
    if add(Opt.Path, Kind.Str, "--path", "", "PATH", "", "LLAMA_ARG_STATIC_PATH", EX_SERVER, 0, 0) {
        h("path to serve static files from (default: ");
        h(params.public_path);
        h(")");
    }
    if add(Opt.CorsOrigins, Kind.Str, "--cors-origins", "", "ORIGINS", "", "LLAMA_ARG_CORS_ORIGINS", EX_SERVER, 0, 0) {
        h("comma-separated list of allowed origins for CORS (default: ");
        h("*");
        h(")\nif set to special value 'localhost', only allow same-origin requests (e.g. the built-in UI)");
    }
    if add(Opt.CorsMethods, Kind.Str, "--cors-methods", "", "METHODS", "", "LLAMA_ARG_CORS_METHODS", EX_SERVER, 0, 0) {
        h("comma-separated list of allowed methods for CORS (default: ");
        h("GET, POST, DELETE, OPTIONS");
        h(")");
    }
    if add(Opt.CorsHeaders, Kind.Str, "--cors-headers", "", "HEADERS", "", "LLAMA_ARG_CORS_HEADERS", EX_SERVER, 0, 0) {
        h("comma-separated list of allowed headers for CORS (default: ");
        h("*");
        h(")");
    }
    if add(Opt.CorsCredentials, Kind.Bool, "--cors-credentials", "--no-cors-credentials", "", "", "LLAMA_ARG_CORS_CREDENTIALS", EX_SERVER, 0, 0) {
        h("whether to allow credentials for CORS (default: ");
        h(if true { "enabled" } else { "disabled" });
        h(")\nnote: if this is enabled and --cors-origins is set to * (default), the Origin header will be echoed back, and credentials will always be allowed");
    }
    if add(Opt.ApiPrefix, Kind.Str, "--api-prefix", "", "PREFIX", "", "LLAMA_ARG_API_PREFIX", EX_SERVER, 0, 0) {
        h("prefix path the server serves from, without the trailing slash (default: ");
        h("");
        h(")");
    }
    if add(Opt.UiConfig, Kind.Str, "--ui-config --webui-config", "", "JSON", "", "LLAMA_ARG_UI_CONFIG", EX_SERVER, 0, 0) {
        h("JSON that provides default UI settings (overrides UI defaults)");
    }
    if add(Opt.UiConfigFile, Kind.Str, "--ui-config-file --webui-config-file", "", "PATH", "", "LLAMA_ARG_UI_CONFIG_FILE", EX_SERVER, 0, 0) {
        h("JSON file that provides default UI settings (overrides UI defaults)");
    }
    if add(Opt.UiMcpProxy, Kind.Bool, "--ui-mcp-proxy --webui-mcp-proxy", "--no-ui-mcp-proxy --no-webui-mcp-proxy", "", "", "LLAMA_ARG_UI_MCP_PROXY", EX_SERVER, 0, 0) {
        h("experimental: whether to enable MCP CORS proxy, for Llama Chat to reach public MCP servers without CORS headers; local and private addresses are refused unless --ui-mcp-proxy-allow-private-urls is set - do not enable in untrusted environments (default: disabled)\nnote: for security reasons, this will limit --cors-origins to localhost by default");
    }
    if add(Opt.UiMcpProxyAllowPrivateUrls, Kind.Void, "--ui-mcp-proxy-allow-private-urls", "", "", "", "LLAMA_ARG_UI_MCP_PROXY_ALLOW_PRIVATE_URLS", EX_SERVER, 0, 0) {
        h("allow the MCP CORS proxy to reach local or private addresses (e.g. MCP servers on localhost or LAN) - anyone who can reach the proxy can then reach your local network (default: disabled, only public addresses)\nnote: only has effect with --ui-mcp-proxy");
    }
    if add(Opt.Tools, Kind.Str, "--tools", "", "TOOL1,TOOL2,...", "", "LLAMA_ARG_TOOLS", EX_SERVER, 0, 0) {
        h("experimental: whether to enable built-in tools for AI agents - do not enable in untrusted environments (default: no tools)\nspecify \"all\" to enable all tools\navailable tools: read_file, file_glob_search, grep_search, exec_shell_command, write_file, edit_file, get_info\nnote: for security reasons, this will limit --cors-origins to localhost by default");
    }
    if add(Opt.ToolsRuntime, Kind.Str, "--tools-runtime", "", "OPTION", "", "LLAMA_ARG_TOOLS_RUNTIME", EX_SERVER, 0, 0) {
        h("experimental: run tools in a separate runtime environment (default: none, use host environment)\navailable options:\n  'docker:<image>', 'podman:<image>': spin up a new container and reuse it for all invocations, clean up on server exit\n  'docker-container:<id>', 'podman-container:<id>': use an existing container by ID, won't stop on server exit\n  'ssh:<target>': run tools on a remote POSIX host over SSH, key-based auth and a trusted host key are required\n");
    }
    if add(Opt.AllowUnauthenticatedTools, Kind.Void, "--allow-unauthenticated-tools", "", "", "", "LLAMA_ARG_ALLOW_UNAUTHENTICATED_TOOLS", EX_SERVER, 0, 0) {
        h("allow server tools, MCP servers or MCP proxy on a non-loopback address without an API key - only for trusted networks (default: disabled)");
    }
    if add(Opt.McpServersConfig, Kind.Str, "--mcp-servers-config", "", "PATH", "", "LLAMA_ARG_MCP_SERVERS_CONFIG", EX_SERVER, 0, 0) {
        h("experimental: path to JSON file with MCP server definitions (Cursor-compatible format) - do not enable in untrusted environments (default: none)\nnote: for security reasons, this will limit --cors-origins to localhost by default");
    }
    if add(Opt.McpServersJson, Kind.Str, "--mcp-servers-json", "", "JSON", "", "LLAMA_ARG_MCP_SERVERS_JSON", EX_SERVER, 0, 0) {
        h("experimental: inline JSON with MCP server definitions (Cursor-compatible format) - do not enable in untrusted environments (default: none)\nnote: for security reasons, this will limit --cors-origins to localhost by default");
    }
    if add(Opt.Agent, Kind.Bool, "-ag --agent", "-no-ag --no-agent", "", "", "LLAMA_ARG_AGENT", EX_SERVER, 0, 0) {
        h("whether to enable CORS proxy and all built-in tools - do not enable in untrusted environments (default: disabled)\nnote: for security reasons, this will limit --cors-origins to localhost by default");
    }
    if add(Opt.Ui, Kind.Bool, "--ui --webui", "--no-ui --no-webui", "", "", "LLAMA_ARG_UI", EX_SERVER, 0, 0) {
        h("whether to enable Llama Chat, the built-in chat app (default: ");
        h(if params.webui { "enabled" } else { "disabled" });
        h(")");
    }
    if add(Opt.Connect, Kind.Void, "--connect", "", "", "", "LLAMA_ARG_CONNECT", EX_SERVER, 0, 0) {
        h("(experimental) open a peer-to-peer tunnel for server using llama-connect (default: ");
        h(if false { "enabled" } else { "disabled" });
        h(")");
    }
    if add(Opt.ConnectCode, Kind.Str, "--connect-code", "", "CODE", "", "LLAMA_ARG_CONNECT_CODE", EX_SERVER, 0, 0) {
        h("manually specify a 40-character code for --connect (default: generate a new one for each run)");
    }
    if add(Opt.Embedding, Kind.Void, "--embedding --embeddings", "", "", "", "LLAMA_ARG_EMBEDDINGS", EX_SERVER | EX_DEBUG, 0, 0) {
        h("restrict to only support embedding use case; use only with dedicated embedding models (default: ");
        h(if false { "enabled" } else { "disabled" });
        h(")");
    }
    if add(Opt.Rerank, Kind.Void, "--rerank --reranking", "", "", "", "LLAMA_ARG_RERANKING", EX_SERVER, 0, 0) {
        h("enable reranking endpoint on server (default: ");
        h("disabled");
        h(")");
    }
    if add(Opt.ApiKey, Kind.Str, "--api-key", "", "KEY", "", "LLAMA_API_KEY", EX_SERVER, 0, 0) {
        h("API key to use for authentication, multiple keys can be provided as a comma-separated list (default: none)");
    }
    if add(Opt.ApiKeyFile, Kind.Str, "--api-key-file", "", "FNAME", "", "LLAMA_ARG_API_KEY_FILE", EX_SERVER, 0, 0) {
        h("path to file containing API keys, one per line; lines starting with a hash are treated as comments (default: none)");
    }
    if add(Opt.SslKeyFile, Kind.Str, "--ssl-key-file", "", "FNAME", "", "LLAMA_ARG_SSL_KEY_FILE", EX_SERVER, 0, 0) {
        h("path to file a PEM-encoded SSL private key");
    }
    if add(Opt.SslCertFile, Kind.Str, "--ssl-cert-file", "", "FNAME", "", "LLAMA_ARG_SSL_CERT_FILE", EX_SERVER, 0, 0) {
        h("path to file a PEM-encoded SSL certificate");
    }
    if add(Opt.ChatTemplateKwargs, Kind.Str, "--chat-template-kwargs", "", "STRING", "", "LLAMA_ARG_CHAT_TEMPLATE_KWARGS", EX_SERVER | EX_CLI, 0, 0) {
        h("sets additional params for the json template parser, must be a valid json object string, e.g. '{\"key1\":\"value1\",\"key2\":\"value2\"}'");
    }
    if add(Opt.Timeout, Kind.Int, "-to --timeout", "", "N", "", "LLAMA_ARG_TIMEOUT", EX_SERVER, 0, 0) {
        h("server read/write timeout in seconds (default: ");
        hi(3600);
        h(")");
    }
    if add(Opt.SsePingInterval, Kind.Int, "--sse-ping-interval", "", "N", "", "LLAMA_ARG_SSE_PING_INTERVAL", EX_SERVER, 0, 0) {
        h("server SSE ping interval in seconds (-1 = disabled, default: ");
        hi(30);
        h(")");
    }
    if add(Opt.ThreadsHttp, Kind.Int, "--threads-http", "", "N", "", "LLAMA_ARG_THREADS_HTTP", EX_SERVER, 0, 0) {
        h("number of threads used to process HTTP requests (default: ");
        hi(-1);
        h(")");
    }
    if add(Opt.CachePrompt, Kind.Bool, "--cache-prompt", "--no-cache-prompt", "", "", "LLAMA_ARG_CACHE_PROMPT", EX_SERVER, 0, 0) {
        h("whether to enable prompt caching (default: ");
        h(if true { "enabled" } else { "disabled" });
        h(")");
    }
    if add(Opt.CacheReuse, Kind.Int, "--cache-reuse", "", "N", "", "LLAMA_ARG_CACHE_REUSE", EX_SERVER, 0, 0) {
        h("min chunk size to attempt reusing from the cache via KV shifting, requires prompt caching to be enabled (default: ");
        hi(0);
        h(")\n[(card)](https://ggml.ai/f0.png)");
    }
    if add(Opt.Metrics, Kind.Void, "--metrics", "", "", "", "LLAMA_ARG_ENDPOINT_METRICS", EX_SERVER, 0, 0) {
        h("enable prometheus compatible metrics endpoint (default: ");
        h(if params.endpoint_metrics { "enabled" } else { "disabled" });
        h(")");
    }
    if add(Opt.Props, Kind.Void, "--props", "", "", "", "LLAMA_ARG_ENDPOINT_PROPS", EX_SERVER, 0, 0) {
        h("enable changing global properties via POST /props (default: ");
        h(if false { "enabled" } else { "disabled" });
        h(")");
    }
    if add(Opt.Slots, Kind.Bool, "--slots", "--no-slots", "", "", "LLAMA_ARG_ENDPOINT_SLOTS", EX_SERVER, 0, 0) {
        h("expose slots monitoring endpoint (default: ");
        h(if params.endpoint_slots { "enabled" } else { "disabled" });
        h(")");
    }
    if add(Opt.SlotSavePath, Kind.Str, "--slot-save-path", "", "PATH", "", "LLAMA_ARG_SLOT_SAVE_PATH", EX_SERVER, 0, 0) {
        h("path to save slot kv cache (default: disabled)");
    }
    if add(Opt.MediaPath, Kind.Str, "--media-path", "", "PATH", "", "", EX_SERVER, 0, 0) {
        h("directory for loading local media files; files can be accessed via file:// URLs using relative paths (default: disabled)");
    }
    if add(Opt.MediaAllowPrivateUrls, Kind.Void, "--media-allow-private-urls", "", "", "", "LLAMA_ARG_MEDIA_ALLOW_PRIVATE_URLS", EX_SERVER, 0, 0) {
        h("allow http(s) media URLs (e.g. image_url) that point to local or private addresses (default: disabled, only public addresses)");
    }
    if add(Opt.ModelsDir, Kind.Str, "--models-dir", "", "PATH", "", "LLAMA_ARG_MODELS_DIR", EX_SERVER, 0, 0) {
        h("directory containing models for the router server (default: disabled)");
    }
    if add(Opt.ModelsPreset, Kind.Str, "--models-preset", "", "PATH", "", "LLAMA_ARG_MODELS_PRESET", EX_SERVER, 0, 0) {
        h("path to INI file containing model presets for the router server (default: disabled)");
    }
    if add(Opt.ModelsMax, Kind.Int, "--models-max", "", "N", "", "LLAMA_ARG_MODELS_MAX", EX_SERVER, 0, 0) {
        h("for router server, maximum number of models to load simultaneously (default: ");
        hi(4);
        h(", 0 = unlimited)");
    }
    if add(Opt.ModelsMemoryMargin, Kind.Int, "--models-memory-margin", "", "N", "", "LLAMA_ARG_MODELS_MEMORY_MARGIN", EX_SERVER, 0, 0) {
        h("for router server, MiB of device memory to keep free per device, unload LRU models until a new one fits (default: ");
        hi(0);
        h(", 0 = disabled)");
    }
    if add(Opt.ModelsAutoload, Kind.Bool, "--models-autoload", "--no-models-autoload", "", "", "LLAMA_ARG_MODELS_AUTOLOAD", EX_SERVER, 0, 0) {
        h("for router server, whether to automatically load models (default: ");
        h(if true { "enabled" } else { "disabled" });
        h(")");
    }
    if add(Opt.Jinja, Kind.Bool, "--jinja", "--no-jinja", "", "", "LLAMA_ARG_JINJA", EX_SERVER | EX_COMPLETION | EX_CLI | EX_MTMD, 0, 0) {
        h("whether to use jinja template engine for chat (default: ");
        h(if params.use_jinja { "enabled" } else { "disabled" });
        h(")");
    }
    if add(Opt.ReasoningFormat, Kind.Str, "--reasoning-format", "", "FORMAT", "", "LLAMA_ARG_THINK", EX_SERVER | EX_COMPLETION | EX_CLI, 0, 0) {
        h("controls whether thought tags are allowed and/or extracted from the response, and in which format they're returned; one of:\n- none: leaves thoughts unparsed in `message.content`\n- deepseek: puts thoughts in `message.reasoning_content`\n- deepseek-legacy: keeps `<think>` tags in `message.content` while also populating `message.reasoning_content`\n(default: auto)");
    }
    if add(Opt.Reasoning, Kind.Str, "-rea --reasoning", "", "[on|off|auto]", "", "LLAMA_ARG_REASONING", EX_SERVER | EX_COMPLETION | EX_CLI, 0, 0) {
        h("Use reasoning/thinking in the chat ('on', 'off', or 'auto', default: 'auto' (detect from template))");
    }
    if add(Opt.ReasoningEffort, Kind.Str, "--reasoning-effort", "", "LEVEL", "", "LLAMA_ARG_REASONING_EFFORT", EX_SERVER | EX_COMPLETION | EX_CLI, 0, 0) {
        h("reasoning effort level given to the chat template: 'default' to keep the template default,\nor a level such as 'minimal', 'low', 'medium', 'high', 'xhigh' or 'max' (default: default)");
    }
    if add(Opt.ReasoningBudget, Kind.Int, "--reasoning-budget", "", "N", "", "LLAMA_ARG_THINK_BUDGET", EX_SERVER | EX_COMPLETION | EX_CLI, 0, 0) {
        h("token budget for thinking: -1 for unrestricted, 0 for immediate end, N>0 for token budget (default: -1)");
    }
    if add(Opt.ReasoningBudgetMessage, Kind.Str, "--reasoning-budget-message", "", "MESSAGE", "", "LLAMA_ARG_THINK_BUDGET_MESSAGE", EX_SERVER | EX_COMPLETION | EX_CLI, 0, 0) {
        h("message injected before the end-of-thinking tag when reasoning budget is exhausted (default: none)");
    }
    if add(Opt.ReasoningPreserve, Kind.Bool, "--reasoning-preserve", "--no-reasoning-preserve", "", "", "LLAMA_ARG_REASONING_PRESERVE", EX_SERVER | EX_COMPLETION | EX_CLI, 0, 0) {
        h("preserve reasoning trace in the full history, not just the last assistant message (default: enabled)\ncompatible with certain templates having 'supports_preserve_reasoning' capability\nexample: https://docs.z.ai/guides/capabilities/thinking-mode#preserved-thinking");
    }
    if add(Opt.ChatTemplate, Kind.Str, "--chat-template", "", "JINJA_TEMPLATE", "", "LLAMA_ARG_CHAT_TEMPLATE", EX_COMPLETION | EX_CLI | EX_SERVER | EX_MTMD, 0, 0) {
        h("set custom jinja chat template (default: template taken from model's metadata)\nif suffix/prefix are specified, template will be disabled\nonly commonly used templates are accepted (unless --jinja is set before this flag):\nlist of built-in templates:\n");
        h(builtin_templates());
    }
    if add(Opt.ChatTemplateFile, Kind.Str, "--chat-template-file", "", "JINJA_TEMPLATE_FILE", "", "LLAMA_ARG_CHAT_TEMPLATE_FILE", EX_COMPLETION | EX_CLI | EX_SERVER, 0, 0) {
        h("set custom jinja chat template file (default: template taken from model's metadata)\nif suffix/prefix are specified, template will be disabled\nonly commonly used templates are accepted (unless --jinja is set before this flag):\nlist of built-in templates:\n");
        h(builtin_templates());
    }
    if add(Opt.SkipChatParsing, Kind.Bool, "--skip-chat-parsing", "--no-skip-chat-parsing", "", "", "LLAMA_ARG_SKIP_CHAT_PARSING", EX_COMPLETION | EX_CLI | EX_SERVER, 0, 0) {
        h("force a pure content parser, even if a Jinja template is specified; model will output everything in the content section, including any reasoning and/or tool calls (default: disabled)");
    }
    if add(Opt.PrefillAssistant, Kind.Bool, "--prefill-assistant", "--no-prefill-assistant", "", "", "LLAMA_ARG_PREFILL_ASSISTANT", EX_SERVER, 0, 0) {
        h("whether to prefill the assistant's response if the last message is an assistant message (default: prefill enabled)\nwhen this flag is set, if the last message is an assistant message then it will be treated as a full message and not prefilled\n");
    }
    if add(Opt.SlotPromptSimilarity, Kind.Str, "-sps --slot-prompt-similarity", "", "SIMILARITY", "", "", EX_SERVER, 0, 0) {
        h("how much the prompt of a request must match the prompt of a slot in order to use that slot (default: ");
        hf(0.1, 2);
        h(", 0.0 = disabled)\n");
    }
    if add(Opt.LoraInitWithoutApply, Kind.Void, "--lora-init-without-apply", "", "", "", "", EX_SERVER, 0, 0) {
        h("load LoRA adapters without applying them (apply later via POST /lora-adapters) (default: ");
        h(if false { "enabled" } else { "disabled" });
        h(")");
    }
    if add(Opt.SleepIdleSeconds, Kind.Int, "--sleep-idle-seconds", "", "SECONDS", "", "", EX_SERVER, 0, 0) {
        h("number of seconds of idleness after which the server will sleep (default: ");
        hi(-1);
        h("; -1 = disabled)");
    }
    if add(Opt.SimpleIo, Kind.Void, "--simple-io", "", "", "", "", EX_COMPLETION | EX_CLI, 0, 0) {
        h("use basic IO for better compatibility in subprocesses and limited consoles");
    }
    if add(Opt.LogDisable, Kind.Void, "--log-disable", "", "", "", "", EX_COMMON, 0, 0) {
        h("Log disable");
    }
    if add(Opt.LogFile, Kind.Str, "--log-file", "", "FNAME", "", "LLAMA_ARG_LOG_FILE", EX_COMMON, 0, 0) {
        h("Log to file");
    }
    if add(Opt.LogJsonl, Kind.Bool, "--log-jsonl", "--no-log-jsonl", "", "", "LLAMA_ARG_LOG_JSONL", EX_COMMON, 0, 0) {
        h("Log as JSONL (one JSON object per line) to stdout, this also disables colored logging (default: disabled)");
    }
    if add(Opt.LogPromptsDir, Kind.Str, "--log-prompts-dir", "", "PATH", "", "", EX_SERVER | EX_CLI, 0, 0) {
        h("Log prompts to directory (auto-created if not present; only used for debugging, default: disabled)");
    }
    if add(Opt.LogColors, Kind.Str, "--log-colors", "", "[on|off|auto]", "", "LLAMA_ARG_LOG_COLORS", EX_COMMON, 0, 0) {
        h("Set colored logging ('on', 'off', or 'auto', default: 'auto')\n'auto' enables colors when output is to a terminal");
    }
    if add(Opt.Verbose, Kind.Void, "-v --verbose --log-verbose", "", "", "", "", EX_COMMON, 0, 0) {
        h("Set verbosity level to infinity (i.e. log all messages, useful for debugging)");
    }
    if add(Opt.Offline, Kind.Void, "--offline", "", "", "", "LLAMA_ARG_OFFLINE", EX_COMMON | EX_DOWNLOAD | EX_TOKENIZE, 0, 0) {
        h("Offline mode: forces use of cache, prevents network access");
    }
    if add(Opt.Verbosity, Kind.Int, "-lv --verbosity --log-verbosity", "", "N", "", "LLAMA_ARG_LOG_VERBOSITY", EX_COMMON, 0, 0) {
        h("Set the verbosity threshold. Messages with a higher verbosity will be ignored. Values:\n - 0: generic output\n - 1: error\n - 2: warning\n - 3: info\n - 4: trace (more info)\n - 5: debug\n(default: ");
        hi(3);
        h(")\n");
    }
    if add(Opt.LogPrefix, Kind.Bool, "--log-prefix", "--no-log-prefix", "", "", "LLAMA_ARG_LOG_PREFIX", EX_COMMON, 0, 0) {
        h("Enable prefix in log messages");
    }
    if add(Opt.LogTimestamps, Kind.Bool, "--log-timestamps", "--no-log-timestamps", "", "", "LLAMA_ARG_LOG_TIMESTAMPS", EX_COMMON, 0, 0) {
        h("Enable timestamps in log messages");
    }
    if add(Opt.SpecDraftHf, Kind.Str, "--spec-draft-hf -hfd -hfrd --hf-repo-draft", "", "<user>/<model>[:quant]", "", "LLAMA_ARG_SPEC_DRAFT_HF_REPO", EX_SPECULATIVE | EX_SERVER | EX_CLI, 0, SPEC) {
        h("Same as --hf-repo, but for the draft model (default: unused)");
    }
    if add(Opt.SpecDraftThreads, Kind.Int, "--spec-draft-threads -td --threads-draft", "", "N", "", "", EX_SPECULATIVE | EX_SERVER | EX_CLI, 0, SPEC) {
        h("number of threads to use during generation (default: same as --threads)");
    }
    if add(Opt.SpecDraftThreadsBatch, Kind.Int, "--spec-draft-threads-batch -tbd --threads-batch-draft", "", "N", "", "", EX_SPECULATIVE | EX_SERVER | EX_CLI, 0, SPEC) {
        h("number of threads to use during batch and prompt processing (default: same as --threads-draft)");
    }
    if add(Opt.SpecDraftCpuMask, Kind.Str, "--spec-draft-cpu-mask -Cd --cpu-mask-draft", "", "M", "", "", EX_SPECULATIVE | EX_SERVER | EX_CLI, 0, SPEC) {
        h("Draft model CPU affinity mask. Complements cpu-range-draft (default: same as --cpu-mask)");
    }
    if add(Opt.SpecDraftCpuRange, Kind.Str, "--spec-draft-cpu-range -Crd --cpu-range-draft", "", "lo-hi", "", "", EX_SPECULATIVE | EX_SERVER | EX_CLI, 0, SPEC) {
        h("Ranges of CPUs for affinity. Complements --cpu-mask-draft");
    }
    if add(Opt.SpecDraftCpuStrict, Kind.Int, "--spec-draft-cpu-strict --cpu-strict-draft", "", "<0|1>", "", "", EX_SPECULATIVE | EX_SERVER | EX_CLI, 0, SPEC) {
        h("Use strict CPU placement for draft model (default: same as --cpu-strict)");
    }
    if add(Opt.SpecDraftPrio, Kind.Int, "--spec-draft-prio --prio-draft", "", "N", "", "", EX_SPECULATIVE | EX_SERVER | EX_CLI, 0, SPEC) {
        h("set draft process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime (default: ");
        hi(0);
        h(")\n");
    }
    if add(Opt.SpecDraftPoll, Kind.Int, "--spec-draft-poll --poll-draft", "", "<0|1>", "", "", EX_SPECULATIVE | EX_SERVER | EX_CLI, 0, SPEC) {
        h("Use polling to wait for draft model work (default: same as --poll)");
    }
    if add(Opt.SpecDraftCpuMaskBatch, Kind.Str, "--spec-draft-cpu-mask-batch -Cbd --cpu-mask-batch-draft", "", "M", "", "", EX_SPECULATIVE | EX_SERVER | EX_CLI, 0, SPEC) {
        h("Draft model CPU affinity mask. Complements cpu-range-draft (default: same as --cpu-mask)");
    }
    if add(Opt.SpecDraftCpuStrictBatch, Kind.Int, "--spec-draft-cpu-strict-batch --cpu-strict-batch-draft", "", "<0|1>", "", "", EX_SPECULATIVE | EX_SERVER | EX_CLI, 0, SPEC) {
        h("Use strict CPU placement for draft model (default: --cpu-strict-draft)");
    }
    if add(Opt.SpecDraftPrioBatch, Kind.Int, "--spec-draft-prio-batch --prio-batch-draft", "", "N", "", "", EX_SPECULATIVE | EX_SERVER | EX_CLI, 0, SPEC) {
        h("set draft process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime (default: ");
        hi(0);
        h(")\n");
    }
    if add(Opt.SpecDraftPollBatch, Kind.Int, "--spec-draft-poll-batch --poll-batch-draft", "", "<0|1>", "", "", EX_SPECULATIVE | EX_SERVER | EX_CLI, 0, SPEC) {
        h("Use polling to wait for draft model work (default: --poll-draft)");
    }
    if add(Opt.SpecDraftTypeK, Kind.Str, "--spec-draft-type-k -ctkd --cache-type-k-draft", "", "TYPE", "", "LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_K", EX_COMMON, 0, 0) {
        h("KV cache data type for K for the draft model\nallowed values: ");
        h(KV_CACHE_TYPES);
        h("\n(default: ");
        h("f16");
        h(")");
    }
    if add(Opt.SpecDraftTypeV, Kind.Str, "--spec-draft-type-v -ctvd --cache-type-v-draft", "", "TYPE", "", "LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_V", EX_COMMON, 0, 0) {
        h("KV cache data type for V for the draft model\nallowed values: ");
        h(KV_CACHE_TYPES);
        h("\n(default: ");
        h("f16");
        h(")");
    }
    if add(Opt.SpecDraftOverrideTensor, Kind.Str, "--spec-draft-override-tensor -otd --override-tensor-draft", "", "<tensor name pattern>=<buffer type>,...", "", "", EX_SPECULATIVE | EX_SERVER | EX_CLI, 0, SPEC) {
        h("override tensor buffer type for draft model");
    }
    if add(Opt.SpecDraftCpuMoe, Kind.Void, "--spec-draft-cpu-moe -cmoed --cpu-moe-draft", "", "", "", "LLAMA_ARG_SPEC_DRAFT_CPU_MOE", EX_SPECULATIVE | EX_SERVER | EX_CLI, 0, SPEC) {
        h("keep all Mixture of Experts (MoE) weights in the CPU for the draft model");
    }
    if add(Opt.SpecDraftNCpuMoe, Kind.Int, "--spec-draft-n-cpu-moe --spec-draft-ncmoe -ncmoed --n-cpu-moe-draft", "", "N", "", "LLAMA_ARG_SPEC_DRAFT_N_CPU_MOE", EX_SPECULATIVE | EX_SERVER | EX_CLI, 0, SPEC) {
        h("keep the Mixture of Experts (MoE) weights of the first N layers in the CPU for the draft model");
    }
    if add(Opt.SpecDraftNMax, Kind.Int, "--spec-draft-n-max", "", "N", "", "LLAMA_ARG_SPEC_DRAFT_N_MAX", EX_SPECULATIVE | EX_LOOKUP | EX_SERVER | EX_CLI, 0, SPEC) {
        h("number of tokens to draft for speculative decoding (default: ");
        hi(3);
        h(")");
    }
    if add(Opt.SpecDraftNMin, Kind.Int, "--spec-draft-n-min", "", "N", "", "LLAMA_ARG_SPEC_DRAFT_N_MIN", EX_SPECULATIVE | EX_LOOKUP | EX_SERVER | EX_CLI, 0, SPEC) {
        h("minimum number of draft tokens to use for speculative decoding (default: ");
        hi(0);
        h(")");
    }
    if add(Opt.SpecDraftNUbatch, Kind.Int, "--spec-draft-n-ubatch", "", "N", "", "LLAMA_ARG_SPEC_DRAFT_N_UBATCH", EX_SPECULATIVE | EX_SERVER | EX_CLI, 0, SPEC) {
        h("physical maximum batch size of the draft context, use a smaller value to spare VRAM (default: ");
        hi(0);
        h(" - same as --n-ubatch)");
    }
    if add(Opt.SpecSynthLen, Kind.Str, "--spec-synth-len", "", "L", "", "LLAMA_ARG_SPEC_SYNTH_LEN", EX_SERVER | EX_CLI, 0, SPEC) {
        h("target mean synthetic acceptance length, including the target token (benchmarking only)");
    }
    if add(Opt.SpecSynthRates, Kind.Str, "--spec-synth-rates", "", "P0,P1,...", "", "LLAMA_ARG_SPEC_SYNTH_RATES", EX_SERVER | EX_CLI, 0, SPEC) {
        h("comma-separated unconditional per-position synthetic acceptance probabilities (benchmarking only)");
    }
    if add(Opt.SpecDraftPSplit, Kind.Str, "--spec-draft-p-split --draft-p-split", "", "P", "", "LLAMA_ARG_SPEC_DRAFT_P_SPLIT", EX_SPECULATIVE | EX_SERVER | EX_CLI, 0, SPEC) {
        h("speculative decoding split probability (default: ");
        hf(0.1, 2);
        h(")");
    }
    if add(Opt.SpecDraftPMin, Kind.Str, "--spec-draft-p-min --draft-p-min", "", "P", "", "LLAMA_ARG_SPEC_DRAFT_P_MIN", EX_SPECULATIVE | EX_SERVER | EX_CLI, 0, SPEC) {
        h("minimum speculative decoding probability (greedy) (default: ");
        hf(0.0, 2);
        h(")");
    }
    if add(Opt.SpecDraftBackendSampling, Kind.Bool, "--spec-draft-backend-sampling", "--no-spec-draft-backend-sampling", "", "", "LLAMA_ARG_SPEC_DRAFT_BACKEND_SAMPLING", EX_SPECULATIVE | EX_SERVER | EX_CLI, 0, SPEC) {
        h("offload draft sampling to the backend (default: ");
        h(if true { "enabled" } else { "disabled" });
        h(")");
    }
    if add(Opt.SpecDraftSampling, Kind.Str, "--spec-draft-sampling", "", "{greedy,probabilistic}", "", "LLAMA_ARG_SPEC_DRAFT_SAMPLING", EX_SPECULATIVE | EX_SERVER | EX_CLI, 0, SPEC) {
        h("how the draft is sampled: greedy takes its argmax, probabilistic samples it and has the target verify by rejection sampling (default: ");
        h(if false { "probabilistic" } else { "greedy" });
        h(")");
    }
    if add(Opt.SpecDraftDevice, Kind.Str, "--spec-draft-device -devd --device-draft", "", "<dev1,dev2,..>", "", "", EX_SPECULATIVE | EX_SERVER | EX_CLI, 0, SPEC) {
        h("comma-separated list of devices to use for offloading the draft model (none = don't offload, default: follows --device)\nuse --list-devices to see a list of available devices");
    }
    if add(Opt.SpecDraftNgl, Kind.Str, "--spec-draft-ngl -ngld --gpu-layers-draft --n-gpu-layers-draft", "", "N", "", "LLAMA_ARG_N_GPU_LAYERS_DRAFT", EX_SPECULATIVE | EX_SERVER | EX_CLI, 0, SPEC) {
        h("max. number of draft model layers to store in VRAM, either an exact number, 'auto', or 'all' (default: ");
        h(if -1 == -1 { "auto" } else { "all" });
        h(")");
    }
    if add(Opt.SpecDraftModel, Kind.Str, "--spec-draft-model -md --model-draft", "", "FNAME", "", "LLAMA_ARG_SPEC_DRAFT_MODEL", EX_SPECULATIVE | EX_SERVER | EX_CLI | EX_IMATRIX, 0, SPEC) {
        h("draft model for speculative decoding (default: unused)");
    }
    if add(Opt.SpecType, Kind.Str, "--spec-type", "", "none,draft-simple,draft-eagle3,draft-mtp,draft-dflash,draft-dspark,draft-dflash2,draft-lilicorr,ngram-simple,ngram-map-k,ngram-map-k4v,ngram-mod,ngram-cache", "", "LLAMA_ARG_SPEC_TYPE", EX_SPECULATIVE | EX_SERVER | EX_CLI, 0, SPEC) {
        h("comma-separated list of types of speculative decoding to use (default: ");
        h("none");
        h(")\n");
    }
    if add(Opt.SpecNgramModNMin, Kind.Int, "--spec-ngram-mod-n-min", "", "N", "", "", EX_SPECULATIVE | EX_SERVER | EX_CLI, 0, SPEC) {
        h("minimum number of ngram tokens to use for ngram-based speculative decoding (default: ");
        hi(48);
        h(")");
    }
    if add(Opt.SpecNgramModNMax, Kind.Int, "--spec-ngram-mod-n-max", "", "N", "", "", EX_SPECULATIVE | EX_SERVER | EX_CLI, 0, SPEC) {
        h("maximum number of ngram tokens to use for ngram-based speculative decoding (default: ");
        hi(64);
        h(")");
    }
    if add(Opt.SpecNgramModNMatch, Kind.Int, "--spec-ngram-mod-n-match", "", "N", "", "", EX_SPECULATIVE | EX_SERVER | EX_CLI, 0, SPEC) {
        h("ngram-mod lookup length (default: ");
        hi(24);
        h(")");
    }
    if add(Opt.SpecNgramSimpleSizeN, Kind.Int, "--spec-ngram-simple-size-n", "", "N", "", "", EX_SPECULATIVE | EX_SERVER | EX_CLI, 0, SPEC) {
        h("ngram size N for ngram-simple speculative decoding, length of lookup n-gram (default: ");
        hi(12);
        h(")");
    }
    if add(Opt.SpecNgramSimpleSizeM, Kind.Int, "--spec-ngram-simple-size-m", "", "N", "", "", EX_SPECULATIVE | EX_SERVER | EX_CLI, 0, SPEC) {
        h("ngram size M for ngram-simple speculative decoding, length of draft m-gram (default: ");
        hi(48);
        h(")");
    }
    if add(Opt.SpecNgramSimpleMinHits, Kind.Int, "--spec-ngram-simple-min-hits", "", "N", "", "", EX_SPECULATIVE | EX_SERVER | EX_CLI, 0, SPEC) {
        h("minimum hits for ngram-simple speculative decoding (default: ");
        hi(1);
        h(")");
    }
    if add(Opt.SpecNgramMapKSizeN, Kind.Int, "--spec-ngram-map-k-size-n", "", "N", "", "", EX_SPECULATIVE | EX_SERVER | EX_CLI, 0, SPEC) {
        h("ngram size N for ngram-map-k speculative decoding, length of lookup n-gram (default: ");
        hi(12);
        h(")");
    }
    if add(Opt.SpecNgramMapKSizeM, Kind.Int, "--spec-ngram-map-k-size-m", "", "N", "", "", EX_SPECULATIVE | EX_SERVER | EX_CLI, 0, SPEC) {
        h("ngram size M for ngram-map-k speculative decoding, length of draft m-gram (default: ");
        hi(48);
        h(")");
    }
    if add(Opt.SpecNgramMapKMinHits, Kind.Int, "--spec-ngram-map-k-min-hits", "", "N", "", "", EX_SPECULATIVE | EX_SERVER | EX_CLI, 0, SPEC) {
        h("minimum hits for ngram-map-k speculative decoding (default: ");
        hi(1);
        h(")");
    }
    if add(Opt.SpecNgramMapK4vSizeN, Kind.Int, "--spec-ngram-map-k4v-size-n", "", "N", "", "", EX_SPECULATIVE | EX_SERVER | EX_CLI, 0, SPEC) {
        h("ngram size N for ngram-map-k4v speculative decoding, length of lookup n-gram (default: ");
        hi(12);
        h(")");
    }
    if add(Opt.SpecNgramMapK4vSizeM, Kind.Int, "--spec-ngram-map-k4v-size-m", "", "N", "", "", EX_SPECULATIVE | EX_SERVER | EX_CLI, 0, SPEC) {
        h("ngram size M for ngram-map-k4v speculative decoding, length of draft m-gram (default: ");
        hi(48);
        h(")");
    }
    if add(Opt.SpecNgramMapK4vMinHits, Kind.Int, "--spec-ngram-map-k4v-min-hits", "", "N", "", "", EX_SPECULATIVE | EX_SERVER | EX_CLI, 0, SPEC) {
        h("minimum hits for ngram-map-k4v speculative decoding (default: ");
        hi(1);
        h(")");
    }
    if add(Opt.Draft, Kind.Int, "--draft --draft-n --draft-max", "", "N", "", "LLAMA_ARG_DRAFT_MAX", EX_SPECULATIVE | EX_LOOKUP | EX_SERVER | EX_CLI, 0, SPEC) {
        h("the argument has been removed. use --spec-draft-n-max or --spec-ngram-mod-n-max");
    }
    if add(Opt.DraftMin, Kind.Int, "--draft-min --draft-n-min", "", "N", "", "LLAMA_ARG_DRAFT_MIN", EX_SPECULATIVE | EX_LOOKUP | EX_SERVER | EX_CLI, 0, SPEC) {
        h("the argument has been removed. use --spec-draft-n-min or --spec-ngram-mod-n-min");
    }
    if add(Opt.SpecNgramSizeN, Kind.Int, "--spec-ngram-size-n", "", "N", "", "", EX_SERVER, 0, SPEC) {
        h("the argument has been removed. use the respective --spec-ngram-*-size-n or --spec-ngram-mod-n-match");
    }
    if add(Opt.SpecNgramSizeM, Kind.Int, "--spec-ngram-size-m", "", "N", "", "", EX_SERVER, 0, SPEC) {
        h("the argument has been removed. use the respective --spec-ngram-*-size-m");
    }
    if add(Opt.SpecNgramMinHits, Kind.Int, "--spec-ngram-min-hits", "", "N", "", "", EX_SERVER, 0, SPEC) {
        h("the argument has been removed. use the respective --spec-ngram-*-min-hits");
    }
    if add(Opt.TtsLang, Kind.Str, "--tts-lang", "", "FNAME", "", "", EX_TTS | EX_SERVER, 0, 0) {
        h("language (ISO 639-1) for audio generation\nsee docs/multimodal/tts.md for per-model usage notes");
    }
    if add(Opt.TtsVoice, Kind.Str, "--tts-voice", "", "VOICE", "", "", EX_TTS | EX_SERVER, 0, 0) {
        h("voice for audio generation (model-dependent)\nsee docs/multimodal/tts.md for per-model usage notes");
    }
    if add(Opt.EmbdGemmaDefault, Kind.Void, "--embd-gemma-default", "", "", "", "", EX_EMBEDDING | EX_SERVER, 0, 0) {
        h("use default EmbeddingGemma model (note: can download weights from the internet)");
    }
    if add(Opt.FimQwen15bDefault, Kind.Void, "--fim-qwen-1.5b-default", "", "", "", "", EX_SERVER, 0, 0) {
        h("use default Qwen 2.5 Coder 1.5B (note: can download weights from the internet)");
    }
    if add(Opt.FimQwen3bDefault, Kind.Void, "--fim-qwen-3b-default", "", "", "", "", EX_SERVER, 0, 0) {
        h("use default Qwen 2.5 Coder 3B (note: can download weights from the internet)");
    }
    if add(Opt.FimQwen7bDefault, Kind.Void, "--fim-qwen-7b-default", "", "", "", "", EX_SERVER, 0, 0) {
        h("use default Qwen 2.5 Coder 7B (note: can download weights from the internet)");
    }
    if add(Opt.FimQwen7bSpec, Kind.Void, "--fim-qwen-7b-spec", "", "", "", "", EX_SERVER, 0, 0) {
        h("use Qwen 2.5 Coder 7B + 0.5B draft for speculative decoding (note: can download weights from the internet)");
    }
    if add(Opt.FimQwen14bSpec, Kind.Void, "--fim-qwen-14b-spec", "", "", "", "", EX_SERVER, 0, 0) {
        h("use Qwen 2.5 Coder 14B + 0.5B draft for speculative decoding (note: can download weights from the internet)");
    }
    if add(Opt.FimQwen30bDefault, Kind.Void, "--fim-qwen-30b-default", "", "", "", "", EX_SERVER, 0, 0) {
        h("use default Qwen 3 Coder 30B A3B Instruct (note: can download weights from the internet)");
    }
    if add(Opt.GptOss20bDefault, Kind.Void, "--gpt-oss-20b-default", "", "", "", "", EX_SERVER | EX_CLI, 0, 0) {
        h("use gpt-oss-20b (note: can download weights from the internet)");
    }
    if add(Opt.GptOss120bDefault, Kind.Void, "--gpt-oss-120b-default", "", "", "", "", EX_SERVER | EX_CLI, 0, 0) {
        h("use gpt-oss-120b (note: can download weights from the internet)");
    }
    if add(Opt.VisionGemma4bDefault, Kind.Void, "--vision-gemma-4b-default", "", "", "", "", EX_SERVER | EX_CLI, 0, 0) {
        h("use Gemma 3 4B QAT (note: can download weights from the internet)");
    }
    if add(Opt.VisionGemma12bDefault, Kind.Void, "--vision-gemma-12b-default", "", "", "", "", EX_SERVER | EX_CLI, 0, 0) {
        h("use Gemma 3 12B QAT (note: can download weights from the internet)");
    }
    if add(Opt.SpecDefault, Kind.Void, "--spec-default", "", "", "", "", EX_SERVER | EX_CLI, 0, 0) {
        h("enable default speculative decoding config");
    }
    close();
}
