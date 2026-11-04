// llama.cpp's dedicated chat handlers: the templates the autoparser cannot
// read get a prompt and output parser of their own. detect picks the
// handler from the template's source, in llama.cpp's order.
// port-of: common/parsers/parsers.cpp, common/chat.cpp (common_chat_template_detect_params_init)
import std;
import exception;
import .json_schema;
import .peg;
import .chat_peg;
import .chat_render;

namespace chat_parsers;

let H_MINISTRAL_3 = 0;
let H_LLM_JP_HARMONY = 1;
let H_HARMONY_XML = 2;
let H_GPT_OSS = 3;
let H_MUSE_SPARK = 4;
let H_MUSE_GLIMMER = 5;
let H_FUNCTIONARY_V3_2 = 6;
let H_KIMI_K2 = 7;
let H_KIMI_K3 = 8;
let H_K2_HORIZON = 9;
let H_LING3 = 10;
let H_COHERE2MOE = 11;
let H_LFM2 = 12;
let H_LFM2_5 = 13;
let H_GIGACHAT_V3 = 14;
let H_MINIMAX_M3 = 15;
let H_DEEPSEEK_V3_2 = 16;
let H_GEMMA4_LEGACY = 17;
let H_GEMMA4 = 18;
let H_MINICPM5 = 19;
let H_TRANSLATE_GEMMA = 20;
let H_QWEN3_CODER = 21;

fn handler_name(h: i64) -> const u8[:] {
    return match h {
        0 => "Ministral 3", 1 => "LLM-jp Harmony", 2 => "Harmony XML", 3 => "GPT-OSS", 4 => "Muse Spark", 5 => "Muse Glimmer",
        6 => "Functionary v3.2", 7 => "Kimi K2", 8 => "Kimi K3", 9 => "K2 Horizon", 10 => "Ling 3", 11 => "Cohere2 MoE",
        12 => "LFM2", 13 => "LFM2.5", 14 => "GigaChat v3", 15 => "MiniMax M3", 16 => "DeepSeek V3.2", 17 => "Gemma 4 (legacy)",
        18 => "Gemma 4", 19 => "MiniCPM5", 20 => "TranslateGemma", 21 => "Qwen3-Coder", _ => "?",
    };
}

// is_lfm2_template
fn is_lfm2(src: const u8[:]) -> bool { find(src, "<|tool_list_start|>") >= 0 && find(src, "<|tool_list_end|>") >= 0 }

// common_chat_template_detect_params_init: the handler, or -1 for the autoparser.
fn detect(src: const u8[:]) -> i64 {
    fn has(s: const u8[:]) -> bool { find(src, s) >= 0 }
    if has("[SYSTEM_PROMPT]") && has("[TOOL_CALLS]") && has("[ARGS]") && !has("[CALL_ID]") { return H_MINISTRAL_3; }
    if has("chat_format=llm-jp-harmony-v1") { return H_LLM_JP_HARMONY; }
    if has("<|channel|>") && has("<|constrain|>xml") && has("rfl:parameter") { return H_HARMONY_XML; }
    if has("<|channel|>") { return H_GPT_OSS; }
    if has("to=commentary") && has("<|eom|>") { return H_MUSE_SPARK; }
    if has("<atem:function_calls>") && has("<|eom|>") { return H_MUSE_GLIMMER; }
    if has(">>>all") && has(">>>${recipient}") { return H_FUNCTIONARY_V3_2; }
    if has("<|tool_calls_section_begin|>") && has("<|tool_call_begin|>") { return H_KIMI_K2; }
    if has("<|open|>") && has("<|close|>") && has("<|end_of_msg|>") { return H_KIMI_K3; }
    if has("<|ifm|im_start|>") && has("<ifm|tool_calls>") { return H_K2_HORIZON; }
    if has("<role>ASSISTANT</role>") && has("<arg_key>") { return H_LING3; }
    if has("<|START_TEXT|>") && has("<|START_ACTION|>") { return H_COHERE2MOE; }
    if is_lfm2(src) { return H_LFM2; }
    if has("List of tools: [") && !has("<|tool_list_start|>") { return H_LFM2_5; }
    if has("<|role_sep|>") && has("<|message_sep|>") && !has("<|function_call|>") { return H_GIGACHAT_V3; }
    if has("]<]minimax[>[") && has("<tool_call>") && has("<invoke name=") { return H_MINIMAX_M3; }
    if (has("dsml_token") && has("DSML") && (has("function_calls") || has("tool_calls"))) || has("｜DSML｜ calls>") { return H_DEEPSEEK_V3_2; }
    if has("'<|tool_call>call:'") {
        if !has("{#- OpenAI Chat Completions:") { return H_GEMMA4_LEGACY; }
        return H_GEMMA4;
    }
    if has("Tool usage guidelines:") && has("<function name=\"") && has("<param name=\"") { return H_MINICPM5; }
    if has("[source_lang_code]") && has("[target_lang_code]") { return H_TRANSLATE_GEMMA; }
    if has("<tool_call>") && has("<function=") && has("<parameter=") && !has("'<tool_call><function=' ~ tool_call.name ~ '>'") { return H_QWEN3_CODER; }
    return -1;
}

var err: u8[>..] = [];

// The handler's params for template t with the inputs g, into chat_render;
// false, and the message in err, if it fails.
fn try_init(t: chat_render::Template, g: chat_render::Gen) -> bool {
    err.clear();
    err.append(str("the dedicated handler ", handler_name(t.handler), " is not ported yet"));
    return false;
}

// The mappers of the formats with their own.
fn map_gemma4() { chat_peg::map_parse(); }
fn map_minimax_m3() { chat_peg::map_parse(); }
