// llama.cpp's built-in chat templates: a template is recognized by its name
// or by strings in its Jinja source, then written out by hand (no Jinja).
// port-of: src/llama-chat.cpp
import std;

namespace chat;

struct Message { role: const u8[:], content: const u8[:] }

let UNKNOWN = 0;
let CHATML = 1;
let LLAMA_2 = 2;
let LLAMA_2_SYS = 3;
let LLAMA_2_SYS_BOS = 4;
let LLAMA_2_SYS_STRIP = 5;
let MISTRAL_V1 = 6;
let MISTRAL_V3 = 7;
let MISTRAL_V3_TEKKEN = 8;
let MISTRAL_V7 = 9;
let MISTRAL_V7_TEKKEN = 10;
let PHI_3 = 11;
let PHI_4 = 12;
let FALCON_3 = 13;
let ZEPHYR = 14;
let MONARCH = 15;
let GEMMA = 16;
let ORION = 17;
let OPENCHAT = 18;
let VICUNA = 19;
let VICUNA_ORCA = 20;
let DEEPSEEK = 21;
let DEEPSEEK_2 = 22;
let DEEPSEEK_3 = 23;
let DEEPSEEK_OCR = 24;
let COMMAND_R = 25;
let LLAMA_3 = 26;
let CHATGLM_3 = 27;
let CHATGLM_4 = 28;
let GLMEDGE = 29;
let MINICPM = 30;
let EXAONE_3 = 31;
let EXAONE_4 = 32;
let EXAONE_MOE = 33;
let RWKV_WORLD = 34;
let GRANITE_3_X = 35;
let GRANITE_4_0 = 36;
let GRANITE_4_1 = 37;
let GIGACHAT = 38;
let MEGREZ = 39;
let YANDEX = 40;
let BAILING = 41;
let BAILING_THINK = 42;
let BAILING2 = 43;
let LLAMA4 = 44;
let SMOLVLM = 45;
let DOTS1 = 46;
let HUNYUAN_MOE = 47;
let OPENAI_MOE = 48;
let HUNYUAN_DENSE = 49;
let HUNYUAN_VL = 50;
let KIMI_K2 = 51;
let SEED_OSS = 52;
let GROK_2 = 53;
let PANGU_EMBED = 54;
let SOLAR_OPEN = 55;

struct Named { name: const u8[:], id: i64 }

// The templates llama_chat_builtin_templates lists, sorted by name (a std::map).
let NAMES = [
    Named { name: "bailing", id: BAILING },
    Named { name: "bailing-think", id: BAILING_THINK },
    Named { name: "bailing2", id: BAILING2 },
    Named { name: "chatglm3", id: CHATGLM_3 },
    Named { name: "chatglm4", id: CHATGLM_4 },
    Named { name: "chatml", id: CHATML },
    Named { name: "command-r", id: COMMAND_R },
    Named { name: "deepseek", id: DEEPSEEK },
    Named { name: "deepseek-ocr", id: DEEPSEEK_OCR },
    Named { name: "deepseek2", id: DEEPSEEK_2 },
    Named { name: "deepseek3", id: DEEPSEEK_3 },
    Named { name: "exaone-moe", id: EXAONE_MOE },
    Named { name: "exaone3", id: EXAONE_3 },
    Named { name: "exaone4", id: EXAONE_4 },
    Named { name: "falcon3", id: FALCON_3 },
    Named { name: "gemma", id: GEMMA },
    Named { name: "gigachat", id: GIGACHAT },
    Named { name: "glmedge", id: GLMEDGE },
    Named { name: "gpt-oss", id: OPENAI_MOE },
    Named { name: "granite", id: GRANITE_3_X },
    Named { name: "granite-4.0", id: GRANITE_4_0 },
    Named { name: "granite-4.1", id: GRANITE_4_1 },
    Named { name: "grok-2", id: GROK_2 },
    Named { name: "hunyuan-dense", id: HUNYUAN_DENSE },
    Named { name: "hunyuan-moe", id: HUNYUAN_MOE },
    Named { name: "hunyuan-vl", id: HUNYUAN_VL },
    Named { name: "kimi-k2", id: KIMI_K2 },
    Named { name: "llama2", id: LLAMA_2 },
    Named { name: "llama2-sys", id: LLAMA_2_SYS },
    Named { name: "llama2-sys-bos", id: LLAMA_2_SYS_BOS },
    Named { name: "llama2-sys-strip", id: LLAMA_2_SYS_STRIP },
    Named { name: "llama3", id: LLAMA_3 },
    Named { name: "llama4", id: LLAMA4 },
    Named { name: "megrez", id: MEGREZ },
    Named { name: "minicpm", id: MINICPM },
    Named { name: "mistral-v1", id: MISTRAL_V1 },
    Named { name: "mistral-v3", id: MISTRAL_V3 },
    Named { name: "mistral-v3-tekken", id: MISTRAL_V3_TEKKEN },
    Named { name: "mistral-v7", id: MISTRAL_V7 },
    Named { name: "mistral-v7-tekken", id: MISTRAL_V7_TEKKEN },
    Named { name: "monarch", id: MONARCH },
    Named { name: "openchat", id: OPENCHAT },
    Named { name: "orion", id: ORION },
    Named { name: "pangu-embedded", id: PANGU_EMBED },
    Named { name: "phi3", id: PHI_3 },
    Named { name: "phi4", id: PHI_4 },
    Named { name: "rwkv-world", id: RWKV_WORLD },
    Named { name: "seed_oss", id: SEED_OSS },
    Named { name: "smolvlm", id: SMOLVLM },
    Named { name: "solar-open", id: SOLAR_OPEN },
    Named { name: "vicuna", id: VICUNA },
    Named { name: "vicuna-orca", id: VICUNA_ORCA },
    Named { name: "yandex", id: YANDEX },
    Named { name: "zephyr", id: ZEPHYR },
];

// C's isspace, which trim uses.
fn c_space(c: u8) -> bool { c == ' ' || (c >= '\t' && c <= '\r') }

fn ctrim(s: const u8[:]) -> const u8[:] {
    var a = 0;
    var b = s.len;
    while a < b && c_space(s[a]) { a += 1; }
    while b > a && c_space(s[b - 1]) { b -= 1; }
    return s[a..b];
}

// llm_chat_detect_template: a name, or a Jinja template recognized by its strings.
fn detect(tmpl: const u8[:]) -> i64 {
    for n in NAMES { if n.name == tmpl { return n.id; } }
    fn has(s: const u8[:]) -> bool { find(tmpl, s) >= 0 }
    if has("<|im_start|>") {
        return if has("<|im_sep|>") { PHI_4 } else if has("<end_of_utterance>") { SMOLVLM } else { CHATML };
    } else if starts_with(tmpl, "mistral") || has("[INST]") {
        if has("[SYSTEM_PROMPT]") {
            return MISTRAL_V7;
        } else if has("' [INST] ' + system_message") || has("[AVAILABLE_TOOLS]") {
            if has(" [INST]") { return MISTRAL_V1; }
            if has("\"[INST]\"") { return MISTRAL_V3_TEKKEN; }
            return MISTRAL_V3;
        } else {
            if has("content.strip()") { return LLAMA_2_SYS_STRIP; }
            if has("bos_token + '[INST]") { return LLAMA_2_SYS_BOS; }
            if has("<<SYS>>") { return LLAMA_2_SYS; }
            return LLAMA_2;
        }
    } else if has("<|assistant|>") && has("<|end|>") {
        return PHI_3;
    } else if has("[gMASK]<sop>") {
        return CHATGLM_4;
    } else if has("<|assistant|>") && has("<|user|>") {
        if has("<|tool_declare|>") { return EXAONE_MOE; }
        return if has("</s>") { FALCON_3 } else { GLMEDGE };
    } else if has("<|{{ item['role'] }}|>") && has("<|begin_of_image|>") {
        return GLMEDGE;
    } else if has("<|user|>") && has("<|endoftext|>") {
        return ZEPHYR;
    } else if has("bos_token + message['role']") {
        return MONARCH;
    } else if has("<start_of_turn>") {
        return GEMMA;
    } else if has("'\\n\\nAssistant: ' + eos_token") {
        return ORION;
    } else if has("GPT4 Correct ") {
        return OPENCHAT;
    } else if has("USER: ") && has("ASSISTANT: ") {
        return if has("SYSTEM: ") { VICUNA_ORCA } else { VICUNA };
    } else if has("### Instruction:") && has("<|EOT|>") {
        return DEEPSEEK;
    } else if has("<|START_OF_TURN_TOKEN|>") && has("<|USER_TOKEN|>") {
        return COMMAND_R;
    } else if has("<|start_header_id|>") && has("<|end_header_id|>") {
        return LLAMA_3;
    } else if has("[gMASK]sop") {
        return CHATGLM_3;
    } else if has("<用户>") {
        return MINICPM;
    } else if has("'Assistant: ' + message['content'] + eos_token") {
        return DEEPSEEK_2;
    } else if has("<｜Assistant｜>") && has("<｜User｜>") && has("<｜end▁of▁sentence｜>") {
        return DEEPSEEK_3;
    } else if has("[|system|]") && has("[|assistant|]") && has("[|endofturn|]") {
        return if has("[|tool|]") { EXAONE_4 } else { EXAONE_3 };
    } else if has("rwkv-world") || has("{{- 'User: ' + message['content']|trim + '\\n\\n' -}}") {
        return RWKV_WORLD;
    } else if has("<|start_of_role|>") {
        if has("<tool_call>") || has("<tools>") {
            return if has("g4_default_system_message") { GRANITE_4_0 } else { GRANITE_4_1 };
        }
        return GRANITE_3_X;
    } else if has("message['role'] + additional_special_tokens[0] + message['content'] + additional_special_tokens[1]") {
        return GIGACHAT;
    } else if has("<|role_start|>") {
        return MEGREZ;
    } else if has(" Ассистент:") {
        return YANDEX;
    } else if has("<role>ASSISTANT</role>") && has("'HUMAN'") {
        return BAILING;
    } else if has("<role>ASSISTANT</role>") && has("\"HUMAN\"") && has("<think>") {
        return BAILING_THINK;
    } else if has("<role>ASSISTANT</role>") && has("<role>HUMAN</role>") && has("<|role_end|>") {
        return BAILING2;
    } else if has("<|header_start|>") && has("<|header_end|>") {
        return LLAMA4;
    } else if has("<|endofuserprompt|>") {
        return DOTS1;
    } else if has("<|extra_0|>") && has("<|extra_4|>") {
        return HUNYUAN_MOE;
    } else if has("<|start|>") && has("<|channel|>") {
        return OPENAI_MOE;
    } else if has("<｜hy_Assistant｜>") && has("<｜hy_begin▁of▁sentence｜>") {
        return HUNYUAN_VL;
    } else if has("<｜hy_Assistant｜>") && has("<｜hy_place▁holder▁no▁3｜>") {
        return HUNYUAN_DENSE;
    } else if has("<|im_assistant|>assistant<|im_middle|>") {
        return KIMI_K2;
    } else if has("<seed:bos>") {
        return SEED_OSS;
    } else if has("'Assistant: '  + message['content'] + '<|separator|>") {
        return GROK_2;
    } else if has("[unused9]系统：[unused10]") {
        return PANGU_EMBED;
    } else if has("<|begin|>") && has("<|end|>") && has("<|content|>") {
        return SOLAR_OPEN;
    }
    return UNKNOWN;
}

// llama_chat_apply_template: the chat written out with the template (a
// name or a Jinja source; "chatml" when empty), ending with the assistant's
// turn opened when add_ass. False if the template is not recognized.
fn apply(tmpl: const u8[:], msgs: const Message[:], add_ass: bool, ss: u8[>..]&) -> bool {
    ss.clear();
    let t = detect(if tmpl.len == 0 { "chatml" } else { tmpl });
    match t {
        CHATML => {
            for m in msgs { format(ss, "<|im_start|>", m.role, "\n", m.content, "<|im_end|>\n"); }
            if add_ass { format(ss, "<|im_start|>assistant\n"); }
        },
        MISTRAL_V7, MISTRAL_V7_TEKKEN => {
            let sp = if t == MISTRAL_V7 { " " } else { "" };
            for m in msgs {
                if m.role == "system" { format(ss, "[SYSTEM_PROMPT]", sp, m.content, "[/SYSTEM_PROMPT]"); }
                else if m.role == "user" { format(ss, "[INST]", sp, m.content, "[/INST]"); }
                else { format(ss, sp, m.content, "</s>"); }
            }
        },
        MISTRAL_V1, MISTRAL_V3, MISTRAL_V3_TEKKEN => {
            let leading = if t == MISTRAL_V1 { " " } else { "" };
            let trailing = if t == MISTRAL_V3_TEKKEN { "" } else { " " };
            var inside = false;
            for m in msgs {
                if !inside { format(ss, leading, "[INST]", trailing); inside = true; }
                if m.role == "system" { format(ss, m.content, "\n\n"); }
                else if m.role == "user" { format(ss, m.content, leading, "[/INST]"); }
                else {
                    format(ss, trailing, if t == MISTRAL_V3 { ctrim(m.content) } else { m.content }, "</s>");
                    inside = false;
                }
            }
        },
        LLAMA_2, LLAMA_2_SYS, LLAMA_2_SYS_BOS, LLAMA_2_SYS_STRIP => {
            let system = t != LLAMA_2;
            let bos_inside = t == LLAMA_2_SYS_BOS;
            let strip = t == LLAMA_2_SYS_STRIP;
            var inside = true;      // no BOS at the beginning
            format(ss, "[INST] ");
            for m in msgs {
                let content = if strip { ctrim(m.content) } else { m.content };
                if !inside {
                    inside = true;
                    format(ss, if bos_inside { "<s>[INST] " } else { "[INST] " });
                }
                if m.role == "system" {
                    if system { format(ss, "<<SYS>>\n", content, "\n<</SYS>>\n\n"); }
                    else { format(ss, content, "\n"); }
                } else if m.role == "user" {
                    format(ss, content, " [/INST]");
                } else {
                    format(ss, content, "</s>");
                    inside = false;
                }
            }
        },
        PHI_3 => {
            for m in msgs { format(ss, "<|", m.role, "|>\n", m.content, "<|end|>\n"); }
            if add_ass { format(ss, "<|assistant|>\n"); }
        },
        PHI_4 => {
            for m in msgs { format(ss, "<|im_start|>", m.role, "<|im_sep|>", m.content, "<|im_end|>"); }
            if add_ass { format(ss, "<|im_start|>assistant<|im_sep|>"); }
        },
        FALCON_3 => {
            for m in msgs { format(ss, "<|", m.role, "|>\n", m.content, "\n"); }
            if add_ass { format(ss, "<|assistant|>\n"); }
        },
        ZEPHYR => {
            for m in msgs { format(ss, "<|", m.role, "|>\n", m.content, "<|endoftext|>\n"); }
            if add_ass { format(ss, "<|assistant|>\n"); }
        },
        MONARCH => {
            for m, i in msgs { format(ss, if i == 0 { "" } else { "<s>" }, m.role, "\n", m.content, "</s>\n"); }
            if add_ass { format(ss, "<s>assistant\n"); }
        },
        GEMMA => {
            var system: u8[>..] = [];
            for m in msgs {
                if m.role == "system" { system.append(ctrim(m.content)); continue; }
                let role = if m.role == "assistant" { "model" } else { m.role };
                format(ss, "<start_of_turn>", role, "\n");
                if system.len > 0 && role != "model" { format(ss, system, "\n\n"); system.clear(); }
                format(ss, ctrim(m.content), "<end_of_turn>\n");
            }
            if add_ass { format(ss, "<start_of_turn>model\n"); }
        },
        ORION => {
            var system: u8[>..] = [];
            for m in msgs {
                if m.role == "system" { system.append(m.content); continue; }
                if m.role == "user" {
                    format(ss, "Human: ");
                    if system.len > 0 { format(ss, system, "\n\n"); system.clear(); }
                    format(ss, m.content, "\n\nAssistant: </s>");
                } else {
                    format(ss, m.content, "</s>");
                }
            }
        },
        OPENCHAT => {
            for m in msgs {
                if m.role == "system" { format(ss, m.content, "<|end_of_turn|>"); continue; }
                format(ss, "GPT4 Correct ");
                if m.role.len > 0 {
                    let c = m.role[0];
                    ss.push(if c >= 'a' && c <= 'z' { c - 32 } else { c });
                    ss.append(m.role[1..]);
                }
                format(ss, ": ", m.content, "<|end_of_turn|>");
            }
            if add_ass { format(ss, "GPT4 Correct Assistant:"); }
        },
        VICUNA, VICUNA_ORCA => {
            for m in msgs {
                if m.role == "system" {
                    if t == VICUNA_ORCA { format(ss, "SYSTEM: ", m.content, "\n"); }
                    else { format(ss, m.content, "\n\n"); }
                } else if m.role == "user" {
                    format(ss, "USER: ", m.content, "\n");
                } else if m.role == "assistant" {
                    format(ss, "ASSISTANT: ", m.content, "</s>\n");
                }
            }
            if add_ass { format(ss, "ASSISTANT:"); }
        },
        DEEPSEEK => {
            for m in msgs {
                if m.role == "system" { format(ss, m.content); }
                else if m.role == "user" { format(ss, "### Instruction:\n", m.content, "\n"); }
                else if m.role == "assistant" { format(ss, "### Response:\n", m.content, "\n<|EOT|>\n"); }
            }
            if add_ass { format(ss, "### Response:\n"); }
        },
        COMMAND_R => {
            for m in msgs {
                let tok = if m.role == "system" { "<|SYSTEM_TOKEN|>" } else if m.role == "user" { "<|USER_TOKEN|>" }
                          else if m.role == "assistant" { "<|CHATBOT_TOKEN|>" } else { "" };
                if tok.len > 0 { format(ss, "<|START_OF_TURN_TOKEN|>", tok, ctrim(m.content), "<|END_OF_TURN_TOKEN|>"); }
            }
            if add_ass { format(ss, "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"); }
        },
        LLAMA_3 => {
            for m in msgs { format(ss, "<|start_header_id|>", m.role, "<|end_header_id|>\n\n", ctrim(m.content), "<|eot_id|>"); }
            if add_ass { format(ss, "<|start_header_id|>assistant<|end_header_id|>\n\n"); }
        },
        CHATGLM_3 => {
            format(ss, "[gMASK]sop");
            for m in msgs { format(ss, "<|", m.role, "|>\n ", m.content); }
            if add_ass { format(ss, "<|assistant|>"); }
        },
        CHATGLM_4 => {
            format(ss, "[gMASK]<sop>");
            for m in msgs { format(ss, "<|", m.role, "|>\n", m.content); }
            if add_ass { format(ss, "<|assistant|>\n"); }
        },
        GLMEDGE => {
            for m in msgs { format(ss, "<|", m.role, "|>\n", m.content); }
            if add_ass { format(ss, "<|assistant|>"); }
        },
        MINICPM => {
            for m in msgs {
                if m.role == "user" { format(ss, "<用户>", ctrim(m.content), "<AI>"); }
                else { format(ss, ctrim(m.content)); }
            }
        },
        DEEPSEEK_2 => {
            for m in msgs {
                if m.role == "system" { format(ss, m.content, "\n\n"); }
                else if m.role == "user" { format(ss, "User: ", m.content, "\n\n"); }
                else if m.role == "assistant" { format(ss, "Assistant: ", m.content, "<｜end▁of▁sentence｜>"); }
            }
            if add_ass { format(ss, "Assistant:"); }
        },
        DEEPSEEK_3 => {
            for m in msgs {
                if m.role == "system" { format(ss, m.content, "\n\n"); }
                else if m.role == "user" { format(ss, "<｜User｜>", m.content); }
                else if m.role == "assistant" { format(ss, "<｜Assistant｜>", m.content, "<｜end▁of▁sentence｜>"); }
            }
            if add_ass { format(ss, "<｜Assistant｜>"); }
        },
        DEEPSEEK_OCR => {
            for m in msgs { format(ss, m.content); }
        },
        EXAONE_3, EXAONE_4 => {
            for m in msgs {
                if m.role == "system" { format(ss, "[|system|]", ctrim(m.content), "[|endofturn|]\n"); }
                else if m.role == "user" { format(ss, "[|user|]", ctrim(m.content), "\n"); }
                else if m.role == "assistant" { format(ss, "[|assistant|]", ctrim(m.content), "[|endofturn|]\n"); }
                else if m.role == "tool" && t == EXAONE_4 { format(ss, "[|tool|]", ctrim(m.content), "[|endofturn|]\n"); }
            }
            if add_ass { format(ss, "[|assistant|]"); }
        },
        EXAONE_MOE => {
            for m in msgs {
                if m.role == "system" || m.role == "user" || m.role == "assistant" || m.role == "tool" {
                    format(ss, "<|", m.role, "|>\n", ctrim(m.content), "<|endofturn|>\n");
                }
            }
            if add_ass { format(ss, "<|assistant|>\n"); }
        },
        RWKV_WORLD => {
            for m, i in msgs {
                if m.role == "system" { format(ss, "System: ", ctrim(m.content), "\n\n"); }
                else if m.role == "user" {
                    format(ss, "User: ", ctrim(m.content), "\n\n");
                    if i == msgs.len - 1 { format(ss, "Assistant:"); }
                } else if m.role == "assistant" { format(ss, "Assistant: ", ctrim(m.content), "\n\n"); }
            }
        },
        GRANITE_3_X => {
            for m in msgs {
                format(ss, "<|start_of_role|>", m.role, "<|end_of_role|>");
                if m.role == "assistant_tool_call" { format(ss, "<|tool_call|>"); }
                format(ss, m.content, "<|end_of_text|>\n");
            }
            if add_ass { format(ss, "<|start_of_role|>assistant<|end_of_role|>"); }
        },
        GRANITE_4_0, GRANITE_4_1 => {
            for m in msgs {
                if m.role == "assistant_tool_call" { format(ss, "<|start_of_role|>assistant<|end_of_role|><|tool_call|>"); }
                else { format(ss, "<|start_of_role|>", m.role, "<|end_of_role|>"); }
                format(ss, m.content, "<|end_of_text|>\n");
            }
            if add_ass { format(ss, "<|start_of_role|>assistant<|end_of_role|>"); }
        },
        GIGACHAT => {
            let has_system = msgs.len > 0 && msgs[0].role == "system";
            if has_system { format(ss, "<s>", msgs[0].content, "<|message_sep|>"); } else { format(ss, "<s>"); }
            for i in (if has_system { 1 } else { 0 })..msgs.len {
                let m = msgs[i];
                if m.role == "user" {
                    format(ss, "user<|role_sep|>", m.content, "<|message_sep|>available functions<|role_sep|>[]<|message_sep|>");
                } else if m.role == "assistant" {
                    format(ss, "assistant<|role_sep|>", m.content, "<|message_sep|>");
                }
            }
            if add_ass { format(ss, "assistant<|role_sep|>"); }
        },
        MEGREZ => {
            for m in msgs { format(ss, "<|role_start|>", m.role, "<|role_end|>", m.content, "<|turn_end|>"); }
            if add_ass { format(ss, "<|role_start|>assistant<|role_end|>"); }
        },
        YANDEX => {
            for m in msgs {
                if m.role == "user" { format(ss, " Пользователь: ", m.content, "\n\n"); }
                else if m.role == "assistant" { format(ss, " Ассистент: ", m.content, "\n\n"); }
            }
            if add_ass { format(ss, " Ассистент:[SEP]"); }
        },
        BAILING, BAILING_THINK, BAILING2 => {
            if t == BAILING2 && !(msgs.len > 0 && msgs[0].role == "system") {
                format(ss, "<role>SYSTEM</role>detailed thinking off<|role_end|>");
            }
            for m in msgs {
                format(ss, "<role>");
                if m.role == "user" { format(ss, "HUMAN"); }
                else { let at = ss.len; ss.append(m.role); to_upper(ss[at..]); }
                format(ss, "</role>", m.content);
                if t == BAILING2 { format(ss, "<|role_end|>"); }
            }
            if add_ass {
                format(ss, "<role>ASSISTANT</role>");
                if t == BAILING_THINK { format(ss, "<think>"); }
            }
        },
        LLAMA4 => {
            for m in msgs { format(ss, "<|header_start|>", m.role, "<|header_end|>\n\n", ctrim(m.content), "<|eot|>"); }
            if add_ass { format(ss, "<|header_start|>assistant<|header_end|>\n\n"); }
        },
        SMOLVLM => {
            format(ss, "<|im_start|>");
            for m in msgs {
                if m.role == "system" { format(ss, m.content, "\n\n"); }
                else if m.role == "user" { format(ss, "User: ", m.content, "<end_of_utterance>\n"); }
                else { format(ss, "Assistant: ", m.content, "<end_of_utterance>\n"); }
            }
            if add_ass { format(ss, "Assistant:"); }
        },
        DOTS1 => {
            for m in msgs {
                if m.role == "system" { format(ss, "<|system|>", m.content, "<|endofsystem|>"); }
                else if m.role == "user" { format(ss, "<|userprompt|>", m.content, "<|endofuserprompt|>"); }
                else { format(ss, "<|response|>", m.content, "<|endofresponse|>"); }
            }
            if add_ass { format(ss, "<|response|>"); }
        },
        HUNYUAN_MOE => {
            for m in msgs {
                if m.role == "system" { format(ss, "<|startoftext|>", m.content, "<|extra_4|>"); }
                else if m.role == "assistant" { format(ss, m.content, "<|eos|>"); }
                else { format(ss, "<|startoftext|>", m.content, "<|extra_0|>"); }
            }
        },
        OPENAI_MOE => {
            for m in msgs {
                format(ss, "<|start|>", m.role, "<|message|>", m.content, if m.role == "assistant" { "<|return|>" } else { "<|end|>" });
            }
            if add_ass { format(ss, "<|start|>assistant"); }
        },
        HUNYUAN_DENSE => {
            for m, i in msgs {
                if i == 0 && m.role == "system" { format(ss, m.content, "<｜hy_place▁holder▁no▁3｜>"); }
                if m.role == "assistant" { format(ss, "<｜hy_Assistant｜>", m.content, "<｜hy_place▁holder▁no▁2｜>"); }
                else if m.role == "user" { format(ss, "<｜hy_User｜>", m.content, "<｜hy_Assistant｜>"); }
            }
        },
        HUNYUAN_VL => {
            format(ss, "<｜hy_begin▁of▁sentence｜>");
            for m, i in msgs {
                if i == 0 && m.role == "system" { format(ss, m.content, "<｜hy_place▁holder▁no▁3｜>"); continue; }
                if m.role == "user" { format(ss, m.content, "<｜hy_User｜>"); }
                else if m.role == "assistant" { format(ss, m.content, "<｜hy_Assistant｜>"); }
            }
        },
        KIMI_K2 => {
            for m in msgs {
                if m.role == "system" { format(ss, "<|im_system|>system<|im_middle|>"); }
                else if m.role == "user" { format(ss, "<|im_user|>user<|im_middle|>"); }
                else if m.role == "assistant" { format(ss, "<|im_assistant|>assistant<|im_middle|>"); }
                else if m.role == "tool" { format(ss, "<|im_system|>tool<|im_middle|>"); }
                format(ss, m.content, "<|im_end|>");
            }
            if add_ass { format(ss, "<|im_assistant|>assistant<|im_middle|>"); }
        },
        SEED_OSS => {
            for m in msgs {
                format(ss, "<seed:bos>", m.role, "\n", if m.role == "assistant" { ctrim(m.content) } else { m.content }, "<seed:eos>");
            }
            if add_ass { format(ss, "<seed:bos>assistant\n"); }
        },
        GROK_2 => {
            for m in msgs {
                if m.role == "system" { format(ss, "System: ", ctrim(m.content), "<|separator|>\n\n"); }
                else if m.role == "user" { format(ss, "Human: ", ctrim(m.content), "<|separator|>\n\n"); }
                else if m.role == "assistant" { format(ss, "Assistant: ", m.content, "<|separator|>\n\n"); }
            }
            if add_ass { format(ss, "Assistant:"); }
        },
        PANGU_EMBED => {
            for m, i in msgs {
                if i == 0 && m.role != "system" { format(ss, "[unused9]系统：[unused10]"); }
                let tag = if m.role == "system" { "系统" } else if m.role == "user" { "用户" }
                          else if m.role == "assistant" { "助手" } else if m.role == "tool" { "工具" }
                          else if m.role == "function" { "方法" } else { "" };
                if tag.len > 0 { format(ss, "[unused9]", tag, "：", m.content, "[unused10]"); }
            }
            if add_ass { format(ss, "[unused9]助手："); }
        },
        SOLAR_OPEN => {
            for m in msgs { format(ss, "<|begin|>", m.role, "<|content|>", m.content, "<|end|>"); }
            if add_ass { format(ss, "<|begin|>assistant"); }
        },
        _ => { return false; },
    }
    return true;
}
