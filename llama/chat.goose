// llama.cpp's built-in chat templates: a template is recognized by its name
// or by strings in its Jinja source, then written out by hand (no Jinja).
// port-of: src/llama-chat.cpp
import std;

namespace chat;

struct Message { role: const u8[:], content: const u8[:] }

enum Template {
    Unknown, Chatml, Llama2, Llama2Sys, Llama2SysBos, Llama2SysStrip, MistralV1, MistralV3,
    MistralV3Tekken, MistralV7, MistralV7Tekken, Phi3, Phi4, Falcon3, Zephyr, Monarch, Gemma,
    Orion, Openchat, Vicuna, VicunaOrca, Deepseek, Deepseek2, Deepseek3, DeepseekOcr, CommandR,
    Llama3, Chatglm3, Chatglm4, Glmedge, Minicpm, Exaone3, Exaone4, ExaoneMoe, RwkvWorld,
    Granite3X, Granite4_0, Granite4_1, Gigachat, Megrez, Yandex, Bailing, BailingThink,
    Bailing2, Llama4, Smolvlm, Dots1, HunyuanMoe, OpenaiMoe, HunyuanDense, HunyuanVl, KimiK2,
    SeedOss, Grok2, PanguEmbed, SolarOpen,
}

struct Named { name: const u8[:], id: Template }

// The templates llama_chat_builtin_templates lists, sorted by name (a std::map).
let NAMES = [
    Named { name: "bailing", id: Template.Bailing },
    Named { name: "bailing-think", id: Template.BailingThink },
    Named { name: "bailing2", id: Template.Bailing2 },
    Named { name: "chatglm3", id: Template.Chatglm3 },
    Named { name: "chatglm4", id: Template.Chatglm4 },
    Named { name: "chatml", id: Template.Chatml },
    Named { name: "command-r", id: Template.CommandR },
    Named { name: "deepseek", id: Template.Deepseek },
    Named { name: "deepseek-ocr", id: Template.DeepseekOcr },
    Named { name: "deepseek2", id: Template.Deepseek2 },
    Named { name: "deepseek3", id: Template.Deepseek3 },
    Named { name: "exaone-moe", id: Template.ExaoneMoe },
    Named { name: "exaone3", id: Template.Exaone3 },
    Named { name: "exaone4", id: Template.Exaone4 },
    Named { name: "falcon3", id: Template.Falcon3 },
    Named { name: "gemma", id: Template.Gemma },
    Named { name: "gigachat", id: Template.Gigachat },
    Named { name: "glmedge", id: Template.Glmedge },
    Named { name: "gpt-oss", id: Template.OpenaiMoe },
    Named { name: "granite", id: Template.Granite3X },
    Named { name: "granite-4.0", id: Template.Granite4_0 },
    Named { name: "granite-4.1", id: Template.Granite4_1 },
    Named { name: "grok-2", id: Template.Grok2 },
    Named { name: "hunyuan-dense", id: Template.HunyuanDense },
    Named { name: "hunyuan-moe", id: Template.HunyuanMoe },
    Named { name: "hunyuan-vl", id: Template.HunyuanVl },
    Named { name: "kimi-k2", id: Template.KimiK2 },
    Named { name: "llama2", id: Template.Llama2 },
    Named { name: "llama2-sys", id: Template.Llama2Sys },
    Named { name: "llama2-sys-bos", id: Template.Llama2SysBos },
    Named { name: "llama2-sys-strip", id: Template.Llama2SysStrip },
    Named { name: "llama3", id: Template.Llama3 },
    Named { name: "llama4", id: Template.Llama4 },
    Named { name: "megrez", id: Template.Megrez },
    Named { name: "minicpm", id: Template.Minicpm },
    Named { name: "mistral-v1", id: Template.MistralV1 },
    Named { name: "mistral-v3", id: Template.MistralV3 },
    Named { name: "mistral-v3-tekken", id: Template.MistralV3Tekken },
    Named { name: "mistral-v7", id: Template.MistralV7 },
    Named { name: "mistral-v7-tekken", id: Template.MistralV7Tekken },
    Named { name: "monarch", id: Template.Monarch },
    Named { name: "openchat", id: Template.Openchat },
    Named { name: "orion", id: Template.Orion },
    Named { name: "pangu-embedded", id: Template.PanguEmbed },
    Named { name: "phi3", id: Template.Phi3 },
    Named { name: "phi4", id: Template.Phi4 },
    Named { name: "rwkv-world", id: Template.RwkvWorld },
    Named { name: "seed_oss", id: Template.SeedOss },
    Named { name: "smolvlm", id: Template.Smolvlm },
    Named { name: "solar-open", id: Template.SolarOpen },
    Named { name: "vicuna", id: Template.Vicuna },
    Named { name: "vicuna-orca", id: Template.VicunaOrca },
    Named { name: "yandex", id: Template.Yandex },
    Named { name: "zephyr", id: Template.Zephyr },
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
fn detect(tmpl: const u8[:]) -> Template {
    for n in NAMES { if n.name == tmpl { return n.id; } }
    fn has(s: const u8[:]) -> bool { find(tmpl, s) >= 0 }
    if has("<|im_start|>") {
        return if has("<|im_sep|>") { Template.Phi4 } else if has("<end_of_utterance>") { Template.Smolvlm } else { Template.Chatml };
    } else if starts_with(tmpl, "mistral") || has("[INST]") {
        if has("[SYSTEM_PROMPT]") {
            return Template.MistralV7;
        } else if has("' [INST] ' + system_message") || has("[AVAILABLE_TOOLS]") {
            if has(" [INST]") { return Template.MistralV1; }
            if has("\"[INST]\"") { return Template.MistralV3Tekken; }
            return Template.MistralV3;
        } else {
            if has("content.strip()") { return Template.Llama2SysStrip; }
            if has("bos_token + '[INST]") { return Template.Llama2SysBos; }
            if has("<<SYS>>") { return Template.Llama2Sys; }
            return Template.Llama2;
        }
    } else if has("<|assistant|>") && has("<|end|>") {
        return Template.Phi3;
    } else if has("[gMASK]<sop>") {
        return Template.Chatglm4;
    } else if has("<|assistant|>") && has("<|user|>") {
        if has("<|tool_declare|>") { return Template.ExaoneMoe; }
        return if has("</s>") { Template.Falcon3 } else { Template.Glmedge };
    } else if has("<|{{ item['role'] }}|>") && has("<|begin_of_image|>") {
        return Template.Glmedge;
    } else if has("<|user|>") && has("<|endoftext|>") {
        return Template.Zephyr;
    } else if has("bos_token + message['role']") {
        return Template.Monarch;
    } else if has("<start_of_turn>") {
        return Template.Gemma;
    } else if has("'\\n\\nAssistant: ' + eos_token") {
        return Template.Orion;
    } else if has("GPT4 Correct ") {
        return Template.Openchat;
    } else if has("USER: ") && has("ASSISTANT: ") {
        return if has("SYSTEM: ") { Template.VicunaOrca } else { Template.Vicuna };
    } else if has("### Instruction:") && has("<|EOT|>") {
        return Template.Deepseek;
    } else if has("<|START_OF_TURN_TOKEN|>") && has("<|USER_TOKEN|>") {
        return Template.CommandR;
    } else if has("<|start_header_id|>") && has("<|end_header_id|>") {
        return Template.Llama3;
    } else if has("[gMASK]sop") {
        return Template.Chatglm3;
    } else if has("<用户>") {
        return Template.Minicpm;
    } else if has("'Assistant: ' + message['content'] + eos_token") {
        return Template.Deepseek2;
    } else if has("<｜Assistant｜>") && has("<｜User｜>") && has("<｜end▁of▁sentence｜>") {
        return Template.Deepseek3;
    } else if has("[|system|]") && has("[|assistant|]") && has("[|endofturn|]") {
        return if has("[|tool|]") { Template.Exaone4 } else { Template.Exaone3 };
    } else if has("rwkv-world") || has("{{- 'User: ' + message['content']|trim + '\\n\\n' -}}") {
        return Template.RwkvWorld;
    } else if has("<|start_of_role|>") {
        if has("<tool_call>") || has("<tools>") {
            return if has("g4_default_system_message") { Template.Granite4_0 } else { Template.Granite4_1 };
        }
        return Template.Granite3X;
    } else if has("message['role'] + additional_special_tokens[0] + message['content'] + additional_special_tokens[1]") {
        return Template.Gigachat;
    } else if has("<|role_start|>") {
        return Template.Megrez;
    } else if has(" Ассистент:") {
        return Template.Yandex;
    } else if has("<role>ASSISTANT</role>") && has("'HUMAN'") {
        return Template.Bailing;
    } else if has("<role>ASSISTANT</role>") && has("\"HUMAN\"") && has("<think>") {
        return Template.BailingThink;
    } else if has("<role>ASSISTANT</role>") && has("<role>HUMAN</role>") && has("<|role_end|>") {
        return Template.Bailing2;
    } else if has("<|header_start|>") && has("<|header_end|>") {
        return Template.Llama4;
    } else if has("<|endofuserprompt|>") {
        return Template.Dots1;
    } else if has("<|extra_0|>") && has("<|extra_4|>") {
        return Template.HunyuanMoe;
    } else if has("<|start|>") && has("<|channel|>") {
        return Template.OpenaiMoe;
    } else if has("<｜hy_Assistant｜>") && has("<｜hy_begin▁of▁sentence｜>") {
        return Template.HunyuanVl;
    } else if has("<｜hy_Assistant｜>") && has("<｜hy_place▁holder▁no▁3｜>") {
        return Template.HunyuanDense;
    } else if has("<|im_assistant|>assistant<|im_middle|>") {
        return Template.KimiK2;
    } else if has("<seed:bos>") {
        return Template.SeedOss;
    } else if has("'Assistant: '  + message['content'] + '<|separator|>") {
        return Template.Grok2;
    } else if has("[unused9]系统：[unused10]") {
        return Template.PanguEmbed;
    } else if has("<|begin|>") && has("<|end|>") && has("<|content|>") {
        return Template.SolarOpen;
    }
    return Template.Unknown;
}

// llama_chat_apply_template: the chat written out with the template (a
// name or a Jinja source; "chatml" when empty), ending with the assistant's
// turn opened when add_ass. False if the template is not recognized.
fn apply(tmpl: const u8[:], msgs: const Message[:], add_ass: bool, ss: u8[>..]&) -> bool {
    ss.clear();
    let t = detect(if tmpl.len == 0 { "chatml" } else { tmpl });
    match t {
        Chatml => {
            for m in msgs { format(ss, "<|im_start|>", m.role, "\n", m.content, "<|im_end|>\n"); }
            if add_ass { format(ss, "<|im_start|>assistant\n"); }
        },
        MistralV7, MistralV7Tekken => {
            let sp = if t == Template.MistralV7 { " " } else { "" };
            for m in msgs {
                if m.role == "system" { format(ss, "[SYSTEM_PROMPT]", sp, m.content, "[/SYSTEM_PROMPT]"); }
                else if m.role == "user" { format(ss, "[INST]", sp, m.content, "[/INST]"); }
                else { format(ss, sp, m.content, "</s>"); }
            }
        },
        MistralV1, MistralV3, MistralV3Tekken => {
            let leading = if t == Template.MistralV1 { " " } else { "" };
            let trailing = if t == Template.MistralV3Tekken { "" } else { " " };
            var inside = false;
            for m in msgs {
                if !inside { format(ss, leading, "[INST]", trailing); inside = true; }
                if m.role == "system" { format(ss, m.content, "\n\n"); }
                else if m.role == "user" { format(ss, m.content, leading, "[/INST]"); }
                else {
                    format(ss, trailing, if t == Template.MistralV3 { ctrim(m.content) } else { m.content }, "</s>");
                    inside = false;
                }
            }
        },
        Llama2, Llama2Sys, Llama2SysBos, Llama2SysStrip => {
            let system = t != Template.Llama2;
            let bos_inside = t == Template.Llama2SysBos;
            let strip = t == Template.Llama2SysStrip;
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
        Phi3 => {
            for m in msgs { format(ss, "<|", m.role, "|>\n", m.content, "<|end|>\n"); }
            if add_ass { format(ss, "<|assistant|>\n"); }
        },
        Phi4 => {
            for m in msgs { format(ss, "<|im_start|>", m.role, "<|im_sep|>", m.content, "<|im_end|>"); }
            if add_ass { format(ss, "<|im_start|>assistant<|im_sep|>"); }
        },
        Falcon3 => {
            for m in msgs { format(ss, "<|", m.role, "|>\n", m.content, "\n"); }
            if add_ass { format(ss, "<|assistant|>\n"); }
        },
        Zephyr => {
            for m in msgs { format(ss, "<|", m.role, "|>\n", m.content, "<|endoftext|>\n"); }
            if add_ass { format(ss, "<|assistant|>\n"); }
        },
        Monarch => {
            for m, i in msgs { format(ss, if i == 0 { "" } else { "<s>" }, m.role, "\n", m.content, "</s>\n"); }
            if add_ass { format(ss, "<s>assistant\n"); }
        },
        Gemma => {
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
        Orion => {
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
        Openchat => {
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
        Vicuna, VicunaOrca => {
            for m in msgs {
                if m.role == "system" {
                    if t == Template.VicunaOrca { format(ss, "SYSTEM: ", m.content, "\n"); }
                    else { format(ss, m.content, "\n\n"); }
                } else if m.role == "user" {
                    format(ss, "USER: ", m.content, "\n");
                } else if m.role == "assistant" {
                    format(ss, "ASSISTANT: ", m.content, "</s>\n");
                }
            }
            if add_ass { format(ss, "ASSISTANT:"); }
        },
        Deepseek => {
            for m in msgs {
                if m.role == "system" { format(ss, m.content); }
                else if m.role == "user" { format(ss, "### Instruction:\n", m.content, "\n"); }
                else if m.role == "assistant" { format(ss, "### Response:\n", m.content, "\n<|EOT|>\n"); }
            }
            if add_ass { format(ss, "### Response:\n"); }
        },
        CommandR => {
            for m in msgs {
                let tok = if m.role == "system" { "<|SYSTEM_TOKEN|>" } else if m.role == "user" { "<|USER_TOKEN|>" }
                          else if m.role == "assistant" { "<|CHATBOT_TOKEN|>" } else { "" };
                if tok.len > 0 { format(ss, "<|START_OF_TURN_TOKEN|>", tok, ctrim(m.content), "<|END_OF_TURN_TOKEN|>"); }
            }
            if add_ass { format(ss, "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"); }
        },
        Llama3 => {
            for m in msgs { format(ss, "<|start_header_id|>", m.role, "<|end_header_id|>\n\n", ctrim(m.content), "<|eot_id|>"); }
            if add_ass { format(ss, "<|start_header_id|>assistant<|end_header_id|>\n\n"); }
        },
        Chatglm3 => {
            format(ss, "[gMASK]sop");
            for m in msgs { format(ss, "<|", m.role, "|>\n ", m.content); }
            if add_ass { format(ss, "<|assistant|>"); }
        },
        Chatglm4 => {
            format(ss, "[gMASK]<sop>");
            for m in msgs { format(ss, "<|", m.role, "|>\n", m.content); }
            if add_ass { format(ss, "<|assistant|>\n"); }
        },
        Glmedge => {
            for m in msgs { format(ss, "<|", m.role, "|>\n", m.content); }
            if add_ass { format(ss, "<|assistant|>"); }
        },
        Minicpm => {
            for m in msgs {
                if m.role == "user" { format(ss, "<用户>", ctrim(m.content), "<AI>"); }
                else { format(ss, ctrim(m.content)); }
            }
        },
        Deepseek2 => {
            for m in msgs {
                if m.role == "system" { format(ss, m.content, "\n\n"); }
                else if m.role == "user" { format(ss, "User: ", m.content, "\n\n"); }
                else if m.role == "assistant" { format(ss, "Assistant: ", m.content, "<｜end▁of▁sentence｜>"); }
            }
            if add_ass { format(ss, "Assistant:"); }
        },
        Deepseek3 => {
            for m in msgs {
                if m.role == "system" { format(ss, m.content, "\n\n"); }
                else if m.role == "user" { format(ss, "<｜User｜>", m.content); }
                else if m.role == "assistant" { format(ss, "<｜Assistant｜>", m.content, "<｜end▁of▁sentence｜>"); }
            }
            if add_ass { format(ss, "<｜Assistant｜>"); }
        },
        DeepseekOcr => {
            for m in msgs { format(ss, m.content); }
        },
        Exaone3, Exaone4 => {
            for m in msgs {
                if m.role == "system" { format(ss, "[|system|]", ctrim(m.content), "[|endofturn|]\n"); }
                else if m.role == "user" { format(ss, "[|user|]", ctrim(m.content), "\n"); }
                else if m.role == "assistant" { format(ss, "[|assistant|]", ctrim(m.content), "[|endofturn|]\n"); }
                else if m.role == "tool" && t == Template.Exaone4 { format(ss, "[|tool|]", ctrim(m.content), "[|endofturn|]\n"); }
            }
            if add_ass { format(ss, "[|assistant|]"); }
        },
        ExaoneMoe => {
            for m in msgs {
                if m.role == "system" || m.role == "user" || m.role == "assistant" || m.role == "tool" {
                    format(ss, "<|", m.role, "|>\n", ctrim(m.content), "<|endofturn|>\n");
                }
            }
            if add_ass { format(ss, "<|assistant|>\n"); }
        },
        RwkvWorld => {
            for m, i in msgs {
                if m.role == "system" { format(ss, "System: ", ctrim(m.content), "\n\n"); }
                else if m.role == "user" {
                    format(ss, "User: ", ctrim(m.content), "\n\n");
                    if i == msgs.len - 1 { format(ss, "Assistant:"); }
                } else if m.role == "assistant" { format(ss, "Assistant: ", ctrim(m.content), "\n\n"); }
            }
        },
        Granite3X => {
            for m in msgs {
                format(ss, "<|start_of_role|>", m.role, "<|end_of_role|>");
                if m.role == "assistant_tool_call" { format(ss, "<|tool_call|>"); }
                format(ss, m.content, "<|end_of_text|>\n");
            }
            if add_ass { format(ss, "<|start_of_role|>assistant<|end_of_role|>"); }
        },
        Granite4_0, Granite4_1 => {
            for m in msgs {
                if m.role == "assistant_tool_call" { format(ss, "<|start_of_role|>assistant<|end_of_role|><|tool_call|>"); }
                else { format(ss, "<|start_of_role|>", m.role, "<|end_of_role|>"); }
                format(ss, m.content, "<|end_of_text|>\n");
            }
            if add_ass { format(ss, "<|start_of_role|>assistant<|end_of_role|>"); }
        },
        Gigachat => {
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
        Megrez => {
            for m in msgs { format(ss, "<|role_start|>", m.role, "<|role_end|>", m.content, "<|turn_end|>"); }
            if add_ass { format(ss, "<|role_start|>assistant<|role_end|>"); }
        },
        Yandex => {
            for m in msgs {
                if m.role == "user" { format(ss, " Пользователь: ", m.content, "\n\n"); }
                else if m.role == "assistant" { format(ss, " Ассистент: ", m.content, "\n\n"); }
            }
            if add_ass { format(ss, " Ассистент:[SEP]"); }
        },
        Bailing, BailingThink, Bailing2 => {
            if t == Template.Bailing2 && !(msgs.len > 0 && msgs[0].role == "system") {
                format(ss, "<role>SYSTEM</role>detailed thinking off<|role_end|>");
            }
            for m in msgs {
                format(ss, "<role>");
                if m.role == "user" { format(ss, "HUMAN"); }
                else { let at = ss.len; ss.append(m.role); to_upper(ss[at..]); }
                format(ss, "</role>", m.content);
                if t == Template.Bailing2 { format(ss, "<|role_end|>"); }
            }
            if add_ass {
                format(ss, "<role>ASSISTANT</role>");
                if t == Template.BailingThink { format(ss, "<think>"); }
            }
        },
        Llama4 => {
            for m in msgs { format(ss, "<|header_start|>", m.role, "<|header_end|>\n\n", ctrim(m.content), "<|eot|>"); }
            if add_ass { format(ss, "<|header_start|>assistant<|header_end|>\n\n"); }
        },
        Smolvlm => {
            format(ss, "<|im_start|>");
            for m in msgs {
                if m.role == "system" { format(ss, m.content, "\n\n"); }
                else if m.role == "user" { format(ss, "User: ", m.content, "<end_of_utterance>\n"); }
                else { format(ss, "Assistant: ", m.content, "<end_of_utterance>\n"); }
            }
            if add_ass { format(ss, "Assistant:"); }
        },
        Dots1 => {
            for m in msgs {
                if m.role == "system" { format(ss, "<|system|>", m.content, "<|endofsystem|>"); }
                else if m.role == "user" { format(ss, "<|userprompt|>", m.content, "<|endofuserprompt|>"); }
                else { format(ss, "<|response|>", m.content, "<|endofresponse|>"); }
            }
            if add_ass { format(ss, "<|response|>"); }
        },
        HunyuanMoe => {
            for m in msgs {
                if m.role == "system" { format(ss, "<|startoftext|>", m.content, "<|extra_4|>"); }
                else if m.role == "assistant" { format(ss, m.content, "<|eos|>"); }
                else { format(ss, "<|startoftext|>", m.content, "<|extra_0|>"); }
            }
        },
        OpenaiMoe => {
            for m in msgs {
                format(ss, "<|start|>", m.role, "<|message|>", m.content, if m.role == "assistant" { "<|return|>" } else { "<|end|>" });
            }
            if add_ass { format(ss, "<|start|>assistant"); }
        },
        HunyuanDense => {
            for m, i in msgs {
                if i == 0 && m.role == "system" { format(ss, m.content, "<｜hy_place▁holder▁no▁3｜>"); }
                if m.role == "assistant" { format(ss, "<｜hy_Assistant｜>", m.content, "<｜hy_place▁holder▁no▁2｜>"); }
                else if m.role == "user" { format(ss, "<｜hy_User｜>", m.content, "<｜hy_Assistant｜>"); }
            }
        },
        HunyuanVl => {
            format(ss, "<｜hy_begin▁of▁sentence｜>");
            for m, i in msgs {
                if i == 0 && m.role == "system" { format(ss, m.content, "<｜hy_place▁holder▁no▁3｜>"); continue; }
                if m.role == "user" { format(ss, m.content, "<｜hy_User｜>"); }
                else if m.role == "assistant" { format(ss, m.content, "<｜hy_Assistant｜>"); }
            }
        },
        KimiK2 => {
            for m in msgs {
                if m.role == "system" { format(ss, "<|im_system|>system<|im_middle|>"); }
                else if m.role == "user" { format(ss, "<|im_user|>user<|im_middle|>"); }
                else if m.role == "assistant" { format(ss, "<|im_assistant|>assistant<|im_middle|>"); }
                else if m.role == "tool" { format(ss, "<|im_system|>tool<|im_middle|>"); }
                format(ss, m.content, "<|im_end|>");
            }
            if add_ass { format(ss, "<|im_assistant|>assistant<|im_middle|>"); }
        },
        SeedOss => {
            for m in msgs {
                format(ss, "<seed:bos>", m.role, "\n", if m.role == "assistant" { ctrim(m.content) } else { m.content }, "<seed:eos>");
            }
            if add_ass { format(ss, "<seed:bos>assistant\n"); }
        },
        Grok2 => {
            for m in msgs {
                if m.role == "system" { format(ss, "System: ", ctrim(m.content), "<|separator|>\n\n"); }
                else if m.role == "user" { format(ss, "Human: ", ctrim(m.content), "<|separator|>\n\n"); }
                else if m.role == "assistant" { format(ss, "Assistant: ", m.content, "<|separator|>\n\n"); }
            }
            if add_ass { format(ss, "Assistant:"); }
        },
        PanguEmbed => {
            for m, i in msgs {
                if i == 0 && m.role != "system" { format(ss, "[unused9]系统：[unused10]"); }
                let tag = if m.role == "system" { "系统" } else if m.role == "user" { "用户" }
                          else if m.role == "assistant" { "助手" } else if m.role == "tool" { "工具" }
                          else if m.role == "function" { "方法" } else { "" };
                if tag.len > 0 { format(ss, "[unused9]", tag, "：", m.content, "[unused10]"); }
            }
            if add_ass { format(ss, "[unused9]助手："); }
        },
        SolarOpen => {
            for m in msgs { format(ss, "<|begin|>", m.role, "<|content|>", m.content, "<|end|>"); }
            if add_ass { format(ss, "<|begin|>assistant"); }
        },
        _ => { return false; },
    }
    return true;
}
