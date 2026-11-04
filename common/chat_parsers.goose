// llama.cpp's dedicated chat handlers: the templates the autoparser cannot
// read get a prompt and output parser of their own. detect picks the
// handler from the template's source, in llama.cpp's order.
// port-of: common/parsers/parsers.cpp, common/chat.cpp (common_chat_template_detect_params_init)
import std;
import exception;
import os;
import .jinja;
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

// The mappers of the formats with their own.
fn map_minimax_m3() { chat_peg::map_parse(); }

//
// what the handlers share
//

var tools_v = -1;
var schema_v = -1;
var src_has_think = false;

fn has_tools() -> bool { tools_v >= 0 && json_schema::json_kind(tools_v) == json_schema::J_ARRAY && json_schema::json_len(tools_v) > 0 }
fn has_response_format() -> bool { schema_v >= 0 && json_schema::json_kind(schema_v) == json_schema::J_OBJECT && json_schema::json_len(schema_v) > 0 }

// The prompt (direct_apply_impl with the overrides, into chat_render's
// prompt) and the generation prompt (generation_prompt_impl, into gp). It
// throws json_schema::Fail.
fn render(t: chat_render::Template, g: chat_render::Gen, ov: chat_render::Override, gp: u8[>..]&) {
    var ok = false;
    fn body() {
        chat_render::generation_prompt(t, g, ov, gp);
        chat_render::direct_apply(t, g, ov);
        ok = true;
    }
    exception::try<jinja::Fail>(body);
    if !ok { json_schema::fail(jinja::err); }
}

// The grammar of the parser built, lazy or not.
fn build_grammar(lazy: bool) {
    chat_render::grammar_lazy = lazy;
    json_schema::begin_grammar();
    peg::build_grammar(chat_render::parser, lazy);
    json_schema::end_grammar(chat_render::grammar);
}

fn preserve(s: const u8[:]) { chat_render::list_push(chat_render::preserved_tokens, s); }

// A function tool (foreach_function's test).
fn function_of(i: i64) -> i64 {
    let tool = json_schema::json_at(tools_v, i);
    let ty = chat_peg::jget(tool, "type");
    if ty < 0 || chat_peg::jstr(ty) != "function" || json_schema::json_kind(ty) != json_schema::J_STRING { return -1; }
    return chat_peg::jget(tool, "function");
}

//
// LFM2, LFM2.5 (lfm2.cpp)
//

fn init_lfm2(t: chat_render::Template, g: chat_render::Gen, tool_list_tokens: bool) {
    let TOOL_CALL_START = "<|tool_call_start|>";
    let TOOL_CALL_END = "<|tool_call_end|>";
    let THINK_START = "<think>";
    let THINK_END = "</think>";
    let GEN_PROMPT = "<|im_start|>assistant\n";
    // the reasoning as the "thinking" field the template reads
    var adjusted: u8[>..] = [];
    var ok = false;
    fn adjust() {
        jinja::reset();
        let arr = jinja::from_json(g.messages, false);
        for i in jinja::arr_len(arr) {
            let m = jinja::arr_at(arr, i);
            let r = jinja::obj_get_name(m, "reasoning_content");
            if r >= 0 && jinja::kind(r) == jinja::K_STRING { jinja::obj_set_name(m, "thinking", r); }
        }
        chat_render::dump(adjusted, arr);
        ok = true;
    }
    exception::try<jinja::Fail>(adjust);
    if !ok { json_schema::fail(jinja::err); }
    var gp: u8[>..] = [];
    render(t, g, chat_render::Override { has_messages: true, messages: adjusted, .. }, gp);
    chat_render::format = chat_render::FORMAT_PEG_NATIVE;
    chat_render::supports_thinking = true;
    preserve(TOOL_CALL_START);
    preserve(TOOL_CALL_END);
    preserve(THINK_START);
    preserve(THINK_END);
    if tool_list_tokens {
        preserve("<|tool_list_start|>");
        preserve("<|tool_list_end|>");
    }
    chat_render::thinking_start.append(THINK_START);
    chat_render::list_push(chat_render::thinking_ends, THINK_END);
    let tools = has_tools();
    let rf = schema_v >= 0 && json_schema::json_kind(schema_v) == json_schema::J_OBJECT;
    let extract = g.reasoning_format != chat_render::REASONING_NONE && find(chat_render::source(t), THINK_START) >= 0;
    let include = rf || (tools && g.tool_choice != chat_render::TOOL_CHOICE_NONE);
    if chat_render::has_continuation(g) {
        gp.clear();
        gp.append(GEN_PROMPT);
        gp.append(THINK_START);
        gp.append(g.continue_reasoning);
        if g.continue_final_message == chat_render::CONTINUE_CONTENT { gp.append(THINK_END); gp.append(g.continue_content); }
        chat_render::append_part(gp, false);
    }
    chat_render::gen_prompt.append(gp);
    peg::begin();
    let generation_prompt = peg::literal(GEN_PROMPT);
    let end = peg::end();
    var reasoning = peg::eps();
    if extract {
        // optional(THINK_START + reasoning(until(THINK_END)) + THINK_END)
        let r = chat_peg::reasoning(peg::until(THINK_END));
        reasoning = peg::optional(peg::chain([peg::cs(THINK_START), peg::ci(r), peg::cs(THINK_END)]));
    }
    var root = -1;
    if !tools || g.tool_choice == chat_render::TOOL_CHOICE_NONE {
        if rf {
            let j = peg::json();
            let node = json_schema::build_document(schema_v);
            let response_format = chat_peg::content(peg::schema(j, "response-format-schema", node, false));
            root = peg::chain([peg::ci(generation_prompt), peg::ci(reasoning), peg::ci(response_format), peg::ci(end)]);
        } else {
            let c = chat_peg::content(peg::rest());
            root = peg::chain([peg::ci(generation_prompt), peg::ci(reasoning), peg::ci(c), peg::ci(end)]);
        }
    } else {
        // rule("tool-calls", trigger_rule("tool-call", literal(START) + python_style_tool_calls(...) + literal(END)))
        let l2 = peg::literal(TOOL_CALL_END);
        let ps = chat_peg::python_style_tool_calls(tools_v, g.parallel_tool_calls, true);
        let l1 = peg::literal(TOOL_CALL_START);
        let tool_calls = peg::rule("tool-calls", peg::trigger_rule("tool-call", peg::chain([peg::ci(l1), peg::ci(ps), peg::ci(l2)])), false);
        let content = chat_peg::content(peg::until(TOOL_CALL_START));
        root = peg::chain([peg::ci(generation_prompt), peg::ci(reasoning), peg::ci(content), peg::ci(tool_calls), peg::ci(end)]);
    }
    chat_render::parser = peg::build(root);
    if include {
        build_grammar(!(rf || (tools && g.tool_choice == chat_render::TOOL_CHOICE_REQUIRED)));
        chat_render::add_trigger(chat_render::TRIGGER_WORD, TOOL_CALL_START);
    }
}

//
// MiniCPM5 (minicpm5.cpp)
//

fn init_minicpm5(t: chat_render::Template, g: chat_render::Gen) {
    var gp: u8[>..] = [];
    render(t, g, chat_render::Override { .. }, gp);
    chat_render::format = chat_render::FORMAT_PEG_NATIVE;
    chat_render::supports_thinking = true;
    preserve("<function");
    preserve("<param");
    preserve("</function>");
    preserve("</param>");
    preserve("<think>");
    preserve("</think>");
    chat_render::thinking_start.append("<think>");
    chat_render::list_push(chat_render::thinking_ends, "</think>");
    let tools = has_tools();
    let rf = has_response_format();
    let extract = g.reasoning_format != chat_render::REASONING_NONE;
    let include = rf || (tools && g.tool_choice != chat_render::TOOL_CHOICE_NONE);
    if chat_render::has_continuation(g) {
        gp.clear();
        gp.append("<|im_start|>assistant\n<think>\n");
        gp.append(g.continue_reasoning);
        if g.continue_final_message == chat_render::CONTINUE_CONTENT { gp.append("\n</think>\n\n"); gp.append(g.continue_content); }
        chat_render::append_part(gp, false);
    }
    chat_render::gen_prompt.append(gp);
    peg::begin();
    let generation_prompt = peg::literal("<|im_start|>assistant\n");
    var reasoning = peg::eps();
    if extract {
        // ("<think>" << reasoning(until("</think>")) << "</think>") + space()
        let s = peg::space();
        let r = chat_peg::reasoning(peg::until("</think>"));
        let open = peg::literal("<think>");
        let a = peg::spaced(open, r);
        let close = peg::literal("</think>");
        let b = peg::spaced(a, close);
        reasoning = peg::seq2(b, s);
    }
    var root = -1;
    if rf {
        let j = peg::json();
        let node = json_schema::build_document(schema_v);
        let c = chat_peg::content(peg::schema(j, "response-format", node, false));
        root = peg::chain([peg::ci(generation_prompt), peg::ci(reasoning), peg::ci(c)]);
    } else if tools && g.tool_choice != chat_render::TOOL_CHOICE_NONE {
        // CDATA, or up to </param>
        var a = -1;
        {
            let cl = chat_peg::tool_arg_close(peg::literal("</param>"));
            let l2 = peg::literal("]]>");
            let sv = chat_peg::tool_arg_string_value(peg::until("]]>"));
            let ac = peg::ac(peg::seq2(sv, l2), "]]>");
            let l1 = peg::literal("<![CDATA[");
            a = peg::chain([peg::ci(l1), peg::ci(ac), peg::ci(cl)]);
        }
        var b = -1;
        {
            let cl = chat_peg::tool_arg_close(peg::literal("</param>"));
            let sv = chat_peg::tool_arg_string_value(peg::until("</param>"));
            let ac = peg::ac(peg::seq2(sv, cl), "</param>");
            let n = peg::negate(peg::literal("<![CDATA["));
            b = peg::seq2(n, ac);
        }
        let string_value = peg::choice([a, b]);
        var tool_choice = peg::choice([]);
        for i in json_schema::json_len(tools_v) {
            let func = function_of(i);
            guard func >= 0 else { continue; }
            let name = str(chat_peg::jstr(chat_peg::jget(func, "name")));
            var arg_rules: i64[>..] = [];
            let root_node = chat_peg::params_schema(func);
            for k in json_schema::prop_count(root_node) {
                let pname = str(json_schema::prop_name(root_node, k));
                let pnode = json_schema::prop_node(root_node, k);
                var value_parser = peg::eps();
                if json_schema::may_be_string(pnode) {
                    value_parser = string_value;
                } else {
                    let cl = chat_peg::tool_arg_close(peg::literal("</param>"));
                    let j = peg::json();
                    let jv = chat_peg::tool_arg_json_value(peg::schema(j, str("tool-", i, "-arg-", k, "-schema"), pnode, false));
                    value_parser = peg::seq2(jv, cl);
                }
                // tool_arg(tool_arg_open("<param name=\"" + tool_arg_name(name) + "\">") + value)
                let l2 = peg::literal("\">");
                let tan = chat_peg::tool_arg_name(peg::literal(pname));
                let l1 = peg::literal("<param name=\"");
                let open = chat_peg::tool_arg_open(peg::chain([peg::ci(l1), peg::ci(tan), peg::ci(l2)]));
                arg_rules.push(chat_peg::tool_arg(peg::seq2(open, value_parser)));
            }
            var args = peg::eps();
            if arg_rules.len > 0 {
                let s = peg::space();
                args = peg::zero_or_more(peg::seq2(peg::choice(arg_rules), s));
            }
            // tool(tool_open("<function name=\"" + tool_name(name) + "\">") << tool_args(args) << tool_close("</function>"))
            let l2 = peg::literal("\">");
            let tn = chat_peg::tool_name(peg::literal(name));
            let l1 = peg::literal("<function name=\"");
            let open = chat_peg::tool_open(peg::chain([peg::ci(l1), peg::ci(tn), peg::ci(l2)]));
            let ta = chat_peg::tool_args(args);
            let x = peg::spaced(open, ta);
            let tc = chat_peg::tool_close(peg::literal("</function>"));
            let tp = chat_peg::tool(peg::spaced(x, tc));
            let r = peg::rule(str("tool-", i), tp, false);
            tool_choice = peg::choice2(tool_choice, r);
        }
        let max_calls = if g.parallel_tool_calls { -1 } else { 1 };
        let s = peg::space();
        let tool_calls = peg::trigger_rule("tool-call", peg::repeat(peg::seq2(tool_choice, s), 1, max_calls));
        let content = chat_peg::content(peg::until("<function"));
        let e = peg::end();
        root = peg::chain([peg::ci(generation_prompt), peg::ci(reasoning), peg::ci(content), peg::ci(tool_calls), peg::ci(e)]);
    } else {
        let e = peg::end();
        let c = chat_peg::content(peg::rest());
        root = peg::chain([peg::ci(generation_prompt), peg::ci(reasoning), peg::ci(c), peg::ci(e)]);
    }
    chat_render::parser = peg::build(root);
    if include {
        build_grammar(!(rf || (tools && g.tool_choice == chat_render::TOOL_CHOICE_REQUIRED)));
        chat_render::add_trigger(chat_render::TRIGGER_WORD, "<function");
    }
}

//
// Qwen3-Coder, Qwen3.5, Nemotron Nano 3, StepFun 3.5 (qwen3-coder.cpp)
//

fn init_qwen3_coder(t: chat_render::Template, g: chat_render::Gen) {
    let GEN_PREFIX = "<|im_start|>assistant\n";
    var gp: u8[>..] = [];
    render(t, g, chat_render::Override { .. }, gp);
    chat_render::format = chat_render::FORMAT_PEG_NATIVE;
    let supports_reasoning = find(chat_render::source(t), "<think>") >= 0;
    chat_render::supports_thinking = supports_reasoning;
    preserve("<tool_call>");
    preserve("</tool_call>");
    let is_qwen3_coder = !supports_reasoning;
    if supports_reasoning {
        chat_render::thinking_start.append("<think>");
        chat_render::list_push(chat_render::thinking_ends, "\n</think>");
        chat_render::list_push(chat_render::thinking_ends, "</think>");
        chat_render::list_push(chat_render::thinking_ends, "<tool_call>");
        preserve("<think>");
        preserve("</think>");
    }
    let tools = has_tools();
    let rf = has_response_format();
    let extract = g.reasoning_format != chat_render::REASONING_NONE;
    let include = rf || (tools && g.tool_choice != chat_render::TOOL_CHOICE_NONE);
    if chat_render::has_continuation(g) {
        gp.clear();
        gp.append(GEN_PREFIX);
        if supports_reasoning {
            gp.append("<think>\n");
            gp.append(g.continue_reasoning);
            if g.continue_final_message == chat_render::CONTINUE_CONTENT { gp.append("\n</think>\n\n"); }
        }
        if g.continue_final_message == chat_render::CONTINUE_CONTENT { gp.append(g.continue_content); }
        chat_render::append_part(gp, false);
    }
    chat_render::gen_prompt.append(gp);
    var starts: u8[>..] = [];
    chat_render::list_push(starts, "<tool_call>");
    if is_qwen3_coder && tools {
        for i in json_schema::json_len(tools_v) {
            let func = function_of(i);
            guard func >= 0 else { continue; }
            chat_render::list_push(starts, str("<function=", chat_peg::jstr(chat_peg::jget(func, "name")), ">"));
        }
    }
    peg::begin();
    let generation_prompt = peg::literal(GEN_PREFIX);
    // the template strips the newline before </think>, so it stays out of the reasoning
    let think_body = peg::until_one_of(["\n</think>", "</think>", "<tool_call>"]);
    let c3 = peg::peek(peg::literal("<tool_call>"));
    let c2 = peg::literal("</think>");
    let c1 = peg::literal("\n</think>");
    let think_close = peg::choice2(peg::choice2(c1, c2), c3);
    var reasoning_extracted = -1;
    {
        let r = chat_peg::reasoning(think_body);
        let s = peg::space();
        reasoning_extracted = peg::optional(peg::chain([peg::cs("<think>"), peg::ci(s), peg::ci(r), peg::ci(think_close)]));
    }
    var reasoning_skipped = -1;
    {
        let s = peg::space();
        reasoning_skipped = peg::optional(peg::chain([peg::cs("<think>"), peg::ci(s), peg::ci(think_body), peg::ci(think_close)]));
    }
    var reasoning = peg::eps();
    if supports_reasoning && extract { reasoning = reasoning_extracted; }
    var root = -1;
    if rf {
        // the answer must be bare JSON: the think block goes even when not extracted
        if supports_reasoning && !extract { reasoning = reasoning_skipped; }
        let j = peg::json();
        let node = json_schema::build_document(schema_v);
        let c = chat_peg::content(peg::schema(j, "response-format", node, false));
        root = peg::seq2(generation_prompt, peg::spaced(reasoning, c));
    } else if tools && g.tool_choice != chat_render::TOOL_CHOICE_NONE {
        let arg_close = chat_peg::tool_arg_close(peg::literal("\n</parameter>\n"));
        let sv = chat_peg::tool_arg_string_value(peg::until("\n</parameter>\n"));
        let arg_string = peg::rule("xml-arg-string", peg::ac(peg::seq2(sv, arg_close), "\n</parameter>\n"), false);
        var tool_choice = peg::choice([]);
        for i in json_schema::json_len(tools_v) {
            let func = function_of(i);
            guard func >= 0 else { continue; }
            let name = str(chat_peg::jstr(chat_peg::jget(func, "name")));
            var required_args: i64[>..] = [];
            var optional_args: i64[>..] = [];
            let root_node = chat_peg::params_schema(func);
            for k in json_schema::prop_count(root_node) {
                let rule_name = str("tool-", i, "-arg-", k);
                let pname = str(json_schema::prop_name(root_node, k));
                let pnode = json_schema::prop_node(root_node, k);
                // tool_arg_open("<parameter=" + tool_arg_name(literal(name)) + ">\n")
                let tan = chat_peg::tool_arg_name(peg::literal(pname));
                let arg_open = chat_peg::tool_arg_open(peg::chain([peg::cs("<parameter="), peg::ci(tan), peg::cs(">\n")]));
                let types = json_schema::value_types(pnode);
                let string_bit = 1 << json_schema::TYPE_STRING;
                var arg_value = peg::eps();
                if (types & string_bit) == 0 {
                    let j = peg::json();
                    let jv = chat_peg::tool_arg_json_value(peg::schema(j, str(rule_name, "-schema"), pnode, false));
                    arg_value = peg::seq2(jv, arg_close);
                } else if types == string_bit {
                    arg_value = arg_string;
                } else {
                    // the string alternative takes any text, so the grammar keeps only it;
                    // the parser tries the JSON ones first, to type the value
                    var json_value = peg::choice([]);
                    if (types & (1 << json_schema::TYPE_OBJECT)) != 0 { json_value = peg::choice2(json_value, peg::json_object()); }
                    if (types & (1 << json_schema::TYPE_ARRAY)) != 0 { json_value = peg::choice2(json_value, peg::json_array()); }
                    if (types & ((1 << json_schema::TYPE_NUMBER) | (1 << json_schema::TYPE_INTEGER))) != 0 { json_value = peg::choice2(json_value, peg::json_number()); }
                    if (types & (1 << json_schema::TYPE_BOOLEAN)) != 0 { json_value = peg::choice2(json_value, peg::json_bool()); }
                    if (types & (1 << json_schema::TYPE_NULL)) != 0 { json_value = peg::choice2(json_value, peg::json_null()); }
                    let at = peg::atomic(peg::seq2(chat_peg::tool_arg_json_value(json_value), arg_close));
                    arg_value = peg::gbnf(peg::choice2(at, arg_string), "xml-arg-string");
                }
                let arg_rule = peg::rule(rule_name, chat_peg::tool_arg(peg::seq2(arg_open, arg_value)), false);
                if json_schema::prop_required(root_node, k) { required_args.push(arg_rule); } else { optional_args.push(arg_rule); }
            }
            // required arguments in any order: Qwen does not always keep it
            var args = chat_peg::permute(str("tool-", i, "-args"), required_args);
            if optional_args.len > 0 { args = peg::seq2(args, peg::zero_or_more(peg::choice(optional_args))); }
            // tool(tool_open("<function=" + tool_name(name) + ">\n") + tool_args(args) + tool_close("</function>\n"))
            let tc = chat_peg::tool_close(peg::literal("</function>\n"));
            let ta = chat_peg::tool_args(args);
            let tn = chat_peg::tool_name(peg::literal(name));
            let open = chat_peg::tool_open(peg::chain([peg::cs("<function="), peg::ci(tn), peg::cs(">\n")]));
            let fp = chat_peg::tool(peg::chain([peg::ci(open), peg::ci(ta), peg::ci(tc)]));
            let r = peg::rule(str("tool-", i), fp, false);
            tool_choice = peg::choice2(tool_choice, r);
        }
        let min_calls = if g.tool_choice == chat_render::TOOL_CHOICE_REQUIRED { 1 } else { 0 };
        let s = peg::space();
        let tool_call_body = peg::chain([peg::ci(tool_choice), peg::cs("</tool_call>"), peg::ci(s)]);
        let tool_call = peg::rule("tool-call", peg::chain([peg::cs("<tool_call>\n"), peg::ci(tool_call_body)]), false);
        var tool_call_first = tool_call;
        if is_qwen3_coder {
            let o = peg::optional(peg::literal("<tool_call>\n"));
            tool_call_first = peg::rule("tool-call-first", peg::seq2(o, tool_call_body), false);
        }
        var calls = tool_call_first;
        if g.parallel_tool_calls {
            let z = peg::zero_or_more(tool_call);
            calls = peg::seq2(tool_call_first, z);
        }
        let tool_calls = peg::trigger_rule("tool-call-root", peg::repeat(calls, min_calls, 1));
        // content stops before the one or two newlines that separate it from the calls
        var ends: u8[>..] = [];
        var start = 0;
        for i in starts.len {
            guard starts[i] == 0 else { continue; }
            let st = starts[start..i];
            chat_render::list_push(ends, str("\n\n", st));
            chat_render::list_push(ends, str("\n", st));
            chat_render::list_push(ends, st);
            start = i + 1;
        }
        var ends_list: const u8[:][>..] = [];
        var s0 = 0;
        for i in ends.len {
            guard ends[i] == 0 else { continue; }
            ends_list.push(ends[s0..i]);
            s0 = i + 1;
        }
        let c = chat_peg::content(peg::until_one_of(ends_list));
        let rc = peg::spaced(reasoning, c);
        let body = peg::spaced(rc, tool_calls);
        root = peg::seq2(generation_prompt, body);
    } else {
        let c = chat_peg::content(peg::rest());
        root = peg::seq2(generation_prompt, peg::spaced(reasoning, c));
    }
    chat_render::parser = peg::build(root);
    if include {
        let lazy = tools && g.tool_choice == chat_render::TOOL_CHOICE_AUTO;
        build_grammar(lazy);
        if lazy {
            var start = 0;
            for i in starts.len {
                guard starts[i] == 0 else { continue; }
                chat_render::add_trigger(chat_render::TRIGGER_WORD, starts[start..i]);
                start = i + 1;
            }
        }
    }
}

//
// dispatch
//

fn init_handler(t: chat_render::Template, g: chat_render::Gen) {
    tools_v = if g.tools.len > 0 { json_schema::json_value(g.tools) } else { -1 };
    schema_v = if g.json_schema.len > 0 { json_schema::json_value(g.json_schema) } else { -1 };
    let h = t.handler;
    if h == H_LFM2 { init_lfm2(t, g, true); return; }
    if h == H_LFM2_5 { init_lfm2(t, g, false); return; }
    if h == H_MINICPM5 { init_minicpm5(t, g); return; }
    if h == H_QWEN3_CODER { init_qwen3_coder(t, g); return; }
    if h == H_GEMMA4 { init_gemma4(t, g); return; }
    if h == H_GEMMA4_LEGACY { init_gemma4_legacy(t, g); return; }
    if h == H_TRANSLATE_GEMMA { init_translate_gemma(t, g); return; }
    if h == H_GIGACHAT_V3 { init_gigachat_v3(t, g); return; }
    if h == H_FUNCTIONARY_V3_2 { init_functionary_v3_2(t, g); return; }
    if h == H_COHERE2MOE { init_cohere2moe(t, g); return; }
    if h == H_KIMI_K2 { init_kimi_k2(t, g); return; }
    if h == H_KIMI_K3 { init_kimi_k3(t, g); return; }
    if h == H_MINISTRAL_3 { init_ministral_3(t, g); return; }
    json_schema::fail(str("the dedicated handler ", handler_name(h), " is not ported yet"));
}

// The handler's params for template t with the inputs g, into chat_render
// (whose params the caller has cleared); false, and the message in err, if
// it fails.
fn try_init(t: chat_render::Template, g: chat_render::Gen) -> bool {
    var ok = false;
    fn body() { init_handler(t, g); ok = true; }
    exception::try<json_schema::Fail>(body) { f => err.clear(); err.append(json_schema::err); };
    return ok;
}

//
// Gemma 4 (gemma4.cpp)
//

fn has_content(m: i64) -> bool {
    let c = jinja::obj_get_name(m, "content");
    if c < 0 || jinja::kind(c) == jinja::K_NONE { return false; }
    if jinja::kind(c) == jinja::K_STRING { return jinja::sbytes(c).len > 0; }
    if jinja::kind(c) == jinja::K_ARRAY { return jinja::arr_len(c) > 0; }
    return false;
}

fn has_tool_calls(m: i64) -> bool {
    let tc = jinja::obj_get_name(m, "tool_calls");
    return tc >= 0 && jinja::kind(tc) == jinja::K_ARRAY && jinja::arr_len(tc) > 0;
}

fn role_of(m: i64) -> const u8[:] {
    let r = jinja::obj_get_name(m, "role");
    return if r >= 0 && jinja::kind(r) == jinja::K_STRING { jinja::sbytes(r) } else { "" };
}

// convert_tool_responses_gemma4: an assistant's tool calls, the tool
// results after them and the assistant's answer as one message with a
// tool_responses field. It throws jinja::Fail.
fn convert_tool_responses_gemma4(messages: i64) -> i64 {
    let result = jinja::mk_array();
    var i = 0;
    let n = jinja::arr_len(messages);
    while i < n {
        let msg = jinja::arr_at(messages, i);
        if role_of(msg) != "assistant" || !has_tool_calls(msg) {
            jinja::arr_push(result, msg);
            i++;
            continue;
        }
        let tool_calls = jinja::mk_array();
        let tool_responses = jinja::mk_array();
        var content = -1;
        var reasoning = -1;
        let rc = jinja::obj_get_name(msg, "reasoning_content");
        if rc >= 0 && jinja::kind(rc) == jinja::K_STRING { reasoning = rc; }
        let tcs = jinja::obj_get_name(msg, "tool_calls");
        for k in jinja::arr_len(tcs) { jinja::arr_push(tool_calls, jinja::arr_at(tcs, k)); }
        i++;
        while i < n && role_of(jinja::arr_at(messages, i)) == "tool" {
            let curr = jinja::arr_at(messages, i);
            var response = jinja::mk_none();
            let c = jinja::obj_get_name(curr, "content");
            if c >= 0 {
                if jinja::kind(c) == jinja::K_STRING {
                    // the content as JSON, or as it is
                    response = c;
                    let text = str(jinja::sbytes(c));
                    var parsed = -1;
                    fn parse_it() { parsed = jinja::from_json(text, false); }
                    exception::try<jinja::Fail>(parse_it);
                    if parsed >= 0 { response = parsed; }
                } else {
                    response = c;
                }
            }
            // the name of the call it answers, or its id
            var name = str("");
            let idx = jinja::arr_len(tool_responses);
            if idx < jinja::arr_len(tool_calls) {
                let tc = jinja::arr_at(tool_calls, idx);
                let f = jinja::obj_get_name(tc, "function");
                if f >= 0 && jinja::kind(f) == jinja::K_OBJECT {
                    let nm = jinja::obj_get_name(f, "name");
                    if nm >= 0 && jinja::kind(nm) == jinja::K_STRING { name = str(jinja::sbytes(nm)); }
                }
            }
            if name.len == 0 {
                let id = jinja::obj_get_name(curr, "tool_call_id");
                if id >= 0 && jinja::kind(id) == jinja::K_STRING { name = str(jinja::sbytes(id)); }
            }
            let entry = jinja::mk_object();
            jinja::obj_set_name(entry, "name", jinja::mk_str(name, false));
            jinja::obj_set_name(entry, "response", response);
            jinja::arr_push(tool_responses, entry);
            i++;
        }
        if i < n && role_of(jinja::arr_at(messages, i)) == "assistant" {
            let next = jinja::arr_at(messages, i);
            if !has_tool_calls(next) && has_content(next) {
                content = jinja::obj_get_name(next, "content");
                i++;
            }
        }
        let built = jinja::mk_object();
        jinja::obj_set_name(built, "role", jinja::mk_str("assistant", false));
        jinja::obj_set_name(built, "tool_calls", tool_calls);
        if jinja::arr_len(tool_responses) > 0 { jinja::obj_set_name(built, "tool_responses", tool_responses); }
        if content >= 0 && jinja::kind(content) != jinja::K_NONE { jinja::obj_set_name(built, "content", content); }
        if reasoning >= 0 { jinja::obj_set_name(built, "reasoning_content", reasoning); }
        jinja::arr_push(result, built);
    }
    return result;
}

fn init_gemma4(t: chat_render::Template, g: chat_render::Gen) {
    var gp: u8[>..] = [];
    render(t, g, chat_render::Override { .. }, gp);
    if g.add_generation_prompt && ends_with(chat_render::prompt, "<turn|>\n") {
        // after content and a tool call the template adds no model turn, which
        // keeps the model from its reasoning tokens
        gp.clear();
        gp.append("<|turn>model\n");
        chat_render::append_part(gp, false);
    }
    chat_render::format = chat_render::FORMAT_PEG_GEMMA4;
    chat_render::supports_thinking = true;
    chat_render::thinking_start.append("<|channel>thought");
    chat_render::list_push(chat_render::thinking_ends, "<channel|>");
    preserve("<|channel>");
    preserve("<channel|>");
    preserve("<|tool_call>");
    preserve("<tool_call|>");
    preserve("<|turn>");
    chat_render::user_start.append("<|turn>user");
    chat_render::assistant_start.append("<|turn>model");
    if chat_render::has_continuation(g) {
        gp.clear();
        if ends_with(chat_render::prompt, "<turn|>\n") { gp.append("<|turn>model\n"); }
        gp.append("<|channel>thought\n");
        gp.append(g.continue_reasoning);
        if g.continue_final_message == chat_render::CONTINUE_CONTENT { gp.append("<channel|>"); gp.append(g.continue_content); }
        chat_render::append_part(gp, false);
    }
    chat_render::gen_prompt.append(gp);
    let tools = has_tools();
    let rf = schema_v >= 0 && json_schema::json_kind(schema_v) == json_schema::J_OBJECT;
    let include = rf || (tools && g.tool_choice != chat_render::TOOL_CHOICE_NONE);
    let extract = g.reasoning_format != chat_render::REASONING_NONE;
    peg::begin();
    let start = peg::rule("start", peg::optional(peg::literal("<|turn>model\n")), false);
    if extract {
        let l2 = peg::literal("<channel|>");
        let r = chat_peg::reasoning(peg::until("<channel|>"));
        let s = peg::space();
        let l1 = peg::literal("<|channel>thought");
        peg::rule("thought", peg::chain([peg::ci(l1), peg::ci(s), peg::ci(r), peg::ci(l2)]), false);
    } else {
        let l2 = peg::literal("<channel|>");
        let u = peg::until("<channel|>");
        let s = peg::space();
        let l1 = peg::literal("<|channel>thought");
        peg::rule("thought", chat_peg::content(peg::chain([peg::ci(l1), peg::ci(s), peg::ci(u), peg::ci(l2)])), false);
    }
    // gbnf(zero_or_more(literal("<|channel>") + negate(literal("thought"))), "")
    let nt = peg::negate(peg::literal("thought"));
    let lc = peg::literal("<|channel>");
    let consume_empty_channels = peg::gbnf(peg::zero_or_more(peg::seq2(lc, nt)), "");
    // (peek("<|channel>") + consume_empty_channels + ref("thought")) | negate("<|channel>")
    let no_channel = peg::negate(peg::literal("<|channel>"));
    let tref = peg::ref("thought");
    let pk = peg::peek(peg::literal("<|channel>"));
    let thought = peg::choice2(peg::chain([peg::ci(pk), peg::ci(consume_empty_channels), peg::ci(tref)]), no_channel);
    var root = -1;
    if rf {
        // "```json" << content(schema(json(), ...)) << "```"
        let l1 = peg::literal("```json");
        let j = peg::json();
        let node = json_schema::build_document(schema_v);
        let c = chat_peg::content(peg::schema(j, "response-format-schema", node, false));
        let a = peg::spaced(l1, c);
        let l2 = peg::literal("```");
        let response_format = peg::spaced(a, l2);
        let o = peg::optional(thought);
        root = peg::chain([peg::ci(start), peg::ci(o), peg::ci(response_format)]);
    } else if tools && g.tool_choice != chat_render::TOOL_CHOICE_NONE {
        // the rules gemma4_to_json walks
        peg::rule("gemma4-string-content", peg::until("<|\"|>"), false);
        {
            let l2 = peg::literal("<|\"|>");
            let r = peg::ref("gemma4-string-content");
            let l1 = peg::literal("<|\"|>");
            peg::rule("gemma4-string", peg::chain([peg::ci(l1), peg::ci(r), peg::ci(l2)]), false);
        }
        peg::rule("gemma4-bool", peg::json_bool(), false);
        peg::rule("gemma4-null", peg::json_null(), false);
        peg::rule("gemma4-number", peg::json_number(), false);
        {
            let colon = peg::literal(":");
            let name = peg::rule("gemma4-dict-key-name", peg::chars("[^:}]", 1, -1), false);
            peg::rule("gemma4-dict-key", peg::seq2(name, colon), false);
        }
        {
            let v = peg::ref("gemma4-value");
            let s = peg::space();
            let k = peg::ref("gemma4-dict-key");
            peg::rule("gemma4-dict-kv", peg::chain([peg::ci(k), peg::ci(s), peg::ci(v)]), false);
        }
        if peg::rule_open("gemma4-dict", false) < 0 {
            let ws = peg::space();
            let member = peg::ref("gemma4-dict-kv");
            let comma = peg::literal(",");
            let members = peg::seq([member, peg::zero_or_more(peg::seq([comma, ws, member]))]);
            let open = peg::literal("{");
            let close1 = peg::literal("}");
            let close2 = peg::literal("}");
            peg::rule_close("gemma4-dict", peg::seq([open, ws, peg::choice([close1, peg::seq([members, ws, close2])])]), false);
        }
        if peg::rule_open("gemma4-array", false) < 0 {
            let ws = peg::space();
            let value = peg::ref("gemma4-value");
            let comma = peg::literal(",");
            let elements = peg::seq([value, peg::zero_or_more(peg::seq([comma, ws, value]))]);
            let open = peg::literal("[");
            let close1 = peg::literal("]");
            let close2 = peg::literal("]");
            peg::rule_close("gemma4-array", peg::seq([open, ws, peg::choice([close1, peg::seq([elements, ws, close2])])]), false);
        }
        if peg::rule_open("gemma4-value", false) < 0 {
            let a = peg::ref("gemma4-string");
            let b = peg::ref("gemma4-dict");
            let c = peg::ref("gemma4-array");
            let d = peg::ref("gemma4-number");
            let e = peg::ref("gemma4-bool");
            let f = peg::ref("gemma4-null");
            peg::rule_close("gemma4-value", peg::choice([a, b, c, d, e, f]), false);
        }
        var tool_choice = peg::choice([]);
        for i in json_schema::json_len(tools_v) {
            let func = function_of(i);
            guard func >= 0 else { continue; }
            let name = str(chat_peg::jstr(chat_peg::jget(func, "name")));
            // tool(sequence({ tool_open(tool_name(literal(name)) + peek(literal("{"))), tool_args(ref("gemma4-dict")) }))
            let pk2 = peg::peek(peg::literal("{"));
            let open = chat_peg::tool_open(peg::seq2(chat_peg::tool_name(peg::literal(name)), pk2));
            let args = chat_peg::tool_args(peg::ref("gemma4-dict"));
            let r = peg::rule(str("tool-", i), chat_peg::tool(peg::seq([open, args])), false);
            tool_choice = peg::choice2(tool_choice, r);
        }
        let min_calls = if g.tool_choice == chat_render::TOOL_CHOICE_REQUIRED { 1 } else { 0 };
        let max_calls = if g.parallel_tool_calls { -1 } else { 1 };
        let call = peg::chain([peg::cs("<|tool_call>call:"), peg::ci(tool_choice), peg::cs("<tool_call|>")]);
        let tool_call = peg::trigger_rule("tool-call", peg::repeat(call, min_calls, max_calls));
        if g.tool_choice == chat_render::TOOL_CHOICE_REQUIRED {
            root = peg::chain([peg::ci(start), peg::ci(thought), peg::ci(tool_call)]);
        } else {
            let scan = peg::rule("scan-to-toolcall", peg::until("<|tool_call>"), false);
            let content = peg::rule("content", chat_peg::content(peg::until_one_of(["<|channel>", "<channel|>", "<|tool_call>"])), false);
            let message = peg::rule("message", peg::seq2(thought, content), false);
            let z = peg::zero_or_more(message);
            root = peg::chain([peg::ci(start), peg::ci(z), peg::ci(scan), peg::ci(tool_call)]);
        }
    } else {
        // the content may end with an empty thought block, or a stray <channel|>
        let content = peg::rule("content", chat_peg::content(peg::until_one_of(["<|channel>", "<channel|>"])), false);
        let message = peg::rule("message", peg::seq2(thought, content), false);
        root = peg::seq2(start, peg::one_or_more(message));
    }
    chat_render::parser = peg::build(root);
    if include {
        build_grammar(!(rf || (tools && g.tool_choice == chat_render::TOOL_CHOICE_REQUIRED)));
        chat_render::add_trigger(chat_render::TRIGGER_WORD, "<|tool_call>");
    }
}

// The old Gemma 4 templates: the tool results folded into the assistant's message.
var gemma4_messages: u8[>..] = [];

fn init_gemma4_legacy(t: chat_render::Template, g: chat_render::Gen) {
    var adjusted: u8[>..] = [];
    var ok = false;
    fn convert() {
        jinja::reset();
        let arr = jinja::from_json(g.messages, false);
        chat_render::dump(adjusted, convert_tool_responses_gemma4(arr));
        ok = true;
    }
    exception::try<jinja::Fail>(convert);
    if !ok { json_schema::fail(jinja::err); }
    var g2 = g;
    g2.messages = adjusted;
    init_gemma4(t, g2);
}

//
// the Gemma 4 mapper (common_chat_peg_gemma4_mapper)
//

fn node_partial(n: i64) -> bool { peg::nodes[n].partial }
fn node_kids(n: i64) -> i64 { peg::nodes[n].kids_n }

recursive fn gemma4_to_json(out: u8[>..]&, n: i64) {
    let text = peg::node_text(n);
    if text.len == 0 { return; }
    let rule = peg::node_rule(n);
    if rule == "gemma4-number" || rule == "gemma4-bool" || rule == "gemma4-null" { out.append(text); return; }
    if rule == "gemma4-string-content" { chat_peg::escape_inner(out, text); return; }
    if rule == "gemma4-string" {
        out.push('"');
        if node_kids(n) > 0 {
            gemma4_to_json(out, peg::node_child(n, 0));
            if !node_partial(n) { out.push('"'); }
        }
        return;
    }
    if rule == "gemma4-array" || rule == "gemma4-dict" {
        let is_dict = rule == "gemma4-dict";
        out.push(if is_dict { '{' } else { '[' });
        for k in node_kids(n) {
            if k > 0 { out.push(','); }
            gemma4_to_json(out, peg::node_child(n, k));
        }
        if !node_partial(n) { out.push(if is_dict { '}' } else { ']' }); }
        return;
    }
    if rule == "gemma4-dict-key-name" { out.append(text); return; }
    if rule == "gemma4-dict-key" {
        out.push('"');
        if node_kids(n) > 0 {
            // the key's JSON at the end of out, then escaped in its place
            let mark = out.len;
            gemma4_to_json(out, peg::node_child(n, 0));
            let inner = str(out[mark..]);
            out.resize(mark);
            chat_peg::escape_inner(out, inner);
        }
        if !node_partial(n) { out.append("\":"); }
        return;
    }
    if rule == "gemma4-dict-kv" {
        for k in node_kids(n) { gemma4_to_json(out, peg::node_child(n, k)); }
        return;
    }
    if rule == "gemma4-value" {
        if node_kids(n) > 0 { gemma4_to_json(out, peg::node_child(n, 0)); }
        return;
    }
}

recursive fn gemma4_visit(n: i64) {
    let tag = peg::node_tag(n);
    if tag == "reasoning" { peg::sanitized_text(n, chat_peg::reasoning_text); return; }
    if tag == "content" { peg::sanitized_text(n, chat_peg::content_text); return; }
    if tag == "tool" {
        let name_id = peg::find_by_tag(n, "tool-name", 3);
        let args_id = peg::find_by_tag(n, "tool-args", 3);
        if name_id >= 0 && args_id >= 0 && !node_partial(name_id) {
            var args: u8[>..] = [];
            if node_kids(args_id) > 0 { gemma4_to_json(args, peg::node_child(args_id, 0)); }
            let nm = chat_peg::keep(peg::node_text(name_id));
            let a = chat_peg::keep(args);
            chat_peg::calls.push(chat_peg::Call { name: nm, arguments: a, id: chat_peg::S { .. } });
        }
        return;
    }
    for k in node_kids(n) { gemma4_visit(peg::node_child(n, k)); }
}

fn map_gemma4() {
    chat_peg::mapper_reset();
    for n in peg::result_nodes { gemma4_visit(n); }
}

//
// TranslateGemma (translate-gemma.cpp)
//

fn context_string(g: chat_render::Gen, key: const u8[:], def: const u8[:]) -> u8[>..] {
    if g.extra_context.len == 0 { return str(def); }
    let v = json_schema::json_value(g.extra_context);
    let x = chat_peg::jget(v, key);
    if x >= 0 && json_schema::json_kind(x) == json_schema::J_STRING { return str(json_schema::json_str(x)); }
    if x < 0 { write_stderr(str("TranslateGemma: ", key, " not set in chat_template_kwargs, defaulting to en-GB\n")); }
    return str(def);
}

fn init_translate_gemma(t: chat_render::Template, g: chat_render::Gen) {
    let src_lang = context_string(g, "source_lang_code", "en-GB");
    let tgt_lang = context_string(g, "target_lang_code", "en-GB");
    // the users' content in the template's schema, with the languages
    var adjusted: u8[>..] = [];
    var ok = false;
    fn adjust() {
        jinja::reset();
        let arr = jinja::from_json(g.messages, false);
        for i in jinja::arr_len(arr) {
            let m = jinja::arr_at(arr, i);
            guard role_of(m) == "user" else { continue; }
            var text: u8[>..] = [];
            let c = jinja::obj_get_name(m, "content");
            if c >= 0 && jinja::kind(c) == jinja::K_STRING {
                text.append(jinja::sbytes(c));
            } else if c >= 0 && jinja::kind(c) == jinja::K_ARRAY {
                for k in jinja::arr_len(c) {
                    if text.len > 0 { text.push(10); }
                    let tx = jinja::obj_get_name(jinja::arr_at(c, k), "text");
                    if tx >= 0 && jinja::kind(tx) == jinja::K_STRING { text.append(jinja::sbytes(tx)); }
                }
            }
            let part = jinja::mk_object();
            jinja::obj_set_name(part, "type", jinja::mk_str("text", false));
            jinja::obj_set_name(part, "text", jinja::mk_str(text, false));
            jinja::obj_set_name(part, "source_lang_code", jinja::mk_str(src_lang, false));
            jinja::obj_set_name(part, "target_lang_code", jinja::mk_str(tgt_lang, false));
            let parts = jinja::mk_array();
            jinja::arr_push(parts, part);
            jinja::obj_set_name(m, "content", parts);
        }
        chat_render::dump(adjusted, arr);
        ok = true;
    }
    exception::try<jinja::Fail>(adjust);
    if !ok { json_schema::fail(jinja::err); }
    var gp: u8[>..] = [];
    render(t, g, chat_render::Override { has_messages: true, messages: adjusted, .. }, gp);
    chat_render::format = chat_render::FORMAT_PEG_NATIVE;
    if chat_render::has_continuation(g) {
        gp.clear();
        gp.append("<start_of_turn>model\n");
        gp.append(g.continue_content);
        chat_render::append_part(gp, false);
    }
    chat_render::gen_prompt.append(gp);
    peg::begin();
    let l = peg::literal(gp);
    let c = chat_peg::content(peg::rest());
    chat_render::parser = peg::build(peg::spaced(l, c));
}

//
// GigaChat 3 (gigachat-v3.cpp)
//

fn init_gigachat_v3(t: chat_render::Template, g: chat_render::Gen) {
    var gp: u8[>..] = [];
    render(t, g, chat_render::Override { .. }, gp);
    chat_render::format = chat_render::FORMAT_PEG_NATIVE;
    preserve("<|message_sep|>\n\n");
    preserve("<|role_sep|>\n");
    if chat_render::has_continuation(g) {
        gp.clear();
        gp.append("assistant<|role_sep|>\n");
        gp.append(g.continue_content);
        chat_render::append_part(gp, false);
    }
    chat_render::gen_prompt.append(gp);
    let tools = has_tools();
    var include = tools && g.tool_choice != chat_render::TOOL_CHOICE_NONE;
    let prefix = "<|message_sep|>\n\nfunction call<|role_sep|>\n";
    peg::begin();
    var ret = peg::eps();
    if tools && g.tool_choice != chat_render::TOOL_CHOICE_NONE {
        var tool_choice = peg::choice([]);
        for i in json_schema::json_len(tools_v) {
            let func = chat_peg::jget(json_schema::json_at(tools_v, i), "function");
            let name = str(chat_peg::jstr(chat_peg::jget(func, "name")));
            // json_member("name", "\"" + tool_name(literal(name)) + "\"")
            let tn = chat_peg::tool_name(peg::literal(name));
            let quoted = peg::chain([peg::cs("\""), peg::ci(tn), peg::cs("\"")]);
            let tool_name = peg::json_member("name", quoted);
            let j = peg::json();
            let node = chat_peg::params_schema(func);
            let ta = chat_peg::tool_args(peg::schema(j, str("tool-", i, "-schema"), node, false));
            let tool_args = peg::json_member("arguments", ta);
            let open = chat_peg::tool_open(peg::spaced(peg::literal("{"), tool_name));
            // tool_open << "," << tool_args << "}"
            let comma = peg::literal(",");
            let a = peg::spaced(open, comma);
            let b = peg::spaced(a, tool_args);
            let close = peg::literal("}");
            let r = peg::rule(str("tool-", i), peg::spaced(b, close), false);
            tool_choice = peg::choice2(tool_choice, r);
        }
        let min_calls = if g.tool_choice == chat_render::TOOL_CHOICE_REQUIRED { 1 } else { 0 };
        let l = peg::literal(prefix);
        let tool_call = peg::rule("tool-call", peg::seq2(l, tool_choice), false);
        let tool_calls = peg::trigger_rule("tool-call-root", peg::repeat(tool_call, min_calls, 1));
        let c = chat_peg::content(peg::until("<|message_sep|>\n\n"));
        ret = peg::spaced(c, tool_calls);
    } else {
        include = false;
        ret = chat_peg::content(peg::rest());
    }
    let l = peg::literal("assistant<|role_sep|>\n");
    chat_render::parser = peg::build(peg::seq2(l, ret));
    if include {
        build_grammar(tools && g.tool_choice == chat_render::TOOL_CHOICE_AUTO);
        chat_render::add_trigger(chat_render::TRIGGER_WORD, prefix);
    }
}

//
// Functionary v3.2 (functionary-v3-2.cpp)
//

fn init_functionary_v3_2(t: chat_render::Template, g: chat_render::Gen) {
    var gp: u8[>..] = [];
    render(t, g, chat_render::Override { .. }, gp);
    chat_render::format = chat_render::FORMAT_PEG_NATIVE;
    preserve(">>>all");
    let tools = has_tools();
    let include = tools && g.tool_choice != chat_render::TOOL_CHOICE_NONE;
    if chat_render::has_continuation(g) {
        gp.clear();
        gp.append("<|start_header_id|>assistant<|end_header_id|>\n\n>>>all\n");
        gp.append(g.continue_content);
        chat_render::append_part(gp, false);
    }
    chat_render::gen_prompt.append(gp);
    peg::begin();
    // >>>all\n{content}, up to the next >>> when there are tools
    let c1 = chat_peg::content(peg::until(">>>"));
    let content_until_tool = peg::seq2(peg::literal("all\n"), c1);
    let c2 = chat_peg::content(peg::rest());
    let content_until_end = peg::seq2(peg::literal("all\n"), c2);
    let generation_prompt = peg::literal("<|start_header_id|>assistant<|end_header_id|>\n\n>>>");
    var root = -1;
    if !tools || g.tool_choice == chat_render::TOOL_CHOICE_NONE {
        let e = peg::end();
        root = peg::chain([peg::ci(generation_prompt), peg::ci(content_until_end), peg::ci(e)]);
    } else {
        var tool_choice = peg::choice([]);
        for i in json_schema::json_len(tools_v) {
            let func = function_of(i);
            guard func >= 0 else { continue; }
            let name = str(chat_peg::jstr(chat_peg::jget(func, "name")));
            // tool(tool_open(tool_name(literal(name)) + literal("\n")) + tool_args(schema(json(), ...)))
            let j = peg::json();
            let node = chat_peg::params_schema(func);
            let ta = chat_peg::tool_args(peg::schema(j, str("tool-", i, "-schema"), node, false));
            let nl = peg::literal("\n");
            let open = chat_peg::tool_open(peg::seq2(chat_peg::tool_name(peg::literal(name)), nl));
            let r = peg::rule(str("tool-", i), chat_peg::tool(peg::seq2(open, ta)), false);
            tool_choice = peg::choice2(tool_choice, r);
        }
        let content_only = content_until_end;
        let tools_only = peg::trigger_rule("tools", peg::one_or_more(tool_choice));
        let content_and_tools = peg::seq2(content_until_tool, tools_only);
        var ret = peg::eps();
        if g.tool_choice == chat_render::TOOL_CHOICE_REQUIRED {
            if g.parallel_tool_calls {
                let e = peg::end();
                ret = peg::seq2(peg::choice([content_and_tools, tools_only]), e);
            } else {
                let e = peg::end();
                let a = peg::seq2(content_until_tool, tool_choice);
                ret = peg::seq2(peg::choice([a, tools_only]), e);
            }
        } else if g.parallel_tool_calls {
            let e = peg::end();
            ret = peg::seq2(peg::choice([content_and_tools, content_only, tools_only]), e);
        } else {
            let content_and_tool = peg::seq2(content_until_tool, tool_choice);
            let e = peg::end();
            ret = peg::seq2(peg::choice([content_and_tool, content_only, tool_choice]), e);
        }
        root = peg::seq2(generation_prompt, ret);
    }
    chat_render::parser = peg::build(root);
    if include {
        build_grammar(g.tool_choice == chat_render::TOOL_CHOICE_AUTO);
        chat_render::add_trigger(chat_render::TRIGGER_PATTERN, ">>>(?!all)");
    }
}

//
// Cohere2 MoE, North Code (cohere2moe.cpp)
//

fn init_cohere2moe(t: chat_render::Template, g: chat_render::Gen) {
    let TURN_START = "<|START_OF_TURN_TOKEN|>";
    let TURN_END = "<|END_OF_TURN_TOKEN|>";
    let CHATBOT = "<|CHATBOT_TOKEN|>";
    let THINK_START = "<|START_THINKING|>";
    let THINK_END = "<|END_THINKING|>";
    let TEXT_START = "<|START_TEXT|>";
    let TEXT_END = "<|END_TEXT|>";
    let ACTION_START = "<|START_ACTION|>";
    let ACTION_END = "<|END_ACTION|>";
    let GEN_PREFIX = str(TURN_START, CHATBOT);
    var gp: u8[>..] = [];
    render(t, g, chat_render::Override { .. }, gp);
    chat_render::format = chat_render::FORMAT_PEG_NATIVE;
    chat_render::supports_thinking = true;
    chat_render::thinking_start.append(THINK_START);
    chat_render::list_push(chat_render::thinking_ends, THINK_END);
    for tk in [TURN_START, TURN_END, CHATBOT, "<|USER_TOKEN|>", "<|SYSTEM_TOKEN|>", THINK_START, THINK_END, TEXT_START, TEXT_END,
               ACTION_START, ACTION_END, "<|START_TOOL_RESULT|>", "<|END_TOOL_RESULT|>"] { preserve(tk); }
    let tools = has_tools();
    let rf = has_response_format();
    let extract = g.reasoning_format != chat_render::REASONING_NONE;
    let include = rf || (tools && g.tool_choice != chat_render::TOOL_CHOICE_NONE);
    if chat_render::has_continuation(g) {
        gp.clear();
        gp.append(GEN_PREFIX);
        gp.append(THINK_START);
        gp.append(g.continue_reasoning);
        if g.continue_final_message == chat_render::CONTINUE_CONTENT { gp.append(THINK_END); gp.append(TEXT_START); gp.append(g.continue_content); }
        chat_render::append_part(gp, false);
    }
    chat_render::gen_prompt.append(gp);
    peg::begin();
    let generation_prompt = peg::literal(GEN_PREFIX);
    let end = peg::end();
    var reasoning = peg::eps();
    if extract {
        let o = peg::optional(peg::literal(THINK_END));
        let r = chat_peg::reasoning(peg::until_one_of([THINK_END, TEXT_START, ACTION_START]));
        let l = peg::literal(THINK_START);
        reasoning = peg::optional(peg::chain([peg::ci(l), peg::ci(r), peg::ci(o)]));
    } else {
        let o = peg::optional(peg::literal(THINK_END));
        let u = peg::until_one_of([THINK_END, TEXT_START, ACTION_START]);
        let l = peg::literal(THINK_START);
        reasoning = peg::optional(chat_peg::content(peg::chain([peg::ci(l), peg::ci(u), peg::ci(o)])));
    }
    var text_content = -1;
    if rf {
        let o = peg::optional(peg::literal(TEXT_END));
        let j = peg::json();
        let node = json_schema::build_document(schema_v);
        let c = chat_peg::content(peg::schema(j, "response-format-schema", node, false));
        let l = peg::literal(TEXT_START);
        text_content = peg::chain([peg::ci(l), peg::ci(c), peg::ci(o)]);
    } else {
        let o = peg::optional(peg::literal(TEXT_END));
        let c = chat_peg::content(peg::until(TEXT_END));
        let l = peg::literal(TEXT_START);
        text_content = peg::chain([peg::ci(l), peg::ci(c), peg::ci(o)]);
    }
    var root = -1;
    if !tools || g.tool_choice == chat_render::TOOL_CHOICE_NONE {
        let o = peg::optional(peg::literal(TURN_END));
        root = peg::chain([peg::ci(generation_prompt), peg::ci(reasoning), peg::ci(text_content), peg::ci(o), peg::ci(end)]);
    } else {
        let require_tools = g.tool_choice == chat_render::TOOL_CHOICE_REQUIRED;
        let tool_calls = chat_peg::standard_json_tools(ACTION_START, ACTION_END, tools_v, g.parallel_tool_calls, true,
            "tool_name", "parameters", true, false, "", "tool_call_id", ["tool_call_id", "tool_name", "parameters"], false);
        // content and tool calls exclude each other
        let body = if require_tools { tool_calls } else { peg::choice([tool_calls, text_content]) };
        let o = peg::optional(peg::literal(TURN_END));
        root = peg::chain([peg::ci(generation_prompt), peg::ci(reasoning), peg::ci(body), peg::ci(o), peg::ci(end)]);
    }
    chat_render::parser = peg::build(root);
    if include {
        build_grammar(!rf && g.tool_choice == chat_render::TOOL_CHOICE_AUTO);
        chat_render::add_trigger(chat_render::TRIGGER_WORD, ACTION_START);
    }
}

//
// Kimi K2 Thinking (kimi-k2.cpp)
//

fn init_kimi_k2(t: chat_render::Template, g: chat_render::Gen) {
    let SECTION_BEGIN = "<|tool_calls_section_begin|>";
    let SECTION_END = "<|tool_calls_section_end|>";
    let CALL_BEGIN = "<|tool_call_begin|>";
    let ARGS_BEGIN = "<|tool_call_argument_begin|>";
    let CALL_END = "<|tool_call_end|>";
    let THINK_START = "<think>";
    let THINK_END = "</think>";
    let GEN_PROMPT = "<|im_assistant|>assistant<|im_middle|>";
    var gp: u8[>..] = [];
    render(t, g, chat_render::Override { .. }, gp);
    chat_render::format = chat_render::FORMAT_PEG_NATIVE;
    chat_render::supports_thinking = true;
    for tk in [SECTION_BEGIN, SECTION_END, CALL_BEGIN, ARGS_BEGIN, CALL_END, THINK_START, THINK_END] { preserve(tk); }
    let tools = has_tools();
    let extract = g.reasoning_format != chat_render::REASONING_NONE;
    let include = tools && g.tool_choice != chat_render::TOOL_CHOICE_NONE;
    chat_render::thinking_start.append(THINK_START);
    chat_render::list_push(chat_render::thinking_ends, THINK_END);
    if chat_render::has_continuation(g) {
        gp.clear();
        gp.append(GEN_PROMPT);
        gp.append(THINK_START);
        gp.append(g.continue_reasoning);
        if g.continue_final_message == chat_render::CONTINUE_CONTENT { gp.append(THINK_END); gp.append(g.continue_content); }
        chat_render::append_part(gp, false);
    }
    chat_render::gen_prompt.append(gp);
    peg::begin();
    let end = peg::end();
    var reasoning = -1;
    if extract {
        // optional(THINK_START + reasoning(until_one_of(...)) + optional(literal(THINK_END)))
        let o = peg::optional(peg::literal(THINK_END));
        let r = chat_peg::reasoning(peg::until_one_of([THINK_END, "<|tool_calls_section_begin|>", "<|tool_call_begin|>"]));
        reasoning = peg::optional(peg::chain([peg::cs(THINK_START), peg::ci(r), peg::ci(o)]));
    } else {
        reasoning = peg::eps();
    }
    let generation_prompt = peg::literal(GEN_PROMPT);
    var root = -1;
    if !tools || g.tool_choice == chat_render::TOOL_CHOICE_NONE {
        let c = chat_peg::content(peg::rest());
        root = peg::chain([peg::ci(generation_prompt), peg::ci(reasoning), peg::ci(c), peg::ci(end)]);
    } else {
        var tool_choice = peg::choice([]);
        for i in json_schema::json_len(tools_v) {
            let func = function_of(i);
            guard func >= 0 else { continue; }
            let name = str(chat_peg::jstr(chat_peg::jget(func, "name")));
            // tool_id("functions." + tool_name(literal(name)) + ":" + chars("[0-9]", 1, -1))
            let ch = peg::chars("[0-9]", 1, -1);
            let colon = peg::literal(":");
            let tn = chat_peg::tool_name(peg::literal(name));
            let fl = peg::literal("functions.");
            let tool_id = chat_peg::tool_id(peg::chain([peg::ci(fl), peg::ci(tn), peg::ci(colon), peg::ci(ch)]));
            // tool(tool_open(tool_id + ARGS_BEGIN) + tool_args(schema(...)) + tool_close(optional(CALL_END)))
            let tc = chat_peg::tool_close(peg::optional(peg::literal(CALL_END)));
            let j = peg::json();
            let node = chat_peg::params_schema(func);
            let ta = chat_peg::tool_args(peg::schema(j, str("tool-", i, "-schema"), node, false));
            let ab = peg::literal(ARGS_BEGIN);
            let open = chat_peg::tool_open(peg::seq2(tool_id, ab));
            let tp = chat_peg::tool(peg::chain([peg::ci(open), peg::ci(ta), peg::ci(tc)]));
            let r = peg::rule(str("tool-", i), tp, false);
            tool_choice = peg::choice2(tool_choice, r);
        }
        let min_calls = if g.tool_choice == chat_render::TOOL_CHOICE_REQUIRED { 1 } else { 0 };
        let max_calls = if g.parallel_tool_calls { -1 } else { 1 };
        // rule("tool-calls", optional(SECTION_BEGIN) + trigger_rule("tool-call", repeat(CALL_BEGIN + tool_choice) + optional(SECTION_END)))
        let o2 = peg::optional(peg::literal(SECTION_END));
        let rep = peg::repeat(peg::chain([peg::cs(CALL_BEGIN), peg::ci(tool_choice)]), min_calls, max_calls);
        let tr = peg::trigger_rule("tool-call", peg::seq2(rep, o2));
        let o1 = peg::optional(peg::literal(SECTION_BEGIN));
        let tool_calls = peg::rule("tool-calls", peg::seq2(o1, tr), false);
        let before = chat_peg::content(peg::until_one_of([SECTION_BEGIN, CALL_BEGIN]));
        root = peg::chain([peg::ci(generation_prompt), peg::ci(reasoning), peg::ci(before), peg::ci(tool_calls), peg::ci(end)]);
    }
    chat_render::parser = peg::build(root);
    if include {
        build_grammar(g.tool_choice == chat_render::TOOL_CHOICE_AUTO);
        chat_render::add_trigger(chat_render::TRIGGER_WORD, "<|tool_call_begin|>");
    }
}

//
// Kimi K3 (kimi-k3.cpp)
//

fn init_kimi_k3(t: chat_render::Template, g: chat_render::Gen) {
    let SEP = "<|sep|>";
    let MSG_START = "<|open|>message role=\"assistant\"<|sep|>";
    let THINK_START = "<|open|>think<|sep|>";
    let THINK_END = "<|close|>think<|sep|>";
    let RESP_START = "<|open|>response<|sep|>";
    let RESP_END = "<|close|>response<|sep|>";
    let TOOLS_START = "<|open|>tools<|sep|>";
    let TOOLS_END = "<|close|>tools<|sep|>";
    let CALL_START = "<|open|>call tool=\"";
    let CALL_END = "<|close|>call<|sep|>";
    let ARG_START = "<|open|>argument key=\"";
    let ARG_END = "<|close|>argument<|sep|>";
    let MSG_END = "<|close|>message<|sep|>";
    let EOM_TOKEN = "<|end_of_msg|>";
    var gp: u8[>..] = [];
    render(t, g, chat_render::Override { .. }, gp);
    chat_render::format = chat_render::FORMAT_PEG_NATIVE;
    chat_render::supports_thinking = true;
    // only the markers are special tokens
    for tk in ["<|open|>", "<|close|>", "<|sep|>", "<|end_of_msg|>"] { preserve(tk); }
    chat_render::thinking_start.append(THINK_START);
    chat_render::list_push(chat_render::thinking_ends, THINK_END);
    let tools = has_tools();
    let extract = g.reasoning_format != chat_render::REASONING_NONE;
    let include = tools && g.tool_choice != chat_render::TOOL_CHOICE_NONE;
    if chat_render::has_continuation(g) {
        gp.clear();
        gp.append(MSG_START);
        gp.append(THINK_START);
        gp.append(g.continue_reasoning);
        if g.continue_final_message == chat_render::CONTINUE_CONTENT { gp.append(THINK_END); gp.append(RESP_START); gp.append(g.continue_content); }
        chat_render::append_part(gp, false);
    }
    chat_render::gen_prompt.append(gp);
    peg::begin();
    let end = peg::end();
    let start = peg::optional(peg::literal(MSG_START));
    let think_body = if extract { chat_peg::reasoning(peg::until_one_of([THINK_END, RESP_START])) } else { chat_peg::content(peg::until_one_of([THINK_END, RESP_START])) };
    // optional(optional(THINK_START) + think_body + optional(THINK_END))
    let ro2 = peg::optional(peg::literal(THINK_END));
    let ro1 = peg::optional(peg::literal(THINK_START));
    let reasoning = peg::optional(peg::chain([peg::ci(ro1), peg::ci(think_body), peg::ci(ro2)]));
    // optional(RESP_START) + content(until_one_of(...)) + optional(RESP_END)
    let po2 = peg::optional(peg::literal(RESP_END));
    let pc = chat_peg::content(peg::until_one_of([RESP_END, TOOLS_START, MSG_END]));
    let po1 = peg::optional(peg::literal(RESP_START));
    let response = peg::chain([peg::ci(po1), peg::ci(pc), peg::ci(po2)]);
    let to2 = peg::optional(peg::literal(EOM_TOKEN));
    let to1 = peg::optional(peg::literal(MSG_END));
    let trailer = peg::seq2(to1, to2);
    var root = -1;
    if !tools || g.tool_choice == chat_render::TOOL_CHOICE_NONE {
        root = peg::chain([peg::ci(start), peg::ci(reasoning), peg::ci(response), peg::ci(trailer), peg::ci(end)]);
    } else {
        var tool_choices = peg::choice([]);
        for i in json_schema::json_len(tools_v) {
            let func = function_of(i);
            guard func >= 0 else { continue; }
            let name = str(chat_peg::jstr(chat_peg::jget(func, "name")));
            let schema = chat_peg::tool_parameters(func);
            var args = peg::eps();
            let props = chat_peg::jget(schema, "properties");
            if props >= 0 && json_schema::json_len(props) > 0 {
                var arg_choices = peg::choice([]);
                for k in json_schema::json_len(props) {
                    let key = str(json_schema::json_key(props, k));
                    let pv = json_schema::json_at(props, k);
                    var ty = str("string");
                    let tyv = chat_peg::jget(pv, "type");
                    if tyv >= 0 && json_schema::json_kind(tyv) == json_schema::J_STRING { ty = str(json_schema::json_str(tyv)); }
                    let value = if ty == "string" { chat_peg::tool_arg_string_value(peg::until(ARG_END)) } else { chat_peg::tool_arg_value(peg::until(ARG_END)) };
                    // the trailing type="..." attribute skipped: anything up to <|sep|>
                    let close = chat_peg::tool_arg_close(peg::literal(ARG_END));
                    let ls = peg::literal(SEP);
                    let u = peg::until(SEP);
                    let q = peg::literal("\"");
                    let tan = chat_peg::tool_arg_name(peg::literal(key));
                    let open = chat_peg::tool_arg_open(peg::literal(ARG_START));
                    let arg = chat_peg::tool_arg(peg::chain([peg::ci(open), peg::ci(tan), peg::ci(q), peg::ci(u), peg::ci(ls), peg::ci(value), peg::ci(close)]));
                    let r = peg::rule(str("kimi-k3-arg-", i, "-", k), arg, false);
                    arg_choices = peg::choice2(arg_choices, r);
                }
                args = peg::zero_or_more(arg_choices);
            }
            // the trailing index="N" attribute skipped the same way
            let tc = chat_peg::tool_close(peg::literal(CALL_END));
            let ta = chat_peg::tool_args(args);
            let ls = peg::literal(SEP);
            let u = peg::until(SEP);
            let q = peg::literal("\"");
            let tn = chat_peg::tool_name(peg::literal(name));
            let cs = peg::literal(CALL_START);
            let open = chat_peg::tool_open(peg::chain([peg::ci(cs), peg::ci(tn), peg::ci(q), peg::ci(u), peg::ci(ls)]));
            let call = chat_peg::tool(peg::chain([peg::ci(open), peg::ci(ta), peg::ci(tc)]));
            let r = peg::rule(str("kimi-k3-tool-", i), call, false);
            tool_choices = peg::choice2(tool_choices, r);
        }
        // one tools section, then the message closed (in the trigger rule, or
        // the lazy grammar rejects the closer once calls have started)
        let e2 = peg::optional(peg::literal(EOM_TOKEN));
        let e1 = peg::optional(peg::literal(MSG_END));
        let te = peg::literal(TOOLS_END);
        let oom = peg::one_or_more(tool_choices);
        let ts = peg::literal(TOOLS_START);
        let section = peg::trigger_rule("kimi-k3-tool-call", peg::chain([peg::ci(ts), peg::ci(oom), peg::ci(te), peg::ci(e1), peg::ci(e2)]));
        let tools_p = if g.tool_choice == chat_render::TOOL_CHOICE_REQUIRED { section } else { peg::optional(section) };
        root = peg::chain([peg::ci(start), peg::ci(reasoning), peg::ci(response), peg::ci(tools_p), peg::ci(trailer), peg::ci(end)]);
    }
    chat_render::parser = peg::build(root);
    if include {
        build_grammar(g.tool_choice != chat_render::TOOL_CHOICE_REQUIRED);
        chat_render::add_trigger(chat_render::TRIGGER_WORD, TOOLS_START);
    }
}

//
// Ministral 3, Magistral (ministral3.cpp)
//

fn init_ministral_3(t: chat_render::Template, g: chat_render::Gen) {
    // the system and assistant messages as typed blocks: thinking, then text
    var adjusted: u8[>..] = [];
    var ok = false;
    fn adjust() {
        jinja::reset();
        let arr = jinja::from_json(g.messages, false);
        let out = jinja::mk_array();
        for i in jinja::arr_len(arr) {
            let m = jinja::arr_at(arr, i);
            let role = role_of(m);
            if role != "system" && role != "assistant" { jinja::arr_push(out, m); continue; }
            let content = jinja::mk_array();
            let rc = jinja::obj_get_name(m, "reasoning_content");
            if rc >= 0 && jinja::kind(rc) == jinja::K_STRING {
                let b = jinja::mk_object();
                jinja::obj_set_name(b, "type", jinja::mk_str("thinking", false));
                jinja::obj_set_name(b, "thinking", rc);
                jinja::arr_push(content, b);
            }
            let c = jinja::obj_get_name(m, "content");
            if c >= 0 {
                if jinja::kind(c) == jinja::K_STRING {
                    let b = jinja::mk_object();
                    jinja::obj_set_name(b, "type", jinja::mk_str("text", false));
                    jinja::obj_set_name(b, "text", c);
                    jinja::arr_push(content, b);
                } else if jinja::kind(c) == jinja::K_ARRAY {
                    for k in jinja::arr_len(c) { jinja::arr_push(content, jinja::arr_at(c, k)); }
                }
            }
            // the message, its content replaced and its reasoning_content gone
            let a = jinja::mk_object();
            var had_content = false;
            for k in jinja::obj_len(m) {
                let key = jinja::sbytes(jinja::obj_key(m, k));
                if key == "reasoning_content" { continue; }
                if key == "content" { jinja::obj_set_name(a, "content", content); had_content = true; continue; }
                jinja::obj_set_name(a, key, jinja::obj_value(m, k));
            }
            if !had_content { jinja::obj_set_name(a, "content", content); }
            jinja::arr_push(out, a);
        }
        chat_render::dump(adjusted, out);
        ok = true;
    }
    exception::try<jinja::Fail>(adjust);
    if !ok { json_schema::fail(jinja::err); }
    let tools = has_tools();
    let rf = has_response_format();
    let extract = g.reasoning_format != chat_render::REASONING_NONE;
    var include = true;
    chat_render::supports_thinking = true;
    chat_render::thinking_start.append("[THINK]");
    chat_render::list_push(chat_render::thinking_ends, "[/THINK]");
    var gp: u8[>..] = [];
    render(t, g, chat_render::Override { has_messages: true, messages: adjusted, .. }, gp);
    chat_render::format = chat_render::FORMAT_PEG_NATIVE;
    for tk in ["[THINK]", "[/THINK]", "[TOOL_CALLS]", "[ARGS]"] { preserve(tk); }
    if chat_render::has_continuation(g) {
        gp.clear();
        gp.append("[THINK]");
        gp.append(g.continue_reasoning);
        if g.continue_final_message == chat_render::CONTINUE_CONTENT { gp.append("[/THINK]"); gp.append(g.continue_content); }
        chat_render::append_part(gp, false);
    }
    chat_render::gen_prompt.append(gp);
    peg::begin();
    let generation_prompt = peg::eps();
    var reasoning = -1;
    if extract {
        let r = chat_peg::reasoning(peg::until("[/THINK]"));
        reasoning = peg::optional(peg::chain([peg::cs("[THINK]"), peg::ci(r), peg::cs("[/THINK]")]));
    } else {
        reasoning = peg::eps();
    }
    var root = -1;
    if rf {
        // the JSON in code fences
        let l1 = peg::literal("```json");
        let a = peg::spaced(reasoning, l1);
        let j = peg::json();
        let node = json_schema::build_document(schema_v);
        let c = chat_peg::content(peg::schema(j, "response-format", node, false));
        let b = peg::spaced(a, c);
        let l2 = peg::literal("```");
        root = peg::seq2(generation_prompt, peg::spaced(b, l2));
    } else if tools && g.tool_choice != chat_render::TOOL_CHOICE_NONE {
        var tool_choice = peg::choice([]);
        for i in json_schema::json_len(tools_v) {
            let func = function_of(i);
            guard func >= 0 else { continue; }
            let name = str(chat_peg::jstr(chat_peg::jget(func, "name")));
            // tool_open(tool_name(literal(name)) + "[ARGS]") + tool_args(schema(...))
            let j = peg::json();
            let node = chat_peg::params_schema(func);
            let ta = chat_peg::tool_args(peg::schema(j, str("tool-", i, "-schema"), node, false));
            let tn = chat_peg::tool_name(peg::literal(name));
            let open = chat_peg::tool_open(peg::chain([peg::ci(tn), peg::cs("[ARGS]")]));
            let r = peg::rule(str("tool-", i), peg::seq2(open, ta), false);
            tool_choice = peg::choice2(tool_choice, r);
        }
        let min_calls = if g.tool_choice == chat_render::TOOL_CHOICE_REQUIRED { 1 } else { 0 };
        let max_calls = if g.parallel_tool_calls { -1 } else { 1 };
        let tool_calls = peg::trigger_rule("tool-call", peg::repeat(peg::chain([peg::cs("[TOOL_CALLS]"), peg::ci(tool_choice)]), min_calls, max_calls));
        let c = chat_peg::content(peg::until("[TOOL_CALLS]"));
        let a = peg::spaced(reasoning, c);
        root = peg::seq2(generation_prompt, peg::spaced(a, tool_calls));
    } else {
        include = false;
        let c = chat_peg::content(peg::rest());
        root = peg::seq2(generation_prompt, peg::spaced(reasoning, c));
    }
    chat_render::parser = peg::build(root);
    if include {
        build_grammar(tools && g.tool_choice == chat_render::TOOL_CHOICE_AUTO);
        chat_render::add_trigger(chat_render::TRIGGER_WORD, "[TOOL_CALLS]");
    }
}
