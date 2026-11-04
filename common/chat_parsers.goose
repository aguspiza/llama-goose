// llama.cpp's dedicated chat handlers: the templates the autoparser cannot
// read get a prompt and output parser of their own. detect picks the
// handler from the template's source, in llama.cpp's order.
// port-of: common/parsers/parsers.cpp, common/chat.cpp (common_chat_template_detect_params_init)
import std;
import exception;
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
fn map_gemma4() { chat_peg::map_parse(); }
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
