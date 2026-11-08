// test-chat.cpp's PEG parser cases (test_template_output_peg_parsers):
// each template applied to its inputs, and the model output parsed back,
// compared with what llama.cpp makes of them (tests/data/peg_chat_cases.txt,
// made by scripts/gen-peg-chat-cases.sh).
//   chat_peg [-v] [template name filter]
import std;
import os;
import common from "../common";
import common.number;
import common.json_schema;
import common.peg;
import common.chat_peg;
import common.chat_render;
import common.chat_apply;

var data: u8[>..] = [];
var at = 0;

fn field() -> const u8[:] {
    var n = 0;
    while data[at] != '\n' { n = n * 10 + (data[at] - '0'); at += 1; }
    at += 1;
    let s = data[at..at + n];
    at += n + 1;
    return s;
}

// the inputs' texts, kept while the case runs
struct R { at: i64 = 0, len: i64 = 0 }
struct MsgR { role: R, content: R, reasoning: R, tool_name: R, tool_call_id: R, calls_at: i64, calls_n: i64, parts_at: i64, parts_n: i64 }
struct CallR { name: R, args: R, id: R }
struct PartR { kind: R, text: R, name: R }
struct ToolR { name: R, description: R, parameters: R, defer_loading: bool }
struct KVR { key: R, value: R }

var store: u8[>..] = [];
fn keep(s: const u8[:]) -> R { let a = store.len; store.append(s); return R { at: a, len: s.len }; }

fn jget(v: i64, k: const u8[:]) -> i64 { if v >= 0 && json_schema::json_kind(v) == json_schema::J_OBJECT { json_schema::get_member(v, k) } else { -1 } }
fn jstr(v: i64) -> const u8[:] { if v >= 0 && json_schema::json_kind(v) == json_schema::J_STRING { json_schema::json_str(v) } else { "" } }
fn jint(v: i64) -> i64 { if v >= 0 { json_schema::jv[v].i } else { 0 } }
fn jbool(v: i64) -> bool { v >= 0 && json_schema::jv[v].i != 0 }

fn strings_json(out: u8[>..]&, l: const u8[:]) {
    out.push('[');
    var start = 0;
    var first = true;
    for i in l.len {
        if l[i] == 0 {
            if !first { out.push(','); }
            first = false;
            json_schema::json_string(out, l[start..i]);
            start = i + 1;
        }
    }
    out.push(']');
}

fn main() {
    let args = args();
    var verbose = false;
    var filter: u8[>..] = [];
    for a, i in args {
        if i == 0 { continue; }
        if a == "-v" { verbose = true; } else { filter.clear(); filter.append(a); }
    }
    if !read_file("tests/data/peg_chat_cases.txt", data) { print("cannot read the cases"); exit(1); }
    field();
    let tf = field();
    var t0 = 0;
    for c in tf { t0 = t0 * 10 + (c - '0'); }
    number::fixed_time = t0;
    // the templates, by name
    var tnames: u8[>..] = [];
    var tsrcs: R[>..] = [];
    var tname_at: R[>..] = [];
    var tsrc_store: u8[>..] = [];
    var cases = 0;
    var failed = 0;
    var checked = 0;
    var failed_templates: u8[>..] = [];
    while at < data.len {
        let kind = field();
        if kind == "S" {
            let name = field();
            let src = field();
            tname_at.push(R { at: tnames.len, len: name.len });
            tnames.append(name);
            tsrcs.push(R { at: tsrc_store.len, len: src.len });
            tsrc_store.append(src);
            continue;
        }
        let path = field();
        let input = field();
        let inputs_json = field();
        let ok = field() == "1";
        // what llama.cpp made, as fields
        var expected: const u8[:][>..] = [];
        if ok {
            for k in 11 { expected.push(field()); }
            let n = field();
            expected.push(n);
            var nparses = 0;
            for c in n { nparses = nparses * 10 + (c - '0'); }
            for p in nparses {
                expected.push(field());
                expected.push(field());
                let pok = field();
                expected.push(pok);
                if pok == "1" {
                    expected.push(field());
                    expected.push(field());
                    expected.push(field());
                    let nc = field();
                    expected.push(nc);
                    var ncalls = 0;
                    for c in nc { ncalls = ncalls * 10 + (c - '0'); }
                    for q in ncalls { expected.push(field()); expected.push(field()); expected.push(field()); }
                } else {
                    expected.push(field());
                }
            }
        } else {
            expected.push(field());
        }
        if filter.len > 0 && find(path, filter) < 0 { continue; }
        cases++;
        // the template
        var ti = -1;
        for r, k in tname_at { if tnames[r.at..r.at + r.len] == path { ti = k; } }
        let src = if ti >= 0 { tsrc_store[tsrcs[ti].at..tsrcs[ti].at + tsrcs[ti].len] } else { tsrc_store[..0] };
        let ts, tok = chat_apply::init(src, "", "", "", false, false);
        if !tok { print("FAIL ", path, ": template: ", chat_apply::err); failed++; continue; }

        // the inputs
        store.clear();
        json_schema::reset();
        let iv = json_schema::json_value(inputs_json);
        var msgs: MsgR[>..] = [];
        var calls: CallR[>..] = [];
        var parts: PartR[>..] = [];
        let mv = jget(iv, "messages");
        for i in json_schema::json_len(mv) {
            let m = json_schema::json_at(mv, i);
            var mr = MsgR { role: keep(jstr(jget(m, "role"))), content: R { .. }, reasoning: keep(jstr(jget(m, "reasoning_content"))),
                            tool_name: keep(jstr(jget(m, "name"))), tool_call_id: keep(jstr(jget(m, "tool_call_id"))),
                            calls_at: calls.len, calls_n: 0, parts_at: parts.len, parts_n: 0 };
            let c = jget(m, "content");
            if c >= 0 && json_schema::json_kind(c) == json_schema::J_ARRAY {
                for k in json_schema::json_len(c) {
                    let p = json_schema::json_at(c, k);
                    parts.push(PartR { kind: keep(jstr(jget(p, "type"))), text: keep(jstr(jget(p, "text"))), name: keep(jstr(jget(p, "name"))) });
                }
                mr.parts_n = parts.len - mr.parts_at;
            } else {
                mr.content = keep(jstr(c));
            }
            let tcs = jget(m, "tool_calls");
            if tcs >= 0 {
                for k in json_schema::json_len(tcs) {
                    let tc = json_schema::json_at(tcs, k);
                    let f = jget(tc, "function");
                    let a = jget(f, "arguments");
                    var args_text: u8[>..] = [];
                    if json_schema::json_kind(a) == json_schema::J_STRING { args_text.append(jstr(a)); } else { json_schema::json_dump(args_text, a); }
                    calls.push(CallR { name: keep(jstr(jget(f, "name"))), args: keep(args_text), id: keep(jstr(jget(tc, "id"))) });
                }
                mr.calls_n = calls.len - mr.calls_at;
            }
            msgs.push(mr);
        }
        var tools: ToolR[>..] = [];
        let tv = jget(iv, "tools");
        if tv >= 0 && json_schema::json_kind(tv) == json_schema::J_ARRAY {
            for i in json_schema::json_len(tv) {
                let f = jget(json_schema::json_at(tv, i), "function");
                var params: u8[>..] = [];
                json_schema::json_dump(params, jget(f, "parameters"));
                tools.push(ToolR { name: keep(jstr(jget(f, "name"))), description: keep(jstr(jget(f, "description"))), parameters: keep(params), defer_loading: jbool(jget(f, "defer_loading")) });
            }
        }
        var kvs: KVR[>..] = [];
        let kw = jget(iv, "chat_template_kwargs");
        for i in json_schema::json_len(kw) { kvs.push(KVR { key: keep(json_schema::json_key(kw, i)), value: keep(jstr(json_schema::json_at(kw, i))) }); }
        let schema_r = keep(jstr(jget(iv, "json_schema")));
        let tool_choice = jint(jget(iv, "tool_choice"));
        let parallel = jbool(jget(iv, "parallel_tool_calls"));
        let reasoning_format = jint(jget(iv, "reasoning_format"));
        let enable_thinking = jbool(jget(iv, "enable_thinking"));
        let add_gen = jbool(jget(iv, "add_generation_prompt"));
        let cont = jint(jget(iv, "continue_final_message"));
        // the structures over store, now that it is all there
        fn s(r: R) -> const u8[:] { store[r.at..r.at + r.len] }
        var mcalls: chat_apply::ToolCall[>..] = [];
        for c in calls { mcalls.push(chat_apply::ToolCall { name: s(c.name), arguments: s(c.args), id: s(c.id) }); }
        var mparts: chat_apply::Part[>..] = [];
        for p in parts { mparts.push(chat_apply::Part { kind: s(p.kind), text: s(p.text), name: s(p.name) }); }
        var messages: chat_apply::Message[>..] = [];
        for m in msgs {
            messages.push(chat_apply::Message { role: s(m.role), content: s(m.content), reasoning: s(m.reasoning),
                tool_calls: mcalls[m.calls_at..m.calls_at + m.calls_n], parts: mparts[m.parts_at..m.parts_at + m.parts_n],
                tool_name: s(m.tool_name), tool_call_id: s(m.tool_call_id), .. });
        }
        var mtools: chat_apply::Tool[>..] = [];
        for t in tools { mtools.push(chat_apply::Tool { name: s(t.name), description: s(t.description), parameters: s(t.parameters), defer_loading: t.defer_loading }); }
        var mkv: chat_apply::KV[>..] = [];
        for kv in kvs { mkv.push(chat_apply::KV { key: s(kv.key), value: s(kv.value) }); }
        let inp = chat_apply::Inputs { messages: messages, json_schema: s(schema_r), add_generation_prompt: add_gen, continue_final_message: cont,
            tools: mtools, tool_choice: tool_choice, parallel_tool_calls: parallel, reasoning_format: reasoning_format,
            enable_thinking: enable_thinking, kwargs: mkv, .. };

        let got_ok = chat_apply::apply(ts, inp);
        checked++;
        var bad = false;
        fn report(what: const u8[:], want: const u8[:], got: const u8[:]) {
            if !bad {
                print("FAIL ", path, " #", cases, ": ", if input.len > 60 { input[..60] } else { input[..] });
                if find(failed_templates, path) < 0 { failed_templates.append(path); failed_templates.push('\n'); }
            }
            bad = true;
            if verbose { print("  ", what, ":\n    llama.cpp: ", want, "\n    goose:     ", got); }
        }
        if got_ok != ok {
            let mine: u8[>..] = if got_ok { str("ok") } else { str("error: ", chat_apply::err) };
            let theirs: u8[>..] = if ok { str("ok") } else { str("error: ", expected[0]) };
            report("apply", theirs, mine);
        } else if ok {
            var parts_json: u8[>..] = [];
            parts_json.push('[');
            var start = 0;
            for p, k in chat_render::prompt_parts {
                if k > 0 { parts_json.push(','); }
                parts_json.push('[');
                json_schema::json_string(parts_json, chat_render::prompt[start..p.end]);
                parts_json.append(if p.input { ",true]" } else { ",false]" });
                start = p.end;
            }
            parts_json.push(']');
            var triggers: u8[>..] = [];
            triggers.push('[');
            var tstart = 0;
            var tk = 0;
            for i in chat_render::trigger_values.len {
                guard chat_render::trigger_values[i] == 0 else { continue; }
                if tk > 0 { triggers.push(','); }
                format(triggers, "[", chat_render::trigger_types[tk], ",");
                json_schema::json_string(triggers, chat_render::trigger_values[tstart..i]);
                triggers.push(']');
                tstart = i + 1;
                tk++;
            }
            triggers.push(']');
            var preserved: u8[>..] = [];
            strings_json(preserved, chat_render::preserved_tokens);
            var stops: u8[>..] = [];
            strings_json(stops, chat_render::additional_stops);
            var ends: u8[>..] = [];
            strings_json(ends, chat_render::thinking_ends);
            if chat_render::prompt != expected[0] { report("prompt", expected[0], chat_render::prompt); }
            if parts_json != expected[1] { report("prompt parts", expected[1], parts_json); }
            if str(chat_render::format) != expected[2] { report("format", expected[2], str(chat_render::format)); }
            if chat_render::grammar != expected[3] { report("grammar", expected[3], chat_render::grammar); }
            let lazy = if chat_render::grammar_lazy { "1" } else { "0" };
            if lazy != expected[4] { report("grammar_lazy", expected[4], lazy); }
            if chat_render::gen_prompt != expected[5] { report("generation_prompt", expected[5], chat_render::gen_prompt); }
            if triggers != expected[6] { report("triggers", expected[6], triggers); }
            if preserved != expected[7] { report("preserved tokens", expected[7], preserved); }
            if stops != expected[8] { report("additional stops", expected[8], stops); }
            if chat_render::thinking_start != expected[9] { report("thinking start", expected[9], chat_render::thinking_start); }
            if ends != expected[10] { report("thinking ends", expected[10], ends); }
            // the parses
            var e = 12;
            var nparses = 0;
            for c in expected[11] { nparses = nparses * 10 + (c - '0'); }
            for p in nparses {
                var cut = 0;
                for c in expected[e] { cut = cut * 10 + (c - '0'); }
                let partial = expected[e + 1] == "1";
                let pok = expected[e + 2] == "1";
                e += 3;
                let mine_ok = chat_apply::parse(input[..cut], partial);
                var want: u8[>..] = [];
                var got: u8[>..] = [];
                if pok {
                    format(want, "content=", expected[e + 1], " reasoning=", expected[e + 2]);
                    var nc = 0;
                    for c in expected[e + 3] { nc = nc * 10 + (c - '0'); }
                    for q in nc { format(want, " call(", expected[e + 4 + 3 * q], ", ", expected[e + 5 + 3 * q], ", ", expected[e + 6 + 3 * q], ")"); }
                    e += 4 + 3 * nc;
                } else {
                    format(want, "error: ", expected[e]);
                    e += 1;
                }
                if mine_ok {
                    format(got, "content=", chat_peg::content_text, " reasoning=", chat_peg::reasoning_text);
                    for c in chat_peg::calls { format(got, " call(", chat_peg::t(c.name), ", ", chat_peg::t(c.arguments), ", ", chat_peg::t(c.id), ")"); }
                } else {
                    format(got, "error: ", chat_apply::err);
                }
                if mine_ok != pok || (pok && want != got) { report(str("parse of ", cut, if partial { " (partial)" } else { "" }), want, got); }
            }
        }
        if bad { failed++; }
    }
    print(checked, " checked, ", failed, " failed");
    if failed > 0 { print("templates with failures:\n", failed_templates); exit(1); }
}
