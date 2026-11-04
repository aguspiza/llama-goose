// common/jdoc: documents parsed, read and written back as nlohmann does.
import std;
import common from "../common";
import common.jdoc;

var failed = 0;

fn check(name: const u8[:], ok: bool) {
    if !ok { failed++; print("FAIL ", name); }
}

fn dumped(src: const u8[:]) -> u8[>..] {
    var out: u8[>..] = [];
    jdoc::dump(out, jdoc::parse(src));
    return out;
}

fn str_out(s: const u8[:]) -> u8[>..] {
    var out: u8[>..] = [];
    jdoc::put_str(out, s);
    return out;
}

fn f32_out(v: f32) -> u8[>..] {
    var out: u8[>..] = [];
    jdoc::put_f32(out, v);
    return out;
}

fn main() {
    let src = "{\"a\": [1, 2.5, \"x\\ny\", true, null], \"b\": {\"c\": -3}, \"e\": 1e-7, \"f\": 100.0}";
    check("round trip", dumped(src) == "{\"a\":[1,2.5,\"x\\ny\",true,null],\"b\":{\"c\":-3},\"e\":1e-07,\"f\":100.0}");
    let v = jdoc::parse("{\"n\": 5, \"s\": \"hi\", \"z\": null, \"t\": true, \"x\": 1.5, \"o\": {\"k\": [3, 4]}}");
    check("get_int", jdoc::get_int(v, "n", 0) == 5);
    check("get_str", jdoc::get_str(v, "s", "") == "hi");
    check("default", jdoc::get_str(v, "missing", "d") == "d");
    check("null is absent", !jdoc::has(v, "z") && jdoc::contains(v, "z"));
    check("bool", jdoc::get_bool(v, "t", false));
    check("num", jdoc::get_num(v, "x", 0.0) == 1.5);
    let o = jdoc::get(v, "o");
    let arr = jdoc::get(o, "k");
    check("array", jdoc::len(arr) == 2 && jdoc::int_of(jdoc::at(arr, 1)) == 4);
    check("key", jdoc::key(v, 1) == "s");
    check("bad json", jdoc::parse("not json") < 0);

    // floats widened to doubles, as nlohmann writes the float fields
    check("0.8f", f32_out(0.8) == "0.800000011920929");
    check("0.95f", f32_out(0.95) == "0.949999988079071");
    check("1.75f", f32_out(1.75) == "1.75");
    check("1.0f", f32_out(1.0) == "1.0");
    check("-1.0f", f32_out(-1.0) == "-1.0");

    check("controls", str_out("a\x01\tb") == "\"a\\u0001\\tb\"");
    check("utf-8 kept", str_out("caf\xC3\xA9 \xF0\x9F\x98\x8A") == "\"caf\xC3\xA9 \xF0\x9F\x98\x8A\"");
    check("cut sequence", str_out("a\xC3") == "\"a\xEF\xBF\xBD\"");
    check("bad byte", str_out("a\xFFb") == "\"a\xEF\xBF\xBDb\"");
    check("broken sequence", str_out("\xE2\x82x") == "\"\xEF\xBF\xBDx\"");
    check("surrogate", str_out("\xED\xA0\x80") == "\"\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD\"");

    if failed > 0 { print(failed, " failed"); exit(1); }
    print("jdoc: all passed");
}
