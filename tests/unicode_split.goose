// The pre-tokenizer splits on a few texts, against what llama.cpp gives.
import std;
import llama from "../llama";
import llama.unicode;

fn check(name: const u8[:], text: const u8[:], regexes: const u8[:][:], want: const u8[:][:]) -> bool {
    var words: u8[>..] = [];
    var ends: i64[>..] = [];
    unicode::regex_split(text, regexes, false, words, ends);
    var ok = ends.len == want.len;
    var start = 0;
    for e, i in ends {
        if ok && words[start..e] != want[i] { ok = false; }
        start = e;
    }
    if !ok {
        var got: u8[>..] = [];
        start = 0;
        for e in ends { format(got, "[", words[start..e], "]"); start = e; }
        print("FAILED ", name, ": ", got);
    }
    return ok;
}

fn main() {
    let gpt2 = ["'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"];
    let llama3 = ["(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"];
    // falcon's split goes through the general matcher
    let falcon = ["[\\p{P}\\$\\+<=>\\^~\\|`]+",
                  "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
                  "[0-9][0-9][0-9]"];
    var ok = true;
    ok = check("gpt2", "Hello world's  123 !!", gpt2, ["Hello", " world", "'s", " ", " 123", " !!"]) && ok;
    ok = check("llama3", "I'M 12345 ok\n\n x", llama3, ["I", "'M", " ", "123", "45", " ok", "\n\n", " x"]) && ok;
    ok = check("falcon", "a+b=1234, ok", falcon, ["a", "+", "b", "=", "123", "4", ",", " ok"]) && ok;
    print(if ok { "unicode splits ok" } else { "unicode splits FAILED" });
    if !ok { exit(1); }
}
