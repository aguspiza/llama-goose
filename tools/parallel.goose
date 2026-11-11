// llama.cpp's examples/parallel: a server simulated, with n_parallel
// clients whose requests (a system prompt, then a question) are decoded
// together in one cache unified for all of them (-kvu; the cache split per
// sequence is not ported), new requests joining the batch as others end.
//   parallel -m model.gguf [-c n_ctx] [-np n_parallel] [-ns n_sequences] [-n n_predict]
//            [-b n_batch] [-pps] [-nocb] [--junk n] [--temp t] [-s seed]
// port-of: examples/parallel/parallel.cpp
import std;
import os;
import ggml from "../ggml";
import ggml_cpu from "../ggml/cpu";
import ggml_shim from "../ggml/shim";
import llama from "../llama";
import llama.vocab;
import llama.model;
import llama.context;
import llama.sampler;
import common from "../common";
import common.console;

let K_SYSTEM = """
Transcript of a never ending dialog, where the User interacts with an Assistant.
The Assistant is helpful, kind, honest, good at writing, and never fails to answer the User's requests immediately and with precision.

User:
Recommend a nice restaurant in the area.
Assistant:
I recommend the restaurant "The Golden Duck". It is a 5 star restaurant with a great view of the city. The food is delicious and the service is excellent. The prices are reasonable and the portions are generous. The restaurant is located at 123 Main Street, New York, NY 10001. The phone number is (212) 555-1234. The hours are Monday through Friday from 11:00 am to 10:00 pm. The restaurant is closed on Saturdays and Sundays.
User:
Who is Richard Feynman?
Assistant:
Richard Feynman was an American physicist who is best known for his work in quantum mechanics and particle physics. He was awarded the Nobel Prize in Physics in 1965 for his contributions to the development of quantum electrodynamics. He was a popular lecturer and author, and he wrote several books, including "Surely You're Joking, Mr. Feynman!" and "What Do You Care What Other People Think?".

""";

let K_QUESTIONS = [
    "What is the tallest mountain in the world?",
    "Who was the first person to win two Nobel Prizes?",
    "Which country invented paper?",
    "What organ is primarily responsible for pumping blood throughout the body?",
    "Which planet is known for its prominent ring system?",
    "Who directed the movie 'Inception'?",
    "What is the freezing point of water in Fahrenheit?",
    "Which animal is known to have the longest lifespan?",
    "What language has the most native speakers worldwide?",
    "What is the capital city of Canada?",
    "Who is credited with inventing the World Wide Web?",
    "Which metal is liquid at room temperature?",
    "What is the term for an animal that eats both plants and meat?",
    "Who painted 'The Starry Night'?",
    "What gas do humans exhale that plants use for photosynthesis?",
    "What year did World War II end?",
    "Which continent has the most countries?",
    "Who wrote the novel 'Frankenstein'?",
    "What does DNA stand for?",
    "What is the main ingredient in traditional Japanese miso soup?",
];

let K_ANSWERS = [
    "The tallest mountain in the world is Mount Everest.",
    "Marie Curie was the first person to win two Nobel Prizes.",
    "Paper was invented in China.",
    "The heart is the organ responsible for pumping blood.",
    "Saturn is known for its prominent ring system.",
    "Christopher Nolan directed the movie 'Inception'.",
    "The freezing point of water in Fahrenheit is 32°F.",
    "The bowhead whale is known to have the longest lifespan among mammals.",
    "Mandarin Chinese has the most native speakers in the world.",
    "The capital city of Canada is Ottawa.",
    "Tim Berners-Lee is credited with inventing the World Wide Web.",
    "Mercury is the metal that is liquid at room temperature.",
    "An animal that eats both plants and meat is called an omnivore.",
    "'The Starry Night' was painted by Vincent van Gogh.",
    "Humans exhale carbon dioxide, which plants use in photosynthesis.",
    "World War II ended in 1945.",
    "Africa is the continent with the most countries.",
    "The novel 'Frankenstein' was written by Mary Shelley.",
    "DNA stands for Deoxyribonucleic Acid.",
    "The main ingredient in traditional Japanese miso soup is fermented soybean paste.",
];

let K_PROMPTS = [
    "What is the meaning of life?",
    "Tell me an interesting fact about llamas.",
    "What is the best way to cook a steak?",
    "Are you familiar with the Special Theory of Relativity and can you explain it to me?",
    "Recommend some interesting books to read.",
    "What is the best way to learn a new language?",
    "How to get a job at Google?",
    "If you could have any superpower, what would it be?",
    "I want to learn how to play the piano. What would be the best way to do it?",
];

struct Client {
    id: i64,
    seq_id: i64 = -1,
    sampled: i32 = 0,
    t_start_prompt: f64 = 0.0,
    n_past: i64 = 0,
    n_prompt: i64 = 0,
    n_decoded: i64 = 0,
    i_batch: i64 = -1,
    input: i64 = 0,              // its question, in K_PROMPTS
    response: u8[..65536] = [],
}

var clients: Client[>..] = [];
var chains: sampler::Chain[>..] = [];

fn usage() {
    print("example usage:\n\n    parallel -m model.gguf [-c n_ctx] [-np n_parallel] [-ns n_sequences] [-n n_predict] [-b n_batch] [-pps] [-nocb] [--junk n] [--temp t] [-s seed]\n");
    exit(1);
}

fn log(text: const u8[:]) { write_stderr(text); }

// n right-aligned in w columns, as %<w>d.
fn padded(n: i64, w: i64) -> u8[>..] {
    var digits: u8[>..] = [];
    format(digits, n);
    var out: u8[>..] = [];
    for i in w - digits.len { out.push(' '); }
    out.append(digits);
    return out;
}

// v with two decimals in 5 columns, as %5.2f.
fn fixed2(v: f64) -> u8[>..] {
    var digits: u8[>..] = [];
    format_flt(digits, v, 2);
    var out: u8[>..] = [];
    for i in 5 - digits.len { out.push(' '); }
    out.append(digits);
    return out;
}

fn is_ws(c0: u8) -> bool { c0 == ' ' || c0 == '\t' || c0 == '\n' || c0 == '\r' || c0 == 11 || c0 == 12 }

fn trimmed(s: const u8[:]) -> const u8[:] {
    var a = 0;
    var b = s.len;
    while a < b && is_ws(s[a]) { a++; }
    while b > a && is_ws(s[b - 1]) { b--; }
    return s[a..b];
}

fn main() {
    let args = console::utf8_args();
    var model_path: u8[>..] = [];
    var n_ctx = 4096;
    var n_clients = 1;
    var n_seq = 1;
    var n_predict = 128;
    var n_batch = 2048;
    var cont_batching = true;
    var is_sp_shared = false;
    var n_junk = 1;
    var sp = sampler::Params { .. };
    var i = 1;
    while i < args.len {
        let a = args[i];
        if a == "-kvu" || a == "--kv-unified" { i++; continue; }
        if a == "-pps" { is_sp_shared = true; i++; continue; }
        if a == "-nocb" || a == "--no-cont-batching" { cont_batching = false; i++; continue; }
        if a == "-cb" || a == "--cont-batching" { cont_batching = true; i++; continue; }
        guard i + 1 < args.len else { usage(); }
        let v = args[i + 1];
        if a == "-m" { model_path.clear(); model_path.append(v); }
        else if a == "--temp" || a == "--top-p" || a == "--min-p" {
            let x, ok = parse_flt(v);
            if !ok { usage(); }
            if a == "--temp" { sp.temp = max(x as f32, 0.0); sp.user_set |= sampler::USER_TEMP; }
            else if a == "--top-p" { sp.top_p = x as f32; sp.user_set |= sampler::USER_TOP_P; }
            else { sp.min_p = x as f32; sp.user_set |= sampler::USER_MIN_P; }
        } else {
            let x, ok = parse_int(v);
            if !ok { usage(); }
            if a == "-c" { n_ctx = x; }
            else if a == "-np" { n_clients = x; }
            else if a == "-ns" { n_seq = x; }
            else if a == "-n" { n_predict = x; }
            else if a == "-b" { n_batch = x; }
            else if a == "--junk" { n_junk = x; }
            else if a == "--top-k" { sp.top_k = x; sp.user_set |= sampler::USER_TOP_K; }
            else if a == "-s" || a == "--seed" { sp.seed = x as! u32; }
            else { usage(); }
        }
        i += 2;
    }
    if model_path.len == 0 { usage(); }
    guard n_clients >= 1 && n_clients < 64 else { abort("n_parallel: 1 to 63"); }
    n_junk = max(1, n_junk);
    // the same seed for every client, or, negative, a smaller one for each next client
    let sseed = sp.seed as! i32;

    var rng = sampler::Mt19937 { mt: [0; 624], index: 624 };
    sampler::mt_seed(rng, 1234);

    var m = model::Model { .. };
    model::load(m, model_path);
    var c = context::Context { .. };
    c.unified = true;
    context::init(c, m, n_ctx);
    let n_ubatch = min(512, n_batch);
    sampler::apply_model_defaults(sp, m.meta);

    log("\n\n");
    if sseed >= 0 {
        log(str("main: initializing all samplers with the same RNG seed: ", sseed, " (use a negative seed to have different seeds)\n"));
    } else {
        log(str("main: initializing samplers with different RNG seeds, starting from ", sseed, "\n"));
    }
    for k in n_clients {
        clients.push(Client { id: k, .. });
        var ch = sampler::Chain { .. };
        sampler::init(ch, sp);
        chains.push(ch);
        if sseed < 0 { sp.seed--; }
    }

    var tokens_system: i32[>..] = [];
    vocab::tokenize(K_SYSTEM, true, false, tokens_system);
    let n_tokens_system = tokens_system.len;

    var g_seq_id = 0;
    var batch: i32[>..] = [];
    var pos: i64[>..] = [];
    var seqs: u64[>..] = [];
    var wants: bool[>..] = [];             // whose logits each token's decode keeps
    var logits: f32[>..] = [];
    var outputs: i32[>..] = [];
    var cands: sampler::Cand[>..] = [];
    let n_vocab = vocab::n_tokens();
    var n_total_prompt = 0;
    var n_total_gen = 0;
    let n_cache_miss = 0;
    let t_main_start = clock();

    log("main: Simulating parallel requests from clients:\n");
    log(str("main: n_parallel = ", n_clients, ", n_sequences = ", n_seq, ", cont_batching = ", if cont_batching { 1 } else { 0 },
            ", system tokens = ", n_tokens_system, "\n"));
    log("\n");

    if is_sp_shared {
        log("main: Evaluating the system prompt ...\n");
        for t, j in tokens_system { batch.push(t); pos.push(j); seqs.push(1); }
        context::decode_seqs(c, m, batch, pos, seqs, logits, [(batch.len - 1) as i32], n_ubatch);
        // the system prompt's cells in every client's sequence too
        for k in 1..n_clients + 1 { context::seq_cp(c, 0, k, -1, -1); }
        log("\n");
    }

    log("Processing requests ...\n\n");

    loop {
        batch.clear();
        pos.clear();
        seqs.clear();
        wants.clear();

        // the ongoing sequences' last tokens
        for &cl in clients {
            guard cl.seq_id >= 0 else { continue; }
            cl.i_batch = batch.len;
            batch.push(cl.sampled);
            pos.push(cl.n_past);
            cl.n_past++;
            seqs.push((1 as u64) << (cl.id + 1));
            wants.push(true);
            cl.n_decoded += 1;
        }

        if batch.len == 0 {
            // all sequences have ended: the cache cleared, but for the system prompt
            for k in 1..n_clients + 1 {
                context::seq_rm(c, k, -1, -1);
                context::seq_cp(c, 0, k, -1, -1);
            }
            log("main: clearing the KV cache\n");
        }

        // new sequences
        if cont_batching || batch.len == 0 {
            for &cl, k in clients {
                guard cl.seq_id == -1 && g_seq_id < n_seq else { continue; }
                cl.seq_id = g_seq_id;
                cl.t_start_prompt = clock();
                cl.input = sampler::mt_next(rng) % K_PROMPTS.len as u32;
                cl.response.clear();

                // [system prompt] + [junk] + [user prompt]
                cl.n_past = 0;
                var prompt: u8[>..] = [];
                if is_sp_shared { cl.n_past = n_tokens_system; } else { prompt.append(K_SYSTEM); }
                let n_junk_cur: i64 = sampler::mt_next(rng) % n_junk as u32;
                for j in n_junk_cur {
                    let r: i64 = sampler::mt_next(rng) % K_QUESTIONS.len as u32;
                    prompt.append(str("User:\n", K_QUESTIONS[r], "\nAssistant:\n ", K_ANSWERS[r], "\n"));
                }
                prompt.append(str("User:\n", K_PROMPTS[cl.input], "\nAssistant:\n"));

                sampler::reset(chains[k]);

                // no BOS: the system prompt comes first
                var tokens_prompt: i32[>..] = [];
                vocab::tokenize(prompt, false, false, tokens_prompt);
                for t in tokens_prompt {
                    batch.push(t);
                    pos.push(cl.n_past);
                    cl.n_past++;
                    seqs.push((1 as u64) << (cl.id + 1));
                    wants.push(false);
                }
                // the logits of the last token only
                if batch.len > 0 { wants[batch.len - 1] = true; }
                cl.n_prompt = tokens_prompt.len;
                cl.n_decoded = 0;
                cl.i_batch = batch.len - 1;

                log(str("\x1b[31mClient ", padded(cl.id, 3), ", seq ", padded(cl.seq_id, 4), ", junk = ", padded(n_junk_cur, 4),
                        ", prompt = ", cl.n_prompt, ", started decoding ...\x1b[0m\n"));
                g_seq_id += 1;
            }
        }

        if batch.len == 0 { break; }

        // in chunks of n_batch
        var start = 0;
        while start < batch.len {
            let n_tokens = min(n_batch, batch.len - start);
            outputs.clear();
            for j in n_tokens { if wants[start + j] { outputs.push(j as i32); } }
            context::decode_seqs(c, m, batch[start..start + n_tokens], pos[start..start + n_tokens], seqs[start..start + n_tokens],
                                 logits, outputs, n_ubatch);

            for &cl, k in clients {
                guard cl.i_batch >= start && cl.i_batch < start + n_tokens else { continue; }
                // its row: how many outputs come before its token
                var row = 0;
                while outputs[row] != cl.i_batch - start { row++; }
                let id = sampler::sample(chains[k], cands, logits[row * n_vocab..(row + 1) * n_vocab]);

                var token_str: u8[>..] = [];
                vocab::token_to_piece(id, token_str, 0, true);
                guard cl.response.len + token_str.len <= cl.response.cap else { abort("a response longer than 64 KB"); }
                cl.response.append(token_str);
                cl.sampled = id;

                let found = find(cl.response, "User:");
                if cl.n_decoded > 2 && (vocab::is_eog(id) || (n_predict > 0 && cl.n_decoded >= n_predict) || found >= 0) {
                    // a basic reverse prompt
                    if found >= 0 { cl.response.resize(found); }
                    // the generated part forgotten, the system prompt kept
                    context::seq_rm(c, cl.id + 1, -1, -1);
                    context::seq_cp(c, 0, cl.id + 1, -1, -1);

                    let dt = clock() - cl.t_start_prompt;
                    log(str("\x1b[31mClient ", padded(cl.id, 3), ", seq ", padded(cl.seq_id, 3), "/", padded(n_seq, 3),
                            ", prompt ", padded(cl.n_prompt, 4), " t, response ", padded(cl.n_decoded, 4), " t, time ", fixed2(dt),
                            " s, speed ", fixed2((cl.n_prompt + cl.n_decoded) / dt), " t/s, cache miss ", n_cache_miss,
                            " \x1b[0m \n\nInput:    ", trimmed(K_PROMPTS[cl.input]), "\n\x1b[35mResponse: ", trimmed(cl.response), "\x1b[0m\n\n"));
                    n_total_prompt += cl.n_prompt;
                    n_total_gen += cl.n_decoded;
                    cl.seq_id = -1;
                }
                cl.i_batch = -1;
            }
            start += n_tokens;
        }
    }

    let dt = clock() - t_main_start;
    log("\n");
    log(str("main: n_parallel = ", n_clients, ", n_sequences = ", n_seq, ", cont_batching = ", if cont_batching { 1 } else { 0 },
            ", system tokens = ", n_tokens_system, "\n"));
    log("External prompt file: \x1b[32mused built-in defaults\x1b[0m\n");
    log(str("Model and path used:  \x1b[32m", model_path, "\x1b[0m\n\n"));
    log(str("Total prompt tokens: ", padded(n_total_prompt, 6), ", speed: ", fixed2(n_total_prompt / dt), " t/s\n"));
    log(str("Total gen tokens:    ", padded(n_total_gen, 6), ", speed: ", fixed2(n_total_gen / dt), " t/s\n"));
    log(str("Total speed (AVG):   ", "      ", "  speed: ", fixed2((n_total_prompt + n_total_gen) / dt), " t/s\n"));
    log(str("Cache misses:        ", padded(n_cache_miss, 6), "\n"));
    log("\n");
    write_stdout("\n\n");
}
