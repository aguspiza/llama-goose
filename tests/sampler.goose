// sampler: the top k candidates, taken without sorting the others, are the
// first k of all of them sorted, also where logits are equal.
import std;
import ggml from "../ggml";
import ggml_cpu from "../ggml/cpu";
import ggml_shim from "../ggml/shim";
import ggml.gguf;
import llama from "../llama";
import llama.sampler;

var failed = 0;

fn main() {
    var rng = sampler::Mt19937 { mt: [0; 624], index: 624 };
    sampler::mt_seed(rng, 7);
    for n in [1, 5, 40, 300, 5000] {
        for k in [1, 3, 40, 200, 5000] {
            guard k <= n else { continue; }
            // few values: many equal logits
            var all: sampler::Cand[>..] = [];
            for i in n { all.push(sampler::Cand { id: i as i32, logit: sampler::mt_next(rng) % 50, p: 0.0 }); }
            var top: sampler::Cand[>..] = [];
            top.append(all);
            sampler::sort_desc(all);
            sampler::top_k_desc(top, k);
            for i in k {
                if top[i].id != all[i].id || top[i].logit != all[i].logit {
                    failed++;
                    print("FAIL n ", n, " k ", k, " at ", i, ": token ", top[i].id, ", not ", all[i].id);
                    break;
                }
            }
        }
    }
    if failed > 0 { print(failed, " failed"); exit(1); }
    print("sampler: all passed");
}
