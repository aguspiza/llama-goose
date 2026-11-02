// ggml from Goose: a 3x4 by 4x2 matrix product, checked against a plain loop.
import ggml from "../ggml";
import ggml_cpu from "../ggml/cpu";
import ggml_shim from "../ggml/shim";

let K = 4;
let M = 3;
let N = 2;

fn main() {
    // a: M rows of K, b: N rows of K, row-major as ggml stores them.
    var a: f32[>..] = [];
    for i in M * K { a.push(i + 1); }
    var b: f32[>..] = [];
    for i in N * K { b.push(i % 3 - 1.0); }

    let ctx = ggml::init(16 * 1024 * 1024, false);
    let ta = ggml::new_tensor_2d(ctx, ggml::TYPE_F32, K, M);
    let tb = ggml::new_tensor_2d(ctx, ggml::TYPE_F32, K, N);
    ggml::write_f32(ta, a);
    ggml::write_f32(tb, b);
    let tc = ggml::mul_mat(ctx, ta, tb);   // ne = [M, N]
    let g = ggml::new_graph(ctx);
    ggml::build_forward_expand(g, tc);
    let status = ggml::graph_compute_with_ctx(ctx, g, 2);

    var got: f32[] = [0.0; M * N];
    ggml::read_f32(tc, got);
    var ok = status == 0 && ggml::ne(tc, 0) == M && ggml::ne(tc, 1) == N;
    ggml::free(ctx);

    for n in N {
        for m in M {
            var want: f32 = 0.0;
            for k in K { want += a[m * K + k] * b[n * K + k]; }
            if got[n * M + m] != want { ok = false; }
        }
    }
    print("ggml mul_mat ", got, if ok { " ok" } else { " FAILED" });
    if !ok { exit(1); }
}
