// dspark-check: correctness / acceptance harness for DeepSeek-V4.1 DSpark speculative decoding.
//
// Uses the normal llama.cpp CLI flags (model, -ngl, -sm, -ts, -lm, -lzm, -fa, -c, -b, -ub,
// --spec-type draft-dspark --spec-draft-model ...). Test selection via env:
//
//   DSC_PROMPTS=file   prompts separated by a line "-----" (raw completion text)
//   DSC_N=200          greedy tokens per prompt
//   DSC_TESTS=ref,full,rb,spec   which runs to do (default: all)
//     ref  : sequential single-token greedy decode (reference tokens + logits)
//     seq  : a second sequential run (determinism check)
//     full : target-only verification batches of K+1 with ALL-CORRECT drafts (batch numerics, no rollback)
//     rb   : verification batches with one corrupted draft token (rollback of raw/compressed/engram state)
//     spec : real DSpark speculation (draft model), greedy, compared with ref
//   DSC_K=5            draft length for full/rb
//
// For every emitted token of a test run the logits row is compared with the reference row at the
// same position: max |diff|, and argmax agreement. The first token that differs from the
// reference stops that run (later positions have different context).
#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

static int env_int(const char * k, int d) { const char * v = getenv(k); return v && *v ? atoi(v) : d; }
static std::string env_str(const char * k, const char * d) { const char * v = getenv(k); return v && *v ? v : d; }

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static int argmax(const float * x, int n) {
    int b = 0;
    for (int i = 1; i < n; ++i) if (x[i] > x[b]) b = i;
    return b;
}

static float margin(const float * x, int n) {
    int b = argmax(x, n);
    float s = -INFINITY;
    for (int i = 0; i < n; ++i) if (i != b && x[i] > s) s = x[i];
    return x[b] - s;
}

struct ref_run {
    std::vector<llama_token> toks;          // greedy tokens g[0..N-1]; g[0] is predicted from the prompt
    std::vector<std::vector<float>> logits; // logits row that produced g[i]
};

static double kl_div(const float * a, const float * b, int n) { // KL(softmax(a) || softmax(b))
    double ma = -INFINITY, mb = -INFINITY;
    for (int k = 0; k < n; ++k) { ma = std::max(ma, (double) a[k]); mb = std::max(mb, (double) b[k]); }
    double za = 0, zb = 0;
    for (int k = 0; k < n; ++k) { za += std::exp(a[k] - ma); zb += std::exp(b[k] - mb); }
    const double la = ma + std::log(za), lb = mb + std::log(zb);
    double kl = 0;
    for (int k = 0; k < n; ++k) { const double lpa = a[k] - la; kl += std::exp(lpa) * (lpa - (b[k] - lb)); }
    return std::max(kl, 0.0);
}

struct cmp_stats {
    int n_cmp = 0, n_argmax_diff = 0, first_div = -1;
    double max_diff = 0, sum_max_diff = 0, sum_kl = 0, max_kl = 0;
    void add(int i, const float * a, const float * b, int n_vocab, bool same_tok) {
        double md = 0;
        for (int k = 0; k < n_vocab; ++k) md = std::max(md, (double) std::fabs(a[k] - b[k]));
        const double kl = kl_div(a, b, n_vocab);
        sum_kl += kl; max_kl = std::max(max_kl, kl);
        max_diff = std::max(max_diff, md); sum_max_diff += md; n_cmp++;
        if (!same_tok) { n_argmax_diff++; if (first_div < 0) first_div = i; }
    }
};

// ---- tensor capture (bisect mode) via the graph eval callback
struct cap_entry { std::string name; int64_t ne[4]; std::vector<float> data; };
struct capture {
    bool on = false;
    std::vector<cap_entry> entries;
    std::map<std::string, int> seen;
    void reset() { entries.clear(); seen.clear(); }
};
static bool cap_cb(struct ggml_tensor * t, bool ask, void * ud) {
    auto * c = (capture *) ud;
    if (ask) return c->on;
    if (!c->on) return true;
    if (!(t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_I32) ||
            !ggml_is_contiguous(t) || ggml_nelements(t) > (int64_t) 64*1024*1024) return true;
    cap_entry e;
    const std::string base = t->name;
    e.name = base + "#" + std::to_string(c->seen[base]++);
    for (int i = 0; i < 4; ++i) e.ne[i] = t->ne[i];
    e.data.resize(ggml_nelements(t));
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, e.data.data(), 0, ggml_nbytes(t));
    } else {
        std::vector<uint8_t> raw(ggml_nbytes(t));
        ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
        for (size_t k = 0; k < e.data.size(); ++k) {
            e.data[k] = t->type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(((const ggml_fp16_t *) raw.data())[k])
                                                 : (float) ((const int32_t *) raw.data())[k];
        }
    }
    c->entries.push_back(std::move(e));
    return true;
}

// compare token row `row` of the batch capture with the single-token capture
static void cap_compare(const capture & single, const capture & batch, int n_tok_batch, int row, double thr, int max_print) {
    std::map<std::string, const cap_entry *> bm;
    for (auto & e : batch.entries) bm[e.name] = &e;
    int printed = 0, n_cmp = 0;
    const std::string skip = env_str("DSC_BISECT_SKIP", "");
    for (auto & a : single.entries) {
        if (!skip.empty() && a.name.find(skip) != std::string::npos) continue;
        auto it = bm.find(a.name);
        if (it == bm.end()) continue;
        const cap_entry & b = *it->second;
        int tdim = -1; bool ok = true;
        for (int d = 0; d < 4; ++d) {
            if (a.ne[d] == b.ne[d]) continue;
            if (a.ne[d] == 1 && b.ne[d] == n_tok_batch && tdim < 0) { tdim = d; continue; }
            ok = false;
        }
        if (!ok) continue;
        if (tdim < 0 && n_tok_batch > 1) continue; // no token dim: state/scratch buffers, not comparable per row
        // strides of b
        int64_t nb[4] = { 1, b.ne[0], b.ne[0]*b.ne[1], b.ne[0]*b.ne[1]*b.ne[2] };
        double num = 0, den = 0, mx = 0;
        size_t k = 0;
        for (int64_t i3 = 0; i3 < a.ne[3]; ++i3)
        for (int64_t i2 = 0; i2 < a.ne[2]; ++i2)
        for (int64_t i1 = 0; i1 < a.ne[1]; ++i1)
        for (int64_t i0 = 0; i0 < a.ne[0]; ++i0, ++k) {
            int64_t idx[4] = { i0, i1, i2, i3 };
            if (tdim >= 0) idx[tdim] = row;
            const float vb = b.data[idx[0]*nb[0] + idx[1]*nb[1] + idx[2]*nb[2] + idx[3]*nb[3]];
            const float va = a.data[k];
            if (!std::isfinite(va) || !std::isfinite(vb)) {
                if (std::isfinite(va) != std::isfinite(vb) || (va != vb)) { mx = INFINITY; num += 1.0; }
                continue;
            }
            num += (double)(va - vb)*(va - vb); den += (double) va*va; mx = std::max(mx, (double) std::fabs(va - vb));
        }
        n_cmp++;
        const double rel = std::sqrt(num / (den + 1e-30));
        if ((rel > thr || mx > 1e-2) && printed < max_print) {
            LOG_INF("[bisect] %-28s ne=[%lld,%lld,%lld,%lld] tdim=%d rel=%.3e max=%.3e\n", a.name.c_str(),
                    (long long) a.ne[0], (long long) a.ne[1], (long long) a.ne[2], (long long) a.ne[3], tdim, rel, mx);
            printed++;
        }
    }
    LOG_INF("[bisect] compared %d tensors (single %zu, batch %zu)\n", n_cmp, single.entries.size(), batch.entries.size());
}

static void prefill(llama_context * ctx, const std::vector<llama_token> & p, int n_batch, common_speculative * spec) {
    llama_memory_clear(llama_get_memory(ctx), true);
    // everything but the last prompt token; the last one is the first "id_last"
    const int n = (int) p.size() - 1;
    for (int i0 = 0; i0 < n; i0 += n_batch) {
        const int m = std::min(n_batch, n - i0);
        llama_batch b = llama_batch_init(m, 0, 1);
        for (int i = 0; i < m; ++i) common_batch_add(b, p[i0 + i], i0 + i, { 0 }, false);
        if (llama_decode(ctx, b)) { LOG_ERR("prefill failed\n"); exit(1); }
        if (spec && !common_speculative_process(spec, b)) { LOG_ERR("spec process failed\n"); exit(1); }
        llama_batch_free(b);
    }
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) return 1;

    const int N = env_int("DSC_N", 200);
    const int K = env_int("DSC_K", 5);
    const std::string tests = env_str("DSC_TESTS", "ref,full,rb,spec");
    auto want = [&](const char * t) { return ("," + tests + ",").find(std::string(",") + t + ",") != std::string::npos; };

    std::vector<std::string> prompts;
    {
        std::ifstream f(env_str("DSC_PROMPTS", ""));
        if (!f) { LOG_ERR("set DSC_PROMPTS\n"); return 1; }
        std::string line, cur;
        while (std::getline(f, line)) {
            if (line == "-----") { if (!cur.empty()) prompts.push_back(cur); cur.clear(); continue; }
            cur += line + "\n";
        }
        if (!cur.empty()) prompts.push_back(cur);
        if (env_int("DSC_STRIP_NL", 0)) {
            for (auto & q : prompts) {
                while (!q.empty() && q.back() == '\n') q.pop_back();
                size_t a = 0; while (a < q.size() && q[a] == '\n') ++a;
                q = q.substr(a);
            }
        }
    }

    if (getenv("DSC_K")) {
        params.speculative.draft.n_max = K; // draft length follows DSC_K
    }
    const bool use_spec = want("spec") && !params.speculative.types.empty();
    if (use_spec) {
        const auto lim = common_speculative_get_output_limits(params.n_batch, params.n_parallel, common_speculative_n_max(&params.speculative));
        params.n_outputs_max = lim.total;
        params.n_outputs_max_per_seq = lim.per_seq;
    }

    static capture cap;
    if (want("bisect") || want("cachebis")) {
        params.cb_eval = cap_cb;
        params.cb_eval_user_data = &cap;
        params.warmup = false;
    }
    llama_backend_init();
    llama_numa_init(params.numa);
    auto init_tgt = common_init_from_params(params);
    llama_model * model = init_tgt->model();
    llama_context * ctx = init_tgt->context();
    if (!model || !ctx) { LOG_ERR("init failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_batch = llama_n_batch(ctx);

    common_speculative_init_result_ptr spec_init;
    common_speculative * spec = nullptr;
    llama_context * ctx_dft = nullptr;
    if (use_spec) {
        common_params params_dft = common_base_params_to_speculative(params);
        spec_init = common_speculative_init_from_params(params_dft, model, ctx);
        params.speculative.draft.ctx_tgt = ctx;
        params.speculative.draft.ctx_dft = spec_init->context();
        ctx_dft = spec_init->context();
    }
    LOG_INF("dspark-check: n_rs_seq(ctx_tgt) = %u, seq_rm type = %d\n", llama_n_rs_seq(ctx), (int) common_context_can_seq_rm(ctx));
    if (use_spec) {
        spec = common_speculative_init(params.speculative, 1);
        if (!spec) { LOG_ERR("spec init failed\n"); return 1; }
    }

    std::mt19937 rng(1234);
    llama_batch bt = llama_batch_init(std::max(n_batch, K + 2), 0, 1);

    for (size_t ip = 0; ip < prompts.size(); ++ip) {
        const auto p = common_tokenize(ctx, prompts[ip], true, true);
        LOG_INF("\n===== prompt %zu: %zu tokens\n", ip, p.size());
        const int n0 = (int) p.size() - 1; // position of id_last

        // ---------------- ref: sequential greedy
        ref_run ref;
        auto run_seq = [&](ref_run & out, const char * name) {
            prefill(ctx, p, n_batch, nullptr);
            llama_token last = p.back();
            const double t0 = now_s();
            for (int i = 0; i < N; ++i) {
                common_batch_clear(bt);
                common_batch_add(bt, last, n0 + i, { 0 }, true);
                if (llama_decode(ctx, bt)) { LOG_ERR("decode failed\n"); exit(1); }
                const float * l = llama_get_logits_ith(ctx, 0);
                out.logits.emplace_back(l, l + n_vocab);
                last = argmax(l, n_vocab);
                out.toks.push_back(last);
                if (llama_vocab_is_eog(vocab, last)) break;
            }
            const double dt = now_s() - t0;
            LOG_INF("[%s] %zu tokens in %.2f s = %.2f tok/s\n", name, out.toks.size(), dt, out.toks.size() / dt);
        };
        run_seq(ref, "ref");
        {
            std::string s; for (auto t : ref.toks) s += common_token_to_piece(ctx, t);
            LOG_INF("[ref] text: %s\n", s.substr(0, 300).c_str());
        }
        const int NR = (int) ref.toks.size();

        auto report = [&](const char * name, const cmp_stats & st, int n_emit, double dt, int n_pass, int n_drafted, int n_acc) {
            LOG_INF("[%s] emitted %d/%d, identical_prefix=%d, argmax_diff_rows=%d, max|dlogit|=%.4g, mean max|dlogit|=%.4g, "
                    "KL mean=%.3g max=%.3g, passes=%d, drafted=%d, accepted=%d (%.1f%%), acc/pass=%.2f, %.2f tok/s, %.1f ms/pass\n",
                    name, n_emit, NR, st.first_div < 0 ? n_emit : st.first_div, st.n_argmax_diff, st.max_diff,
                    st.n_cmp ? st.sum_max_diff / st.n_cmp : 0.0, st.n_cmp ? st.sum_kl / st.n_cmp : 0.0, st.max_kl, n_pass, n_drafted, n_acc,
                    n_drafted ? 100.0 * n_acc / n_drafted : 0.0, n_pass ? (double) n_acc / n_pass : 0.0, n_emit / dt, n_pass ? 1e3 * dt / n_pass : 0.0);
            if (st.first_div >= 0) {
                const float * r = ref.logits[st.first_div].data();
                LOG_INF("[%s]   first divergence at token %d: ref margin top1-top2 = %.4g\n", name, st.first_div, margin(r, n_vocab));
            }
        };

        if (want("bisect")) {
            // single-token decode of step S vs row 0 of a K+1 verification batch at the same position
            const int S = std::min(env_int("DSC_BISECT_STEP", 2), NR - K - 1);
            capture c_batch;
            const bool prefix_batches = env_int("DSC_BISECT_PREFIX_BATCH", 0) != 0;
            auto run_to = [&](int steps) {
                prefill(ctx, p, n_batch, nullptr);
                llama_token last = p.back();
                if (prefix_batches) {
                    // build the cache the way verification does: K+1 oracle batches
                    int i = 0;
                    while (i < steps) {
                        const int k = std::min(K, steps - i - 1);
                        common_batch_clear(bt);
                        common_batch_add(bt, last, n0 + i, { 0 }, true);
                        for (int j = 0; j < k; ++j) common_batch_add(bt, ref.toks[i + j], n0 + i + 1 + j, { 0 }, true);
                        if (llama_decode(ctx, bt)) exit(1);
                        i += k + 1;
                        last = ref.toks[i - 1];
                    }
                    return last;
                }
                for (int i = 0; i < steps; ++i) {
                    common_batch_clear(bt);
                    common_batch_add(bt, last, n0 + i, { 0 }, true);
                    if (llama_decode(ctx, bt)) exit(1);
                    last = ref.toks[i];
                }
                return last;
            };
            // single-token captures for steps S..S+K, then the K+1 batch at S; compare row j with step S+j
            std::vector<capture> c_singles;
            std::vector<std::vector<float>> l_singles;
            llama_token last = run_to(S);
            for (int j = 0; j <= K; ++j) {
                cap.reset(); cap.on = true;
                common_batch_clear(bt);
                common_batch_add(bt, last, n0 + S + j, { 0 }, true);
                if (llama_decode(ctx, bt)) exit(1);
                cap.on = false; c_singles.push_back(cap);
                const float * ls = llama_get_logits_ith(ctx, 0);
                l_singles.emplace_back(ls, ls + n_vocab);
                last = ref.toks[S + j];
            }
            last = run_to(S);
            cap.reset(); cap.on = true;
            common_batch_clear(bt);
            common_batch_add(bt, last, n0 + S, { 0 }, true);
            for (int j = 0; j < K; ++j) common_batch_add(bt, ref.toks[S + j], n0 + S + 1 + j, { 0 }, true);
            if (llama_decode(ctx, bt)) exit(1);
            cap.on = false; c_batch = cap;
            for (int j = 0; j <= K; ++j) {
                const float * lb = llama_get_logits_ith(ctx, j);
                double mx = 0; for (int k = 0; k < n_vocab; ++k) mx = std::max(mx, (double) std::fabs(lb[k] - l_singles[j][k]));
                LOG_INF("[bisect] ===== row %d (step %d): logits max|d| single-vs-batch = %.4g\n", j, S + j, mx);
                if (mx > 0 || env_int("DSC_BISECT_ALL", 0)) {
                    cap_compare(c_singles[j], c_batch, K + 1, j, env_int("DSC_BISECT_THR_E6", 1000) * 1e-6, env_int("DSC_BISECT_PRINT", 400));
                }
            }
        }

        std::vector<int> cb_steps;
        {
            std::stringstream ss(env_str("DSC_BISECT_STEPS", ""));
            std::string tok;
            while (std::getline(ss, tok, '/')) if (!tok.empty()) cb_steps.push_back(atoi(tok.c_str()));
            if (cb_steps.empty()) cb_steps.push_back(env_int("DSC_BISECT_STEP", 2));
        }
        for (int S_req : cb_steps) if (want("cachebis")) {
            // same single-token decode at step S on two caches: built by single steps vs by K+1 oracle batches
            const int S = std::min(S_req, NR - 1);
            capture cA, cB;
            prefill(ctx, p, n_batch, nullptr);
            llama_token last = p.back();
            for (int i = 0; i < S; ++i) {
                common_batch_clear(bt);
                common_batch_add(bt, last, n0 + i, { 0 }, true);
                if (llama_decode(ctx, bt)) exit(1);
                last = ref.toks[i];
            }
            cap.reset(); cap.on = true;
            common_batch_clear(bt);
            common_batch_add(bt, last, n0 + S, { 0 }, true);
            if (llama_decode(ctx, bt)) exit(1);
            cap.on = false; cA = cap;
            std::vector<float> lA(llama_get_logits_ith(ctx, 0), llama_get_logits_ith(ctx, 0) + n_vocab);

            prefill(ctx, p, n_batch, nullptr);
            last = p.back();
            int i = 0;
            while (i < S) {
                const int k = std::min(K, S - i - 1);
                common_batch_clear(bt);
                common_batch_add(bt, last, n0 + i, { 0 }, true);
                for (int j = 0; j < k; ++j) common_batch_add(bt, ref.toks[i + j], n0 + i + 1 + j, { 0 }, true);
                if (llama_decode(ctx, bt)) exit(1);
                i += k + 1;
                last = ref.toks[i - 1];
            }
            cap.reset(); cap.on = true;
            common_batch_clear(bt);
            common_batch_add(bt, last, n0 + S, { 0 }, true);
            if (llama_decode(ctx, bt)) exit(1);
            cap.on = false; cB = cap;
            double mx = 0; const float * lB = llama_get_logits_ith(ctx, 0);
            for (int k = 0; k < n_vocab; ++k) mx = std::max(mx, (double) std::fabs(lB[k] - lA[k]));
            LOG_INF("[bisect] cachebis step %d: logits max|d| (single-built cache vs batch-built cache) = %.4g\n", S, mx);
            cap_compare(cA, cB, 1, 0, env_int("DSC_BISECT_THR_E6", 0) * 1e-6, env_int("DSC_BISECT_PRINT", 40));
        }

        if (want("seq")) {
            ref_run r2; run_seq(r2, "seq");
            cmp_stats st;
            for (int i = 0; i < (int) std::min(r2.toks.size(), ref.toks.size()); ++i) {
                st.add(i, ref.logits[i].data(), r2.logits[i].data(), n_vocab, r2.toks[i] == ref.toks[i]);
                if (r2.toks[i] != ref.toks[i]) break;
            }
            report("seq", st, (int) r2.toks.size(), 1, 0, 0, 0);
        }

        // ---------------- target-only verification with oracle drafts
        auto run_verify = [&](const char * name, bool corrupt, int K) {
            prefill(ctx, p, n_batch, nullptr);
            llama_token last = p.back();
            int i = 0; // number of emitted tokens == ref index of the next token
            cmp_stats st; int n_pass = 0, n_drafted = 0, n_acc = 0, cyc = 0;
            const double t0 = now_s();
            while (i < NR && st.first_div < 0) {
                const int k = std::min(K, NR - i - 1);
                std::vector<llama_token> draft(ref.toks.begin() + i, ref.toks.begin() + i + std::max(k, 0));
                if (corrupt && k > 0) {
                    const int m = cyc++ % (k + 1); // corrupt draft[m] (m == k: no corruption)
                    if (m < k) draft[m] = (draft[m] + 1 + rng() % 1000) % n_vocab;
                }
                common_batch_clear(bt);
                common_batch_add(bt, last, n0 + i, { 0 }, true);
                for (int j = 0; j < (int) draft.size(); ++j) common_batch_add(bt, draft[j], n0 + i + 1 + j, { 0 }, true);
                if (llama_decode(ctx, bt)) { LOG_ERR("decode failed\n"); exit(1); }
                n_pass++; n_drafted += draft.size();
                int j = 0;
                for (;; ++j) {
                    const float * l = llama_get_logits_ith(ctx, j);
                    const llama_token t = argmax(l, n_vocab);
                    st.add(i, ref.logits[i].data(), l, n_vocab, t == ref.toks[i]);
                    if (t != ref.toks[i]) break;
                    last = t; i++;
                    if (j >= (int) draft.size() || draft[j] != t || i >= NR) break;
                    n_acc++;
                }
                // drop everything after the last committed position (n0 + i - 1 holds the last fed token)
                llama_memory_seq_rm(llama_get_memory(ctx), 0, n0 + i, -1);
            }
            report(name, st, i, now_s() - t0, n_pass, n_drafted, n_acc);
        };
        if (want("full")) run_verify("full", false, K);
        if (want("rb"))   run_verify("rb",   true, K);
        if (want("ksweep")) {
            // verification cost per pass vs batch width (oracle drafts, all accepted)
            for (int k = 0; k <= K; ++k) {
                const std::string nm = "ksweep k=" + std::to_string(k);
                run_verify(nm.c_str(), false, k);
            }
        }

        // ---------------- real DSpark speculation (greedy target sampling)
        if (use_spec) {
            // the spec hook seeds the draft KV with the prompt features
            llama_memory_clear(llama_get_memory(ctx_dft), true);
            prefill(ctx, p, n_batch, spec);
            llama_tokens prompt_tgt(p.begin(), p.end() - 1);
            common_speculative_begin(spec, 0, prompt_tgt);
            llama_token last = p.back();
            int i = 0; cmp_stats st; int n_pass = 0, n_drafted = 0, n_acc = 0;
            std::vector<int> hist(K + 2, 0);
            double t_draft = 0, t_verify = 0, t_target = 0, t_inject = 0;
            llama_tokens draft;
            const double t0 = now_s();
            const bool cont = env_int("DSC_CONTINUE", 0) != 0;
            while (i < NR && (cont || st.first_div < 0)) {
                draft.clear();
                const double td = now_s();
                common_speculative_get_draft_params(spec, 0) = {
                    /* .drafting = */ true,
                    /* .n_max    = */ std::min(K, NR - i - 1),
                    /* .n_past   = */ n0 + i,
                    /* .id_last  = */ last,
                    /* .prompt   = */ &prompt_tgt,
                    /* .result   = */ &draft,
                };
                common_speculative_draft(spec);
                if ((int) draft.size() > NR - i - 1) draft.resize(std::max(0, NR - i - 1));
                llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, n0 + i, -1);
                t_draft += now_s() - td;

                const double tv = now_s();
                common_batch_clear(bt);
                common_batch_add(bt, last, n0 + i, { 0 }, true);
                for (int j = 0; j < (int) draft.size(); ++j) common_batch_add(bt, draft[j], n0 + i + 1 + j, { 0 }, true);
                if (llama_decode(ctx, bt)) { LOG_ERR("decode failed\n"); exit(1); }
                llama_synchronize(ctx);
                const double tinj = now_s();
                t_target += tinj - tv;
                if (!common_speculative_process(spec, bt)) { LOG_ERR("spec process failed\n"); exit(1); }
                llama_synchronize(ctx_dft);
                t_inject += now_s() - tinj;
                n_pass++; n_drafted += draft.size();
                int j = 0, acc = 0;
                for (;; ++j) {
                    const float * l = llama_get_logits_ith(ctx, j);
                    const llama_token t = argmax(l, n_vocab);
                    if (st.first_div < 0) st.add(i, ref.logits[i].data(), l, n_vocab, t == ref.toks[i]);
                    prompt_tgt.push_back(last);
                    last = t; i++;
                    if (!cont && t != ref.toks[i - 1]) break;
                    if (j >= (int) draft.size() || draft[j] != t || i >= NR) break;
                    acc++;
                }
                n_acc += acc; hist[acc]++;
                common_speculative_accept(spec, 0, acc);
                llama_memory_seq_rm(llama_get_memory(ctx), 0, n0 + i, -1);
                llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, n0 + i, -1);
                t_verify += now_s() - tv;
            }
            report("spec", st, i, now_s() - t0, n_pass, n_drafted, n_acc);
            std::string h; for (int a = 0; a <= K; ++a) h += std::to_string(hist[a]) + " ";
            LOG_INF("[spec]   accepted-per-pass histogram (0..%d): %s| draft %.1f ms/pass, verify %.1f ms/pass (target %.1f, inject %.1f)\n",
                    K, h.c_str(), 1e3 * t_draft / std::max(1, n_pass), 1e3 * t_verify / std::max(1, n_pass),
                    1e3 * t_target / std::max(1, n_pass), 1e3 * t_inject / std::max(1, n_pass));
        }
    }

    llama_batch_free(bt);
    if (spec) common_speculative_free(spec);
    llama_backend_free();
    return 0;
}
