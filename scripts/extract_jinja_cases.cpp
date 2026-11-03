// Writes the template cases of llama.cpp's tests/test-jinja.cpp, and what its
// Jinja engine makes of each, to the file named on the command line, for
// tests/jinja.goose. Each field is "<byte length>\n<bytes>\n"; each case is
//   T  name  template  vars (JSON)  expected  status  result
//   E  name  template  ""           expected  status  result
//   L  name  template  ""           ""        status  result
// for test_template, test_template_error and a template the lexer must
// reject. status is ok (result: the output), error (the message) or skip
// (not implemented). Built by scripts/gen-jinja-cases.sh, which extracts the
// test functions into cases.inc.
#include <cstdio>
#include <functional>
#include <string>

#include "json.h"
#include "jinja/runtime.h"
#include "jinja/parser.h"
#include "jinja/lexer.h"
#include "jinja/utils.h"
#include "jinja/caps.h"

using json = common_json;

static FILE * out;

static void field(const std::string & s) {
    fprintf(out, "%zu\n", s.size());
    fwrite(s.data(), 1, s.size(), out);
    fputc('\n', out);
}

// what test-jinja's testing does with the cases: nests their names
struct testing {
    std::string prefix;
    void test(const std::string & name, const std::function<void(testing &)> & fn) {
        std::string saved = prefix;
        prefix += name + " / ";
        fn(*this);
        prefix = saved;
    }
    bool assert_true(const std::string &, bool cond) { return cond; }
    void log(const std::string &) {}
};

static bool g_python_mode = false;

static void run(const std::string & tmpl, const json * vars, std::string & status, std::string & result) {
    try {
        jinja::lexer lexer;
        auto lexer_res = lexer.tokenize(tmpl);
        jinja::program ast = jinja::parse_from_tokens(lexer_res);
        jinja::context ctx(tmpl);
        if (vars) {
            jinja::global_from_json(ctx, *vars, true);
        }
        jinja::runtime runtime(ctx);
        auto parts = runtime.gather_string_parts(runtime.execute(ast));
        result.clear();
        for (const auto & part : parts->as_string().parts) {
            result += part.val;
        }
        status = "ok";
    } catch (const jinja::not_implemented_exception & e) {
        status = "skip";
        result = e.what();
    } catch (const std::exception & e) {
        status = "error";
        result = e.what();
    }
}

static void test_template(testing & t, const std::string & name, const std::string & tmpl, const json & vars, const std::string & expect) {
    std::string status, result;
    run(tmpl, &vars, status, result);
    field("T"); field(t.prefix + name); field(tmpl); field(vars.dump()); field(expect); field(status); field(result);
}

static void test_template_error(testing & t, const std::string & name, const std::string & tmpl, const std::string & expect) {
    std::string status, result;
    run(tmpl, nullptr, status, result);
    field("E"); field(t.prefix + name); field(tmpl); field(""); field(expect); field(status); field(result);
}

// jinja::lexer().tokenize, in the cases that check what the lexer rejects
static jinja::lexer_result rec_lexer_tokenize(const std::string & tmpl) {
    try {
        auto res = jinja::lexer().tokenize(tmpl);
        field("L"); field(""); field(tmpl); field(""); field(""); field("ok"); field("");
        return res;
    } catch (const jinja::lexer_exception & e) {
        field("L"); field(""); field(tmpl); field(""); field(""); field("error"); field(e.what());
        throw;
    }
}

#include "cases.inc"

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s out.txt\n", argv[0]);
        return 1;
    }
    out = fopen(argv[1], "wb");
    testing t;
    test_whitespace_control(t);
    test_conditionals(t);
    test_loops(t);
    test_expressions(t);
    test_set_statement(t);
    test_filters(t);
    test_literals(t);
    test_hexadecimal_escapes(t);
    test_comments(t);
    test_macros(t);
    test_namespace(t);
    test_tests(t);
    test_string_methods(t);
    test_array_methods(t);
    test_object_methods(t);
    fclose(out);
    return 0;
}
