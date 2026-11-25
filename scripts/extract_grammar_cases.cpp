// Writes the cases of llama.cpp's tests/test-grammar-parser.cpp and
// tests/test-grammar-integration.cpp, and what its grammar code makes of
// each, to the file named on the command line, for tests/grammar.goose.
// Each field is "<byte length>\n<bytes>\n"; each case is
//   P  grammar  ok  symbols  rules       the parser (test-grammar-parser)
//   B  grammar  root  ok                 llama_grammar_init_impl
//   G  grammar                           the grammar of the M cases after it
//   M  input  matched  masks             a string through that grammar; masks has a
//                                        "1" or "0" per token tried: whether the
//                                        sampler mask allowed it
// symbols is "name id\n" per symbol, by name; rules is one line per rule,
// "type:value " per element. Built by scripts/gen-grammar-cases.sh, which
// extracts the test functions into parser.inc and integration.inc.
#include <cassert>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "json-schema-to-grammar.h"
#include "unicode.h"            // llama.cpp's src/
#include "llama-grammar.h"
#include "json.h"

using json = common_json;

static FILE * out;

static void field(const std::string & s) {
    fprintf(out, "%zu\n", s.size());
    fwrite(s.data(), 1, s.size(), out);
    fputc('\n', out);
}

// --- test-grammar-parser ---

static void record_parse(const char * grammar_bytes) {
    llama_grammar_parser parsed;
    bool ok = parsed.parse(grammar_bytes);
    std::string symbols, rules;
    for (const auto & kv : parsed.symbol_ids) {
        symbols += kv.first + " " + std::to_string(kv.second) + "\n";
    }
    for (const auto & rule : parsed.rules) {
        for (const auto & e : rule) {
            rules += std::to_string((int) e.type) + ":" + std::to_string(e.value) + " ";
        }
        rules += "\n";
    }
    field("P"); field(grammar_bytes); field(ok ? "1" : "0"); field(symbols); field(rules);
}

static void verify_parsing(const char * grammar_bytes, const std::vector<std::pair<std::string, uint32_t>>, const std::vector<llama_grammar_element> &) {
    record_parse(grammar_bytes);
}

static void verify_failure(const char * grammar_bytes) {
    record_parse(grammar_bytes);
}

static void parser_cases() {
#include "parser.inc"
}

// --- test-grammar-integration ---

static llama_grammar * build_grammar_with_root(const std::string & grammar_str, const char * grammar_root) {
    llama_grammar * g = llama_grammar_init_impl(nullptr, grammar_str.c_str(), grammar_root, false, nullptr, 0, nullptr, 0);
    field("B"); field(grammar_str); field(grammar_root); field(g ? "1" : "0");
    return g;
}

static llama_grammar * build_grammar(const std::string & grammar_str) {
    return build_grammar_with_root(grammar_str, "root");
}

static bool test_build_grammar_fails(const std::string & grammar_str) {
    llama_grammar * g = build_grammar(grammar_str);
    bool fails = g == nullptr;
    if (g) llama_grammar_free_impl(g);
    return fails;
}

struct token_and_piece {
    llama_token token;
    std::string piece;
};

// a 0xff marker, the ID in big-endian order, the piece's length and the piece
static std::string token(llama_token id, const std::string & piece) {
    return std::string{
        static_cast<char>(0xff),
        static_cast<char>((id >> 24) & 0xff),
        static_cast<char>((id >> 16) & 0xff),
        static_cast<char>((id >> 8) & 0xff),
        static_cast<char>(id & 0xff),
        static_cast<char>(piece.size())
    } + piece;
}

static std::string token(llama_token id) {
    return token(id, "<[" + std::to_string(id) + "]>");
}

static std::vector<token_and_piece> parse_tokens(const std::string & input) {
    std::vector<token_and_piece> result;
    size_t offset = 0;
    while (offset < input.size()) {
        try {
            if (static_cast<unsigned char>(input[offset]) == 0xff) {
                if (offset + 6 > input.size()) {
                    throw std::runtime_error("not enough bytes for token id");
                }
                uint32_t val =
                    (static_cast<unsigned char>(input[offset + 1]) << 24) |
                    (static_cast<unsigned char>(input[offset + 2]) << 16) |
                    (static_cast<unsigned char>(input[offset + 3]) << 8)  |
                    (static_cast<unsigned char>(input[offset + 4]));
                size_t len = static_cast<unsigned char>(input[offset + 5]);
                if (offset + 6 + len > input.size()) {
                    throw std::runtime_error("not enough bytes for token piece");
                }
                result.push_back({static_cast<llama_token>(val), input.substr(offset + 6, len)});
                offset += 6 + len;
            } else {
                uint32_t cpt = unicode_cpt_from_utf8(input, offset);
                result.push_back({0, unicode_cpt_to_utf8(cpt)});
            }
        } catch (const std::invalid_argument &) {
            ++offset;
            result.push_back({0, unicode_cpt_to_utf8(0xFFFD)});
        }
    }
    return result;
}

// whether the sampler mask allows the token: it rejects one only if every stack does
static bool mask_allows(llama_grammar * grammar, const token_and_piece & in) {
    const auto decoded = llama_grammar_decode_utf8(in.piece, grammar->partial_utf8);
    llama_grammar_candidates rejects = { { 0, decoded.first.data(), decoded.second, in.token, llama_grammar_is_opaque(*grammar, in.token) } };
    for (const auto & stack : llama_grammar_get_stacks(grammar)) {
        rejects = llama_grammar_reject_candidates_for_stack(llama_grammar_get_rules(grammar), stack, rejects);
    }
    return rejects.empty();
}

static bool match_string(const std::string & input, llama_grammar * grammar, std::string & masks) {
    const auto parsed = parse_tokens(input);
    auto & stacks_cur = llama_grammar_get_stacks(grammar);
    for (const auto & in : parsed) {
        masks += mask_allows(grammar, in) ? "1" : "0";
        try {
            llama_grammar_accept_token(*grammar, in.token, in.piece);
        } catch (const std::runtime_error &) {
            return false;
        }
    }
    for (const auto & stack : stacks_cur) {
        if (stack.empty()) {
            return true;
        }
    }
    return false;
}

static void test(const std::string &, const std::string & grammar_str, const std::vector<std::string> & passing, const std::vector<std::string> & failing) {
    auto * grammar = build_grammar(grammar_str);
    if (!grammar) {
        return;
    }
    const llama_grammar_stacks stacks_org = llama_grammar_get_stacks(grammar);
    const llama_partial_utf8   partial_org = grammar->partial_utf8;
    llama_grammar_stacks & stacks_cur = llama_grammar_get_stacks(grammar);
    field("G"); field(grammar_str);
    for (const auto * list : { &passing, &failing }) {
        for (const auto & s : *list) {
            std::string masks;
            bool matched = match_string(s, grammar, masks);
            field("M"); field(s); field(matched ? "1" : "0"); field(masks);
            stacks_cur = stacks_org;
            grammar->partial_utf8 = partial_org;
        }
    }
    llama_grammar_free_impl(grammar);
}

static void test_grammar(const std::string & desc, const std::string & grammar_str, const std::vector<std::string> & passing, const std::vector<std::string> & failing) {
    test(desc, grammar_str, passing, failing);
}

static void test_schema(const std::string & desc, const std::string & schema_str, const std::vector<std::string> & passing, const std::vector<std::string> & failing) {
    test(desc, json_schema_to_grammar(json::parse(schema_str), true), passing, failing);
}

#include "integration.inc"

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s out.txt\n", argv[0]);
        return 1;
    }
    out = fopen(argv[1], "wb");
    parser_cases();
    test_simple_grammar();
    test_complex_grammar();
    test_special_chars();
    test_quantifiers();
    test_failure_missing_root();
    test_failure_missing_reference();
    test_failure_left_recursion();
    test_failure_missing_root_symbol();
    test_custom_root_symbol_check();
    test_json_schema();
    fclose(out);
    return 0;
}
