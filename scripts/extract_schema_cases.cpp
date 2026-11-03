// Writes the cases of llama.cpp's tests/test-json-schema-to-grammar.cpp, and
// the grammar its json_schema_to_grammar makes of each, to the file named on
// the command line, for tests/json_schema.goose. Each field is
// "<byte length>\n<bytes>\n"; each case is
//   S  name  schema  ok  grammar (or the error)
// Built by scripts/gen-schema-cases.sh, which extracts the test cases into
// cases.inc.
#include <cassert>
#include <cstdio>
#include <functional>
#include <regex>
#include <stdexcept>
#include <string>

#include "json-schema-to-grammar.h"
#include "llama-grammar.h"
#include "json.h"

static FILE * out;

static void field(const std::string & s) {
    fprintf(out, "%zu\n", s.size());
    fwrite(s.data(), 1, s.size(), out);
    fputc('\n', out);
}

#include "cases.inc"

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s out.txt\n", argv[0]);
        return 1;
    }
    out = fopen(argv[1], "wb");
    test_all("JSON schema conversion", [](const TestCase & tc) {
        try {
            std::string g = json_schema_to_grammar(common_json::parse(tc.schema), true);
            field("S"); field(tc.name); field(tc.schema); field("1"); field(g);
        } catch (const std::exception & ex) {
            field("S"); field(tc.name); field(tc.schema); field("0"); field(ex.what());
        }
    });
    fclose(out);
    return 0;
}
