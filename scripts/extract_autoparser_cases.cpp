// Writes what llama.cpp's differential autoparser finds in each template
// named on the command line (after the output file), for
// tests/autoparser.goose. Each field is "<byte length>\n<bytes>\n"; each
// template is
//   A  name  ok  (ok: the analysis fields, in the order of FIELDS below;
//                 else the error)
// Built by scripts/gen-autoparser-cases.sh.
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#include "chat.h"
#include "chat-auto-parser.h"

static FILE * out;

static void field(const std::string & s) {
    fprintf(out, "%zu\n", s.size());
    fwrite(s.data(), 1, s.size(), out);
    fputc('\n', out);
}

static std::string join(const std::vector<std::string> & v) {
    std::string s;
    for (const auto & x : v) {
        s += x;
        s += '\0';
    }
    return s;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s out.txt templates...\n", argv[0]);
        return 1;
    }
    out = fopen(argv[1], "wb");
    for (int i = 2; i < argc; i++) {
        std::ifstream f(argv[i], std::ios::binary);
        std::stringstream ss;
        ss << f.rdbuf();
        std::string name = argv[i];
        name = name.substr(name.find_last_of("/\\") + 1);
        field("A");
        field(name);
        try {
            common_chat_template tmpl(ss.str(), "<s>", "</s>");
            autoparser::autoparser a;
            a.analyze_template(tmpl);
            field("1");
            field(std::to_string((int) a.reasoning.mode));
            field(a.reasoning.start);
            field(a.reasoning.end);
            field(std::to_string((int) a.content.mode));
            field(a.content.start);
            field(a.content.end);
            const auto & t = a.tools;
            field(std::to_string((int) t.format.mode));
            field(t.format.section_start);
            field(t.format.section_end);
            field(t.format.per_call_start);
            field(t.format.per_call_end);
            field(t.format.fun_name_is_key ? "1" : "0");
            field(t.format.tools_array_wrapped ? "1" : "0");
            field(t.format.openai_wrapper_trigger ? "1" : "0");
            field(t.format.function_field);
            field(t.format.name_field);
            field(t.format.args_field);
            field(t.format.id_field);
            field(t.format.gen_id_field);
            field(join(t.format.parameter_order));
            field(t.function.name_prefix);
            field(t.function.name_suffix);
            field(t.function.args_separator);
            field(t.function.close);
            field(t.arguments.start);
            field(t.arguments.end);
            field(t.arguments.name_prefix);
            field(t.arguments.name_suffix);
            field(t.arguments.value_prefix);
            field(t.arguments.value_suffix);
            field(t.arguments.separator);
            field(t.arguments.tolerate_intertag_whitespace ? "1" : "0");
            field(std::to_string((int) t.call_id.pos));
            field(t.call_id.prefix);
            field(t.call_id.suffix);
            field(a.user_start);
            field(a.assistant_start);
            field(join(a.preserved_tokens));
            field(join(a.additional_stops));
        } catch (const std::exception & e) {
            field("0");
            field(e.what());
        }
    }
    fclose(out);
    return 0;
}
