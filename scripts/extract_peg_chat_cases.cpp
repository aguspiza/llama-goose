// Writes the cases of test-chat.cpp's test_template_output_peg_parsers, and
// what llama.cpp makes of each, to the file named on the command line, for
// tests/chat_peg.goose. Built by scripts/gen-peg-chat-cases.sh, which
// includes test-chat.cpp with peg_test_builder::run() sending each case to
// record_case instead of checking it.
// Each field is "<byte length>\n<bytes>\n". The file starts with
//   T  the time the prompts were made with (seconds since 1970)
// then, before the first case of each template,
//   S  template  its source
// and each case is
//   P  template  input  inputs (JSON)  ok
//   ok:  prompt  prompt parts (JSON)  format  grammar  grammar_lazy
//        generation_prompt  triggers (JSON [[type, value], ...])
//        preserved tokens (JSON)  additional stops (JSON)
//        thinking start  thinking ends (JSON)  number of parses
//        and per parse: cut  is_partial  ok  (ok: msg fields, else the error)
//   else: the error
// msg fields: role  content  reasoning  number of tool calls, and per call
// name  arguments  id.
#include <cstdio>
#include <ctime>
#include <string>

static FILE * out;

static void field(const std::string & s) {
    fprintf(out, "%zu\n", s.size());
    fwrite(s.data(), 1, s.size(), out);
    fputc('\n', out);
}

#define main test_chat_main
#include "test-chat-recorded.cpp"
#undef main

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s out.txt\n", argv[0]);
        return 1;
    }
    out = fopen(argv[1], "wb");
    field("T");
    field(std::to_string((long long) time(nullptr)));
    test_template_output_peg_parsers(false);
    fclose(out);
    return 0;
}
