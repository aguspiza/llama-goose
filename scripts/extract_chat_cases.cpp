// Writes llama.cpp's chat template test cases, and what libllama's
// llama_chat_apply_template and libcommon's Jinja path
// (common_chat_templates_apply, as test-chat-template's format_using_common
// calls it) make of them, to a data file: a sequence of fields, each
// "<byte length>\n<bytes>\n".
#include "llama.h"
#include "chat.h"
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

#define U8C(x) (const char*)(u8##x)

static FILE * out;

static void field(const std::string & s) {
    fprintf(out, "%zu\n", s.size());
    fwrite(s.data(), 1, s.size(), out);
    fputc('\n', out);
}

static std::string apply(const char * tmpl, const std::vector<llama_chat_message> & conv, bool add_ass) {
    int32_t n = llama_chat_apply_template(tmpl, conv.data(), conv.size(), add_ass, nullptr, 0);
    if (n < 0) return "<ERROR>";
    std::vector<char> buf(n + 1);
    llama_chat_apply_template(tmpl, conv.data(), conv.size(), add_ass, buf.data(), buf.size());
    return std::string(buf.data(), n);
}

// format_using_common: the Jinja rendering, or "<ERROR>" and the message.
static std::string apply_jinja(const std::string & tmpl, const std::string & bos, const std::string & eos,
                               const std::vector<llama_chat_message> & conv) {
    try {
        auto tmpls = common_chat_templates_init(nullptr, tmpl, bos, eos);
        common_chat_templates_inputs inputs;
        inputs.use_jinja = true;
        for (const auto & m : conv) {
            common_chat_msg msg;
            msg.role = m.role;
            msg.content = m.content;
            inputs.messages.push_back(msg);
        }
        inputs.add_generation_prompt = true;
        return common_chat_templates_apply(tmpls.get(), inputs).prompt.str();
    } catch (const std::exception & e) {
        return std::string("<ERROR>") + e.what();
    }
}

int main(int argc, char ** argv) {
    out = fopen(argv[1], "wb");
#include "cases.inc"
    // the conversation
    field(std::to_string(conversation.size()));
    for (auto & m : conversation) { field(m.role); field(m.content); }
    // the built-in templates, by name
    std::vector<const char *> names(llama_chat_builtin_templates(nullptr, 0));
    llama_chat_builtin_templates(names.data(), names.size());
    field(std::to_string(names.size()));
    for (auto * name : names) {
        field(name);
        field(apply(name, conversation, true));
        field(apply(name, conversation, false));
    }
    // the test cases
    field(std::to_string(test_cases.size()));
    for (auto & tc : test_cases) {
        auto conv = conversation;
        conv.insert(conv.end(), tc.extra_conversation.begin(), tc.extra_conversation.end());
        field(tc.name);
        field(tc.template_str);
        field(tc.expected_output);
        field(tc.expected_output_jinja);
        field(tc.bos_token);
        field(tc.eos_token);
        field(tc.supported_with_jinja ? "1" : "0");
        field(std::to_string(tc.extra_conversation.size()));
        for (auto & m : tc.extra_conversation) { field(m.role); field(m.content); }
        field(apply(tc.template_str.c_str(), conv, true));
        field(apply(tc.template_str.c_str(), conv, false));
        field(tc.supported_with_jinja ? apply_jinja(tc.template_str, tc.bos_token, tc.eos_token, conv) : "");
    }
    fclose(out);
    return 0;
}
