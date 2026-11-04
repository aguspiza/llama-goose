// record_case: what llama.cpp makes of a test_template_output_peg_parsers
// case (see extract_peg_chat_cases.cpp). Included into test-chat.cpp before
// peg_test_builder.

static size_t record_utf8_safe(const std::string & s, size_t len) {
    // back to the start of a sequence that is cut short
    size_t i = len;
    for (size_t back = 0; back < 4 && i > 0; ++back) {
        --i;
        unsigned char c = s[i];
        if ((c & 0x80) == 0) {
            return len;
        }
        if ((c & 0xC0) == 0xC0) {
            size_t expected = (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 0;
            return expected && len - i >= expected ? len : i;
        }
    }
    return len;
}

static std::string record_strings(const std::vector<std::string> & v) {
    json a = json::array();
    for (const auto & s : v) {
        a.push_back(json(s));
    }
    return a.dump();
}

static void record_msg(const common_chat_msg & msg) {
    field(msg.role);
    field(msg.content);
    field(msg.reasoning_content);
    field(std::to_string(msg.tool_calls.size()));
    for (const auto & tc : msg.tool_calls) {
        field(tc.name);
        field(tc.arguments);
        field(tc.id);
    }
}

static void record_case(const std::string & path, common_chat_templates * tmpls, peg_test_case tc) {
    if (tc.params.messages.empty()) {
        tc.params.messages = { message_user };
    }
    const auto & p = tc.params;
    json inputs = json::object();
    inputs["messages"]               = common_chat_msgs_to_json_oaicompat(p.messages);
    inputs["tools"]                  = p.tools.empty() ? json() : common_chat_tools_to_json_oaicompat(p.tools);
    inputs["tool_choice"]            = (int) p.tool_choice;
    inputs["parallel_tool_calls"]    = p.parallel_tool_calls;
    inputs["reasoning_format"]       = (int) p.reasoning_format;
    inputs["enable_thinking"]        = p.enable_thinking;
    inputs["add_generation_prompt"]  = p.add_generation_prompt;
    inputs["continue_final_message"] = (int) p.continue_final_message;
    inputs["json_schema"]            = p.json_schema;
    inputs["chat_template_kwargs"]   = json::object();
    for (const auto & [k, v] : p.chat_template_kwargs) {
        inputs["chat_template_kwargs"][k] = v;
    }
    inputs["is_partial"] = tc.is_partial;

    // the template's source, before its first case
    static std::set<std::string> seen;
    if (seen.insert(path).second) {
        field("S");
        field(path);
        field(path == "deepseek-v4.1-min" ? std::string(DEEPSEEK_V4_1_MIN_TEMPLATE) : read_file(path));
    }

    field("P");
    field(path);
    field(tc.input);
    field(inputs.dump());

    common_chat_params applied;
    try {
        applied = common_chat_templates_apply(tmpls, tc.params);
    } catch (const std::exception & e) {
        field("0");
        field(e.what());
        return;
    }
    field("1");
    field(applied.prompt.str());
    json parts = json::array();
    for (const auto & part : applied.prompt.parts) {
        json pair = json::array();
        pair.push_back(json(part.text));
        pair.push_back(json(part.is_input));
        parts.push_back(pair);
    }
    field(parts.dump());
    field(std::to_string((int) applied.format));
    field(applied.grammar);
    field(applied.grammar_lazy ? "1" : "0");
    field(applied.generation_prompt);
    json triggers = json::array();
    for (const auto & t : applied.grammar_triggers) {
        json pair = json::array();
        pair.push_back(json((int) t.type));
        pair.push_back(json(t.value));
        triggers.push_back(pair);
    }
    field(triggers.dump());
    field(record_strings(applied.preserved_tokens));
    field(record_strings(applied.additional_stops));
    field(applied.thinking_start_tag);
    field(record_strings(applied.thinking_end_tags));

    common_chat_parser_params pp(applied);
    pp.parser = applied.parser;
    const auto & in = tc.input;
    std::vector<std::pair<size_t, bool>> cuts;
    if (in.size() >= 3) {
        cuts.push_back({ record_utf8_safe(in, in.size() / 3), true });
        cuts.push_back({ record_utf8_safe(in, 2 * in.size() / 3), true });
    }
    cuts.push_back({ in.size(), tc.is_partial });
    field(std::to_string(cuts.size()));
    for (const auto & [cut, partial] : cuts) {
        field(std::to_string(cut));
        field(partial ? "1" : "0");
        try {
            auto msg = common_chat_parse(common_chat_input(in.substr(0, cut)), partial, pp);
            field("1");
            record_msg(msg);
        } catch (const std::exception & e) {
            field("0");
            field(e.what());
        }
    }
}
