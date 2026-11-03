// llama.cpp's libllama in Goose. The package's modules are imported one by
// one: import llama.unicode; import llama.vocab; ...
namespace llama;

// No log messages (llama_log_set with a callback that drops them), only errors.
var quiet = false;
