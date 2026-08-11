// A minimal JSON reader and writer for the harness protocol: one JSON object
// per line on stdin and stdout. Supports null, bool, integer, string, array
// and object; that is all the protocol uses. Numbers with fractions or
// exponents are rejected.

#ifndef TEST_HARNESS_JSON_HPP
#define TEST_HARNESS_JSON_HPP

#include <string>
#include <utility>
#include <vector>

#include <Fw/FPrimeBasicTypes.hpp>

namespace harness {

struct JsonValue {
    enum class Type { NUL, BOOL, INT, STRING, ARRAY, OBJECT };

    Type type = Type::NUL;
    bool boolValue = false;
    I64 intValue = 0;
    std::string stringValue;
    std::vector<JsonValue> items;                              // ARRAY
    std::vector<std::pair<std::string, JsonValue>> members;    // OBJECT

    static JsonValue makeInt(I64 value);
    static JsonValue makeBool(bool value);
    static JsonValue makeString(const std::string& value);
    static JsonValue makeArray();
    static JsonValue makeObject();

    // Object helpers. get() returns nullptr when the key is absent.
    const JsonValue* get(const std::string& key) const;
    void set(const std::string& key, JsonValue value);
};

// Parses one JSON document. Throws std::runtime_error on malformed input.
JsonValue jsonParse(const std::string& text);

// Serializes a JSON document on a single line.
std::string jsonDump(const JsonValue& value);

}  // namespace harness

#endif
