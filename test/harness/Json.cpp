#include "test/harness/Json.hpp"

#include <cstdio>
#include <stdexcept>

namespace harness {

JsonValue JsonValue::makeInt(I64 value) {
    JsonValue v;
    v.type = Type::INT;
    v.intValue = value;
    return v;
}

JsonValue JsonValue::makeBool(bool value) {
    JsonValue v;
    v.type = Type::BOOL;
    v.boolValue = value;
    return v;
}

JsonValue JsonValue::makeString(const std::string& value) {
    JsonValue v;
    v.type = Type::STRING;
    v.stringValue = value;
    return v;
}

JsonValue JsonValue::makeArray() {
    JsonValue v;
    v.type = Type::ARRAY;
    return v;
}

JsonValue JsonValue::makeObject() {
    JsonValue v;
    v.type = Type::OBJECT;
    return v;
}

const JsonValue* JsonValue::get(const std::string& key) const {
    for (const auto& member : this->members) {
        if (member.first == key) {
            return &member.second;
        }
    }
    return nullptr;
}

void JsonValue::set(const std::string& key, JsonValue value) {
    this->members.emplace_back(key, std::move(value));
}

namespace {

class Parser {
  public:
    explicit Parser(const std::string& text) : m_text(text) {}

    JsonValue parse() {
        JsonValue value = this->parseValue();
        this->skipSpace();
        if (m_pos != m_text.size()) {
            this->fail("trailing characters");
        }
        return value;
    }

  private:
    [[noreturn]] void fail(const char* message) {
        throw std::runtime_error(std::string("JSON parse error at offset ") + std::to_string(m_pos) + ": " + message);
    }

    void skipSpace() {
        while (m_pos < m_text.size()) {
            char c = m_text[m_pos];
            if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
                break;
            }
            m_pos++;
        }
    }

    char peek() {
        if (m_pos >= m_text.size()) {
            this->fail("unexpected end of input");
        }
        return m_text[m_pos];
    }

    char next() {
        char c = this->peek();
        m_pos++;
        return c;
    }

    void expect(char wanted) {
        if (this->next() != wanted) {
            this->fail("unexpected character");
        }
    }

    void expectWord(const char* word) {
        for (const char* c = word; *c != '\0'; c++) {
            this->expect(*c);
        }
    }

    JsonValue parseValue() {
        this->skipSpace();
        char c = this->peek();
        switch (c) {
            case '{':
                return this->parseObject();
            case '[':
                return this->parseArray();
            case '"':
                return JsonValue::makeString(this->parseString());
            case 't':
                this->expectWord("true");
                return JsonValue::makeBool(true);
            case 'f':
                this->expectWord("false");
                return JsonValue::makeBool(false);
            case 'n':
                this->expectWord("null");
                return JsonValue();
            default:
                return this->parseNumber();
        }
    }

    JsonValue parseObject() {
        JsonValue value = JsonValue::makeObject();
        this->expect('{');
        this->skipSpace();
        if (this->peek() == '}') {
            m_pos++;
            return value;
        }
        while (true) {
            this->skipSpace();
            std::string key = this->parseString();
            this->skipSpace();
            this->expect(':');
            value.set(key, this->parseValue());
            this->skipSpace();
            char c = this->next();
            if (c == '}') {
                return value;
            }
            if (c != ',') {
                this->fail("expected ',' or '}'");
            }
        }
    }

    JsonValue parseArray() {
        JsonValue value = JsonValue::makeArray();
        this->expect('[');
        this->skipSpace();
        if (this->peek() == ']') {
            m_pos++;
            return value;
        }
        while (true) {
            value.items.push_back(this->parseValue());
            this->skipSpace();
            char c = this->next();
            if (c == ']') {
                return value;
            }
            if (c != ',') {
                this->fail("expected ',' or ']'");
            }
        }
    }

    std::string parseString() {
        this->expect('"');
        std::string out;
        while (true) {
            char c = this->next();
            if (c == '"') {
                return out;
            }
            if (c != '\\') {
                out.push_back(c);
                continue;
            }
            char escape = this->next();
            switch (escape) {
                case '"':
                case '\\':
                case '/':
                    out.push_back(escape);
                    break;
                case 'b':
                    out.push_back('\b');
                    break;
                case 'f':
                    out.push_back('\f');
                    break;
                case 'n':
                    out.push_back('\n');
                    break;
                case 'r':
                    out.push_back('\r');
                    break;
                case 't':
                    out.push_back('\t');
                    break;
                case 'u': {
                    U32 code = 0;
                    for (int i = 0; i < 4; i++) {
                        char h = this->next();
                        code <<= 4;
                        if (h >= '0' && h <= '9') {
                            code += static_cast<U32>(h - '0');
                        } else if (h >= 'a' && h <= 'f') {
                            code += static_cast<U32>(h - 'a' + 10);
                        } else if (h >= 'A' && h <= 'F') {
                            code += static_cast<U32>(h - 'A' + 10);
                        } else {
                            this->fail("bad \\u escape");
                        }
                    }
                    // Encode the code point as UTF-8. Surrogate pairs are not
                    // supported; the protocol only ever escapes control chars.
                    if (code < 0x80) {
                        out.push_back(static_cast<char>(code));
                    } else if (code < 0x800) {
                        out.push_back(static_cast<char>(0xC0 | (code >> 6)));
                        out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
                    } else {
                        out.push_back(static_cast<char>(0xE0 | (code >> 12)));
                        out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
                        out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
                    }
                    break;
                }
                default:
                    this->fail("bad escape");
            }
        }
    }

    JsonValue parseNumber() {
        size_t start = m_pos;
        if (this->peek() == '-') {
            m_pos++;
        }
        while (m_pos < m_text.size() && m_text[m_pos] >= '0' && m_text[m_pos] <= '9') {
            m_pos++;
        }
        if (m_pos == start || (m_text[start] == '-' && m_pos == start + 1)) {
            this->fail("bad number");
        }
        if (m_pos < m_text.size() && (m_text[m_pos] == '.' || m_text[m_pos] == 'e' || m_text[m_pos] == 'E')) {
            this->fail("only integers are supported");
        }
        return JsonValue::makeInt(std::stoll(m_text.substr(start, m_pos - start)));
    }

    const std::string& m_text;
    size_t m_pos = 0;
};

void dumpString(const std::string& value, std::string& out) {
    out.push_back('"');
    for (char c : value) {
        switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    (void)snprintf(buf, sizeof buf, "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back(c);
                }
        }
    }
    out.push_back('"');
}

void dumpValue(const JsonValue& value, std::string& out) {
    switch (value.type) {
        case JsonValue::Type::NUL:
            out += "null";
            break;
        case JsonValue::Type::BOOL:
            out += value.boolValue ? "true" : "false";
            break;
        case JsonValue::Type::INT:
            out += std::to_string(value.intValue);
            break;
        case JsonValue::Type::STRING:
            dumpString(value.stringValue, out);
            break;
        case JsonValue::Type::ARRAY: {
            out.push_back('[');
            bool first = true;
            for (const JsonValue& item : value.items) {
                if (!first) {
                    out.push_back(',');
                }
                first = false;
                dumpValue(item, out);
            }
            out.push_back(']');
            break;
        }
        case JsonValue::Type::OBJECT: {
            out.push_back('{');
            bool first = true;
            for (const auto& member : value.members) {
                if (!first) {
                    out.push_back(',');
                }
                first = false;
                dumpString(member.first, out);
                out.push_back(':');
                dumpValue(member.second, out);
            }
            out.push_back('}');
            break;
        }
    }
}

}  // namespace

JsonValue jsonParse(const std::string& text) {
    return Parser(text).parse();
}

std::string jsonDump(const JsonValue& value) {
    std::string out;
    dumpValue(value, out);
    return out;
}

}  // namespace harness
