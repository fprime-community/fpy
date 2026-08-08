// ======================================================================
// \title  Json.cpp
// \brief  Minimal JSON reader/writer for the harness protocol
// ======================================================================

#include "Json.hpp"

#include <cstdio>
#include <cstdlib>

namespace FpyHarness {

namespace {
const Json NULL_VALUE;
}

// ----------------------------------------------------------------------
// Json accessors
// ----------------------------------------------------------------------

bool Json::asBool(bool fallback) const {
    return m_kind == Kind::BOOL ? m_bool : fallback;
}

int64_t Json::asInt(int64_t fallback) const {
    return m_kind == Kind::NUMBER ? static_cast<int64_t>(m_number) : fallback;
}

double Json::asDouble(double fallback) const {
    return m_kind == Kind::NUMBER ? m_number : fallback;
}

std::string Json::asString(const std::string& fallback) const {
    return m_kind == Kind::STRING ? m_string : fallback;
}

const Json& Json::operator[](const std::string& key) const {
    if (m_kind != Kind::OBJECT) {
        return NULL_VALUE;
    }
    auto it = m_object.find(key);
    return it == m_object.end() ? NULL_VALUE : it->second;
}

// ----------------------------------------------------------------------
// Parsing
// ----------------------------------------------------------------------

class Parser {
  public:
    Parser(const std::string& text) : m_text(text), m_pos(0) {}

    bool parseValue(Json& out) {
        skipSpace();
        if (m_pos >= m_text.size()) {
            return false;
        }
        switch (m_text[m_pos]) {
            case '{':
                return parseObject(out);
            case '[':
                return parseArray(out);
            case '"':
                out.m_kind = Json::Kind::STRING;
                return parseString(out.m_string);
            case 't':
                return parseLiteral("true", out, true);
            case 'f':
                return parseLiteral("false", out, false);
            case 'n':
                return parseNull(out);
            default:
                return parseNumber(out);
        }
    }

    bool atEnd() {
        skipSpace();
        return m_pos >= m_text.size();
    }

  private:
    void skipSpace() {
        while (m_pos < m_text.size() && (m_text[m_pos] == ' ' || m_text[m_pos] == '\t' ||
                                         m_text[m_pos] == '\n' || m_text[m_pos] == '\r')) {
            m_pos++;
        }
    }

    bool expect(char c) {
        skipSpace();
        if (m_pos >= m_text.size() || m_text[m_pos] != c) {
            return false;
        }
        m_pos++;
        return true;
    }

    bool parseLiteral(const char* text, Json& out, bool val) {
        const size_t length = std::string(text).size();
        if (m_text.compare(m_pos, length, text) != 0) {
            return false;
        }
        m_pos += length;
        out.m_kind = Json::Kind::BOOL;
        out.m_bool = val;
        return true;
    }

    bool parseNull(Json& out) {
        if (m_text.compare(m_pos, 4, "null") != 0) {
            return false;
        }
        m_pos += 4;
        out.m_kind = Json::Kind::NUL;
        return true;
    }

    bool parseNumber(Json& out) {
        const char* start = m_text.c_str() + m_pos;
        char* end = nullptr;
        const double val = std::strtod(start, &end);
        if (end == start) {
            return false;
        }
        m_pos += static_cast<size_t>(end - start);
        out.m_kind = Json::Kind::NUMBER;
        out.m_number = val;
        return true;
    }

    bool parseString(std::string& out) {
        if (!expect('"')) {
            return false;
        }
        out.clear();
        while (m_pos < m_text.size()) {
            const char c = m_text[m_pos++];
            if (c == '"') {
                return true;
            }
            if (c != '\\') {
                out.push_back(c);
                continue;
            }
            if (m_pos >= m_text.size()) {
                return false;
            }
            const char esc = m_text[m_pos++];
            switch (esc) {
                case 'n': out.push_back('\n'); break;
                case 't': out.push_back('\t'); break;
                case 'r': out.push_back('\r'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'u': {
                    // The protocol only carries ASCII, so a \u escape is
                    // rendered from its low byte.
                    if (m_pos + 4 > m_text.size()) {
                        return false;
                    }
                    const std::string digits = m_text.substr(m_pos, 4);
                    m_pos += 4;
                    out.push_back(static_cast<char>(std::strtol(digits.c_str(), nullptr, 16)));
                    break;
                }
                default: out.push_back(esc); break;
            }
        }
        return false;
    }

    bool parseArray(Json& out) {
        if (!expect('[')) {
            return false;
        }
        out.m_kind = Json::Kind::ARRAY;
        skipSpace();
        if (expect(']')) {
            return true;
        }
        while (true) {
            Json element;
            if (!parseValue(element)) {
                return false;
            }
            out.m_array.push_back(element);
            skipSpace();
            if (expect(']')) {
                return true;
            }
            if (!expect(',')) {
                return false;
            }
        }
    }

    bool parseObject(Json& out) {
        if (!expect('{')) {
            return false;
        }
        out.m_kind = Json::Kind::OBJECT;
        skipSpace();
        if (expect('}')) {
            return true;
        }
        while (true) {
            std::string name;
            skipSpace();
            if (!parseString(name)) {
                return false;
            }
            if (!expect(':')) {
                return false;
            }
            Json member;
            if (!parseValue(member)) {
                return false;
            }
            out.m_object[name] = member;
            skipSpace();
            if (expect('}')) {
                return true;
            }
            if (!expect(',')) {
                return false;
            }
        }
    }

    const std::string& m_text;
    size_t m_pos;
};

bool Json::parse(const std::string& text, Json& out) {
    Parser parser(text);
    Json parsed;
    if (!parser.parseValue(parsed) || !parser.atEnd()) {
        return false;
    }
    out = parsed;
    return true;
}

// ----------------------------------------------------------------------
// Writing
// ----------------------------------------------------------------------

JsonWriter::JsonWriter() : m_out("{"), m_needComma(false) {}

void JsonWriter::separate() {
    if (m_needComma) {
        m_out += ",";
    }
    m_needComma = true;
}

JsonWriter& JsonWriter::key(const std::string& name) {
    separate();
    m_out += "\"" + name + "\":";
    m_needComma = false;
    return *this;
}

JsonWriter& JsonWriter::value(bool val) {
    separate();
    m_out += val ? "true" : "false";
    return *this;
}

JsonWriter& JsonWriter::value(int64_t val) {
    separate();
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(val));
    m_out += buf;
    return *this;
}

JsonWriter& JsonWriter::value(const std::string& val) {
    separate();
    m_out += "\"";
    for (const char c : val) {
        switch (c) {
            case '"': m_out += "\\\""; break;
            case '\\': m_out += "\\\\"; break;
            case '\n': m_out += "\\n"; break;
            case '\r': m_out += "\\r"; break;
            case '\t': m_out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                    m_out += buf;
                } else {
                    m_out += c;
                }
                break;
        }
    }
    m_out += "\"";
    return *this;
}

JsonWriter& JsonWriter::nullValue() {
    separate();
    m_out += "null";
    return *this;
}

JsonWriter& JsonWriter::beginArray() {
    separate();
    m_out += "[";
    m_needComma = false;
    return *this;
}

JsonWriter& JsonWriter::endArray() {
    m_out += "]";
    m_needComma = true;
    return *this;
}

JsonWriter& JsonWriter::beginObject() {
    separate();
    m_out += "{";
    m_needComma = false;
    return *this;
}

JsonWriter& JsonWriter::endObject() {
    m_out += "}";
    m_needComma = true;
    return *this;
}

// ----------------------------------------------------------------------
// Hex
// ----------------------------------------------------------------------

std::string toHex(const uint8_t* bytes, size_t length) {
    static const char* DIGITS = "0123456789abcdef";
    std::string out;
    out.reserve(length * 2);
    for (size_t i = 0; i < length; i++) {
        out.push_back(DIGITS[bytes[i] >> 4]);
        out.push_back(DIGITS[bytes[i] & 0x0F]);
    }
    return out;
}

bool fromHex(const std::string& hex, std::vector<uint8_t>& out) {
    if (hex.size() % 2 != 0) {
        return false;
    }
    out.clear();
    out.reserve(hex.size() / 2);
    for (size_t i = 0; i < hex.size(); i += 2) {
        uint8_t byte = 0;
        for (size_t half = 0; half < 2; half++) {
            const char c = hex[i + half];
            uint8_t nibble = 0;
            if (c >= '0' && c <= '9') {
                nibble = static_cast<uint8_t>(c - '0');
            } else if (c >= 'a' && c <= 'f') {
                nibble = static_cast<uint8_t>(c - 'a' + 10);
            } else if (c >= 'A' && c <= 'F') {
                nibble = static_cast<uint8_t>(c - 'A' + 10);
            } else {
                return false;
            }
            byte = static_cast<uint8_t>((byte << 4) | nibble);
        }
        out.push_back(byte);
    }
    return true;
}

}  // namespace FpyHarness
