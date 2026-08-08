// ======================================================================
// \title  Json.hpp
// \brief  Minimal JSON reader/writer for the harness protocol
// ======================================================================

#ifndef FPY_HARNESS_JSON_HPP
#define FPY_HARNESS_JSON_HPP

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace FpyHarness {

//! A JSON value.
class Json {
  public:
    enum class Kind { NUL, BOOL, NUMBER, STRING, ARRAY, OBJECT };

    Json() : m_kind(Kind::NUL), m_bool(false), m_number(0.0) {}

    Kind kind() const { return m_kind; }
    bool isNull() const { return m_kind == Kind::NUL; }

    //! Parse *text*. Returns false and leaves *out* null on malformed input.
    static bool parse(const std::string& text, Json& out);

    // Each accessor returns *fallback* when the value is absent or of another
    // kind, so an optional field reads in one line.
    bool asBool(bool fallback = false) const;
    int64_t asInt(int64_t fallback = 0) const;
    double asDouble(double fallback = 0.0) const;
    std::string asString(const std::string& fallback = "") const;

    //! An object's member, or a null value when absent.
    const Json& operator[](const std::string& key) const;
    //! An array's elements, empty for any other kind.
    const std::vector<Json>& items() const { return m_array; }
    //! An object's members, empty for any other kind.
    const std::map<std::string, Json>& members() const { return m_object; }

  private:
    Kind m_kind;
    bool m_bool;
    double m_number;
    std::string m_string;
    std::vector<Json> m_array;
    std::map<std::string, Json> m_object;

    friend class Parser;
};

//! Builds one JSON object, written in the order fields are added.
class JsonWriter {
  public:
    JsonWriter();

    JsonWriter& key(const std::string& name);
    JsonWriter& value(bool val);
    JsonWriter& value(int64_t val);
    JsonWriter& value(const std::string& val);
    JsonWriter& nullValue();
    JsonWriter& beginArray();
    JsonWriter& endArray();
    JsonWriter& beginObject();
    JsonWriter& endObject();

    //! The finished object. Valid once every begin has been matched.
    std::string str() const { return m_out + "}"; }

  private:
    void separate();

    std::string m_out;
    bool m_needComma;
};

//! Lowercase hex of *bytes*.
std::string toHex(const uint8_t* bytes, size_t length);
//! Parses hex of either case. Returns false on odd length or a non-hex digit.
bool fromHex(const std::string& hex, std::vector<uint8_t>& out);

}  // namespace FpyHarness

#endif
