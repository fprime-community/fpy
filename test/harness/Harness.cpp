#include "test/harness/Harness.hpp"

namespace harness {

std::vector<U8> hexDecode(const std::string& hex) {
    if (hex.size() % 2 != 0) {
        throw std::runtime_error("hex string with odd length");
    }
    std::vector<U8> out;
    out.reserve(hex.size() / 2);
    for (size_t i = 0; i < hex.size(); i += 2) {
        out.push_back(static_cast<U8>(std::stoul(hex.substr(i, 2), nullptr, 16)));
    }
    return out;
}

std::string hexEncode(const std::vector<U8>& data) {
    static const char digits[] = "0123456789abcdef";
    std::string out;
    out.reserve(data.size() * 2);
    for (U8 byte : data) {
        out.push_back(digits[byte >> 4]);
        out.push_back(digits[byte & 0xF]);
    }
    return out;
}

namespace {

const JsonValue& require(const JsonValue& object, const std::string& key) {
    const JsonValue* value = object.get(key);
    if (value == nullptr) {
        throw std::runtime_error("request is missing the '" + key + "' field");
    }
    return *value;
}

}  // namespace

HarnessRequest parseRequest(const JsonValue& json) {
    HarnessRequest request;
    request.seqFile = require(json, "seqFile").stringValue;
    if (const JsonValue* args = json.get("args")) {
        request.hasArgs = true;
        request.args = hexDecode(args->stringValue);
    }
    if (const JsonValue* tlm = json.get("tlm")) {
        for (const auto& member : tlm->members) {
            request.tlm[static_cast<U32>(std::stoull(member.first))] = hexDecode(member.second.stringValue);
        }
    }
    if (const JsonValue* prms = json.get("prms")) {
        for (const auto& member : prms->members) {
            request.prms[static_cast<U32>(std::stoull(member.first))] = hexDecode(member.second.stringValue);
        }
    }
    if (const JsonValue* time = json.get("time")) {
        request.timeBase = static_cast<U16>(require(*time, "base").intValue);
        request.timeContext = static_cast<U8>(require(*time, "context").intValue);
        request.seconds = static_cast<U32>(require(*time, "seconds").intValue);
        request.useconds = static_cast<U32>(require(*time, "useconds").intValue);
    }
    if (const JsonValue* opcodes = json.get("failOpcodes")) {
        for (const JsonValue& opcode : opcodes->items) {
            request.failOpcodes.insert(static_cast<U32>(opcode.intValue));
        }
    }
    if (const JsonValue* opcodes = json.get("seqRunOpcodes")) {
        for (const JsonValue& opcode : opcodes->items) {
            request.seqRunOpcodes.insert(static_cast<U32>(opcode.intValue));
        }
    }
    if (const JsonValue* size = json.get("seqArgsBufferSize")) {
        request.seqArgsBufferSize = static_cast<U32>(size->intValue);
    }
    if (const JsonValue* response = json.get("cmdResponse")) {
        request.cmdResponse = static_cast<U8>(response->intValue);
    }
    return request;
}

JsonValue resultToJson(const HarnessResult& result) {
    JsonValue json = JsonValue::makeObject();
    if (!result.error.empty()) {
        json.set("error", JsonValue::makeString(result.error));
    }
    if (result.gotCmdResponse) {
        json.set("cmdResponse", JsonValue::makeInt(result.cmdResponse));
    }
    json.set("state", JsonValue::makeInt(result.state));
    json.set("reachedRunning", JsonValue::makeBool(result.reachedRunning));
    json.set("statementsDispatched", JsonValue::makeInt(static_cast<I64>(result.statementsDispatched)));
    json.set("lastDirectiveError", JsonValue::makeInt(result.lastDirectiveError));
    if (result.exited) {
        json.set("exitCode", JsonValue::makeInt(result.exitCode));
    }

    JsonValue events = JsonValue::makeArray();
    for (const HarnessEvent& event : result.events) {
        JsonValue entry = JsonValue::makeObject();
        entry.set("id", JsonValue::makeInt(static_cast<I64>(event.id)));
        entry.set("severity", JsonValue::makeInt(event.severity));
        entry.set("text", JsonValue::makeString(event.text));
        if (event.guest) {
            entry.set("guest", JsonValue::makeBool(true));
        }
        events.items.push_back(entry);
    }
    json.set("events", events);

    JsonValue cmds = JsonValue::makeArray();
    for (const std::vector<U8>& cmd : result.cmds) {
        cmds.items.push_back(JsonValue::makeString(hexEncode(cmd)));
    }
    json.set("cmds", cmds);

    JsonValue serial = JsonValue::makeArray();
    for (const HarnessSerialWrite& write : result.serial) {
        JsonValue entry = JsonValue::makeObject();
        entry.set("port", JsonValue::makeInt(write.port));
        entry.set("data", JsonValue::makeString(hexEncode(write.data)));
        serial.items.push_back(entry);
    }
    json.set("serial", serial);

    json.set("stack", JsonValue::makeString(hexEncode(result.stack)));
    json.set("frameStart", JsonValue::makeInt(result.frameStart));
    json.set("sequencesSucceeded", JsonValue::makeInt(static_cast<I64>(result.sequencesSucceeded)));
    return json;
}

}  // namespace harness
