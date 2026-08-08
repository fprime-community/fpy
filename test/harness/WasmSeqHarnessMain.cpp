// ======================================================================
// \title  WasmSeqHarnessMain.cpp
// \brief  JSON-lines protocol over a real Svc::WasmSequencer
//
// Reads one request object per line on stdin and writes one response object
// per line on stdout. stderr carries framework logging and is not part of the
// protocol.
// ======================================================================

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

#include "Fw/Types/Assert.hpp"
#include "WasmSequencerHarness.hpp"
#include "Json.hpp"
#include "Svc/FpySequencer/FppConstantsAc.hpp"
#include "config/SerialPortIndexEnumAc.hpp"

using namespace FpyHarness;

namespace {

//! Raised in place of aborting, so one failed assert costs a single request
//! rather than the process.
struct AssertFailure : public std::exception {
    std::string text;
    explicit AssertFailure(const std::string& message) : text(message) {}
    const char* what() const noexcept override { return text.c_str(); }
};

class ThrowingAssertHook final : public Fw::AssertHook {
  public:
    void printAssert(const CHAR* msg) override { m_text = msg; }
    void doAssert() override { throw AssertFailure(m_text); }

  private:
    std::string m_text;
};

//! Reads a hex-keyed map of id -> serialized value.
template <typename KeyType>
std::map<KeyType, std::vector<uint8_t>> readValueMap(const Json& object) {
    std::map<KeyType, std::vector<uint8_t>> out;
    for (const auto& member : object.members()) {
        std::vector<uint8_t> bytes;
        if (!fromHex(member.second.asString(), bytes)) {
            continue;
        }
        out[static_cast<KeyType>(std::strtoull(member.first.c_str(), nullptr, 10))] = bytes;
    }
    return out;
}

template <typename ElementType>
std::set<ElementType> readIntSet(const Json& array) {
    std::set<ElementType> out;
    for (const Json& element : array.items()) {
        out.insert(static_cast<ElementType>(element.asInt()));
    }
    return out;
}

Svc::HarnessRequest readRequest(const Json& json) {
    Svc::HarnessRequest request;
    request.seqPath = json["seq_path"].asString();
    request.cwd = json["cwd"].asString();
    fromHex(json["args_hex"].asString(), request.args);

    const Json& time = json["time"];
    request.timeBase = static_cast<U16>(time["base"].asInt());
    request.timeContext = static_cast<U8>(time["context"].asInt());
    request.initialTimeUs = static_cast<U64>(time["initial_us"].asInt());

    request.tlm = readValueMap<FwChanIdType>(json["tlm"]);
    request.prm = readValueMap<FwPrmIdType>(json["prm"]);
    request.failOpcodes = readIntSet<FwOpcodeType>(json["fail_opcodes"]);
    request.seqRunOpcodes = readIntSet<FwOpcodeType>(json["seq_run_opcodes"]);
    request.disconnectedSerialPorts = readIntSet<FwIndexType>(json["disconnected_serial_ports"]);
    request.cmdResponse = static_cast<U32>(json["cmd_response"].asInt(0));
    if (!json["max_dispatch"].isNull()) {
        request.maxDispatch = static_cast<U32>(json["max_dispatch"].asInt());
    }
    return request;
}

std::string writeResult(int64_t id, const Svc::HarnessResult& result) {
    JsonWriter out;
    out.key("id").value(id);
    out.key("ok").value(result.ok);
    out.key("error_code").value(static_cast<int64_t>(result.errorCode));
    out.key("exit_code");
    if (result.hasExitCode) {
        out.value(static_cast<int64_t>(result.exitCode));
    } else {
        out.nullValue();
    }
    out.key("validation_failed").value(result.validationFailed);
    out.key("stack_size").value(static_cast<int64_t>(result.stackSize));
    out.key("statements_dispatched").value(static_cast<int64_t>(result.statementsDispatched));
    out.key("final_state").value(result.finalState);
    out.key("sim_time_us").value(static_cast<int64_t>(result.simTimeUs));
    out.key("error").value(result.error);

    out.key("cmds").beginArray();
    for (const auto& cmd : result.cmds) {
        out.value(toHex(cmd.data(), cmd.size()));
    }
    out.endArray();

    out.key("events").beginArray();
    for (const auto& event : result.events) {
        out.beginArray().value(static_cast<int64_t>(event.first)).value(event.second).endArray();
    }
    out.endArray();

    out.key("serial").beginArray();
    for (const auto& write : result.serialWrites) {
        out.beginArray()
            .value(static_cast<int64_t>(write.first))
            .value(toHex(write.second.data(), write.second.size()))
            .endArray();
    }
    out.endArray();
    return out.str();
}

std::string writePing(int64_t id) {
    JsonWriter out;
    out.key("id").value(id);
    out.key("ok").value(true);
    out.key("schema_version").value(static_cast<int64_t>(Svc::Fpy::SCHEMA_VERSION));
    out.key("constants").beginObject();
    out.key("MAX_SEQUENCE_ARG_COUNT").value(static_cast<int64_t>(Svc::Fpy::MAX_SEQUENCE_ARG_COUNT));
    out.key("MAX_SEQUENCE_STATEMENT_COUNT")
        .value(static_cast<int64_t>(Svc::Fpy::MAX_SEQUENCE_STATEMENT_COUNT));
    out.key("MAX_STACK_SIZE").value(static_cast<int64_t>(Svc::Fpy::MAX_STACK_SIZE));
    out.key("MAX_DIRECTIVE_SIZE").value(static_cast<int64_t>(Svc::Fpy::MAX_DIRECTIVE_SIZE));
    out.key("MAX_SERIAL_PORTS")
        .value(static_cast<int64_t>(Svc::Fpy::SerialPortIndex::MAX_SERIAL_PORTS));
    out.key("FW_SERIALIZE_TRUE_VALUE").value(static_cast<int64_t>(FW_SERIALIZE_TRUE_VALUE));
    out.key("FW_SERIALIZE_FALSE_VALUE").value(static_cast<int64_t>(FW_SERIALIZE_FALSE_VALUE));
    out.endObject();
    return out.str();
}

std::string writeFailure(int64_t id, const std::string& kind, const std::string& detail) {
    JsonWriter out;
    out.key("id").value(id);
    out.key("ok").value(false);
    out.key("error").value(kind);
    out.key("detail").value(detail);
    return out.str();
}

}  // namespace

int main() {
    ThrowingAssertHook assertHook;
    assertHook.registerHook();

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) {
            continue;
        }
        Json request;
        if (!Json::parse(line, request)) {
            std::cout << writeFailure(0, "bad_request", "malformed JSON") << std::endl;
            continue;
        }
        const int64_t id = request["id"].asInt();
        const std::string op = request["op"].asString();

        if (op == "quit") {
            break;
        }
        if (op == "ping") {
            std::cout << writePing(id) << std::endl;
            continue;
        }
        if (op != "run") {
            std::cout << writeFailure(id, "bad_request", "unknown op " + op) << std::endl;
            continue;
        }

        try {
            Svc::WasmSequencerTester tester;
            const Svc::HarnessResult result = tester.run(readRequest(request));
            std::cout << writeResult(id, result) << std::endl;
        } catch (const AssertFailure& failure) {
            std::cout << writeFailure(id, "assert", failure.what()) << std::endl;
        }
    }
    return 0;
}
