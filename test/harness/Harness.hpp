// The protocol both sequencer harnesses speak with pytest, and the shared
// main loop: one JSON request per stdin line, one JSON reply per stdout line.
// Each request runs on a brand-new sequencer owned by a brand-new tester.

#ifndef TEST_HARNESS_HARNESS_HPP
#define TEST_HARNESS_HARNESS_HPP

#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

#include <Fw/FPrimeBasicTypes.hpp>

#include "test/harness/Json.hpp"

namespace harness {

// One run request, as sent by pytest.
struct HarnessRequest {
    // Path of the compiled sequence file. Must fit in a command string
    // argument (FW_CMD_STRING_MAX_SIZE), like it would for real ground
    // control.
    std::string seqFile;
    // Serialized sequence arguments.
    bool hasArgs = false;
    std::vector<U8> args;
    // Telemetry the spacecraft "has": channel id to serialized value.
    std::map<U32, std::vector<U8>> tlm;
    // Parameters the spacecraft "has": param id to serialized value.
    std::map<U32, std::vector<U8>> prms;
    // The simulated clock's starting value.
    U16 timeBase = 0;
    U8 timeContext = 0;
    U32 seconds = 0;
    U32 useconds = 0;
    // Commands that complete with EXECUTION_ERROR.
    std::set<U32> failOpcodes;
    // Commands that mean "run another sequence". Their arguments are parsed
    // as (fileName, blockState, seqArgs) and the child sequence is run for
    // real on a nested tester; its outcome becomes the command response.
    std::set<U32> seqRunOpcodes;
    // The dictionary's SeqArgs buffer length, needed to parse the arguments
    // of a seq-run command. (The dictionary's buffer length can differ from
    // the flight build's Svc::SeqArgs, so the flight type cannot be used.)
    U32 seqArgsBufferSize = 0;
    // Response for all other commands (an Fw.CmdResponse value, default OK).
    U8 cmdResponse = 0;
};

struct HarnessEvent {
    U32 id = 0;
    I32 severity = 0;
    std::string text;
    // True when the sequence itself emitted this event (its log builtin), in
    // which case text is the raw message rather than formatted event text.
    bool guest = false;
};

struct HarnessSerialWrite {
    I32 port = 0;
    std::vector<U8> data;
};

// Everything that happened during one run.
struct HarnessResult {
    // Harness-level failure (dispatch cap hit, malformed command, ...).
    // Empty when the run itself completed, successfully or not.
    std::string error;
    // The sequencer's response to the RUN command: the same signal a ground
    // station would judge the run by.
    bool gotCmdResponse = false;
    I32 cmdResponse = 0;
    // Final sequencer state, and whether a running state was ever reached
    // (false means the sequence failed validation or loading).
    I32 state = 0;
    bool reachedRunning = false;
    U64 statementsDispatched = 0;
    // The last directive error the sequencer recorded (its telemetry).
    I32 lastDirectiveError = 0;
    // Exit code, present only when the sequence exited with a nonzero code.
    // Reported through the SequenceExitedWithError event: the sequencer has
    // no other path for it.
    bool exited = false;
    I32 exitCode = 0;
    std::vector<HarnessEvent> events;
    // Each command the sequence dispatched: serialized opcode + arguments.
    std::vector<std::vector<U8>> cmds;
    std::vector<HarnessSerialWrite> serial;
    // The bytes left on the sequencer's stack after the run.
    std::vector<U8> stack;
    U32 frameStart = 0;
    U64 sequencesSucceeded = 0;
};

std::vector<U8> hexDecode(const std::string& hex);
std::string hexEncode(const std::vector<U8>& data);

HarnessRequest parseRequest(const JsonValue& json);
JsonValue resultToJson(const HarnessResult& result);

// The main loop: for each stdin line, run the request on a fresh Tester (a
// default-constructed object with a `HarnessResult run(const
// HarnessRequest&)` member) and print the reply.
template <typename Tester>
int harnessMain() {
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) {
            continue;
        }
        JsonValue response;
        try {
            JsonValue json = jsonParse(line);
            HarnessRequest request = parseRequest(json);
            // Relative paths (the sequence file, child sequences launched by
            // the sequence under test) resolve against the request's working
            // directory.
            if (const JsonValue* cwd = json.get("cwd")) {
                if (chdir(cwd->stringValue.c_str()) != 0) {
                    throw std::runtime_error("could not change directory to " + cwd->stringValue);
                }
            }
            Tester tester;
            response = resultToJson(tester.run(request));
        } catch (const std::exception& e) {
            response = JsonValue::makeObject();
            response.set("error", JsonValue::makeString(e.what()));
        }
        std::cout << jsonDump(response) << "\n" << std::flush;
    }
    return 0;
}

}  // namespace harness

#endif
