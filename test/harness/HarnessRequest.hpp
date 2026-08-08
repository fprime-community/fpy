// ======================================================================
// \title  HarnessRequest.hpp
// \brief  What a harness is asked to run, and what came of it
//
// Shared by every sequencer harness so the wire protocol stays one shape
// whichever component is running the sequence.
// ======================================================================

#ifndef FPY_HARNESS_REQUEST_HPP
#define FPY_HARNESS_REQUEST_HPP

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "Fw/FPrimeBasicTypes.hpp"

namespace Svc {

//! What a sequence run should see: the sequence itself, the values its
//! environment answers with, and how dispatched commands complete.
struct HarnessRequest {
    std::string seqPath;
    //! Directory to run from. The sequencer receives paths through a 40-char
    //! command string, so seqPath is kept short and resolved against this.
    std::string cwd;
    std::vector<uint8_t> args;

    std::map<FwChanIdType, std::vector<uint8_t>> tlm;
    std::map<FwPrmIdType, std::vector<uint8_t>> prm;

    U16 timeBase = 0;
    U8 timeContext = 0;
    U64 initialTimeUs = 0;

    //! Opcodes that complete with EXECUTION_ERROR.
    std::set<FwOpcodeType> failOpcodes;
    //! Opcodes that name a child sequence to run rather than a command to
    //! complete. The child's outcome becomes the command's response.
    std::set<FwOpcodeType> seqRunOpcodes;
    //! The Fw::CmdResponse every other command completes with.
    U32 cmdResponse = 0;
    //! Serial ports left unconnected.
    std::set<FwIndexType> disconnectedSerialPorts;

    //! Upper bound on queue dispatches, so a sequence that cannot make
    //! progress ends the run instead of hanging.
    U32 maxDispatch = 200000;
    //! How deep child sequences may nest.
    U32 depth = 0;
};

//! What a sequence run produced.
struct HarnessResult {
    bool ok = false;
    //! The component's last directive error.
    U8 errorCode = 0;
    //! The code an exit() reported, read off SequenceExitedWithError.
    bool hasExitCode = false;
    I32 exitCode = 0;
    //! The sequence failed before it began running.
    bool validationFailed = false;
    //! Bytes still on the component's stack when the run ended.
    U32 stackSize = 0;
    U64 statementsDispatched = 0;
    std::string finalState;
    //! Why the run ended, when it did not end on its own.
    std::string error;

    std::vector<std::vector<uint8_t>> cmds;
    std::vector<std::pair<U32, std::string>> events;
    std::vector<std::pair<FwIndexType, std::vector<uint8_t>>> serialWrites;

    U64 simTimeUs = 0;
};

}  // namespace Svc

#endif
