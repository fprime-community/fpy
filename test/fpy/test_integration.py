from fpy.model import DirectiveErrorCode
from fpy.types import FpyValue, U32

from fpy.test_helpers import assert_run_success, assert_run_failure


class TestReadmeExamples:

    def test_readme_examples(self, fprime_test_api):
        seq = """
Ref.sendBuffComp.PARAMETER4_PRM_SET(1 - 2 + 3 * 4 + 10 / 5 * 2)
param4: F32 = 15.0
Ref.sendBuffComp.PARAMETER4_PRM_SET(param4)

#prm_3: U8 = Ref.sendBuffComp.parameter3
#cmds_dispatched: U32 = CdhCore.cmdDisp.CommandsDispatched
cmds_dispatched: U32 = 0

signal_pair: Ref.SignalPair = Ref.SignalPair(0, 0)

signal_pair.time = 0.2

# Svc.ComQueueDepth is an array type
com_queue_depth: Svc.ComQueueDepth = Svc.ComQueueDepth(0, 0)
com_queue_depth[0] = 1
#signal_pair_time: F32 = Ref.SG1.PairOutput.time
#com_queue_depth_0: U32 = ComCcsds.comQueue.comQueueDepth[0]

int_value: U8 = 123
float_value: F32 = int_value
int_value = U8(float_value)
assert int_value == 123

uint: U32 = 123123
int: I32 = I32(uint)
assert int == 123123

CdhCore.cmdDisp.CMD_NO_OP_STRING("second 0")
# sleep for 1 second
sleep(1)
CdhCore.cmdDisp.CMD_NO_OP_STRING("second 1")
# sleep for half a second
sleep(useconds=500_000)

value: bool = 1 > 2 and (3 + 4) != 5
many_cmds_dispatched: bool = cmds_dispatched >= 123
record1: Svc.DpRecord = Svc.DpRecord(0, 1, 2, 3, 4, 5, Fw.DpState.UNTRANSMITTED)
record2: Svc.DpRecord = Svc.DpRecord(0, 1, 2, 3, 4, 5, Fw.DpState.UNTRANSMITTED)
records_equal: bool = record1 == record2 # == True
random_value: I8 = 4 # chosen by fair dice roll. guaranteed to be random

if random_value < 0:
    CdhCore.cmdDisp.CMD_NO_OP_STRING("won't happen")
elif random_value > 0 and random_value <= 6:
    CdhCore.cmdDisp.CMD_NO_OP_STRING("should happen!")
else:
    CdhCore.cmdDisp.CMD_NO_OP_STRING("uh oh...")

time_interval: Fw.TimeInterval = {seconds: 15, useconds: 1000}

array_var: Ref.DpDemo.U32Array = [0, 1, 2, 3, 4]

counter: U64 = 0
while counter < 100:
    counter = counter + 1

assert counter == 100
sum: I64 = 0
# loop i from 0 inclusive to 5 exclusive
for i in 0 .. 5:
    sum = sum + i

assert sum == 10
counter = 0
while True:
    counter = counter + 1
    if counter == 100:
        break

assert counter == 100
odd_numbers_sum: I64 = 0
for i in 0 .. 10:
    if i % 2 == 0:
        continue
    odd_numbers_sum = odd_numbers_sum + i

assert odd_numbers_sum == 25

low_bitwidth_int: U8 = 123
high_bitwidth_int: U32 = low_bitwidth_int
# high_bitwidth_int == 123
low_bitwidth_float: F32 = 123.0
high_bitwidth_float: F64 = low_bitwidth_float
# high_bitwidth_float == 123.0

# Global variable for increment example - must be declared before functions that use it
counter_global: I64 = 0

def foobar():
    if 1 + 2 == 3:
        CdhCore.cmdDisp.CMD_NO_OP_STRING("foo")

foobar()

def add_vals(a: U64, b: U64) -> U64:
    return a + b
    
assert add_vals(1, 2) == 3

def greet(times: I64 = 3):
    for i in 0..times:
        CdhCore.cmdDisp.CMD_NO_OP_STRING("hello")

greet()  # uses default: prints 3 times
greet(1) # prints once

def increment():
    counter_global = counter_global + 1

increment()
increment()
assert counter_global == 2

def recurse(limit: U64):
    if limit == 0:
        return
    CdhCore.cmdDisp.CMD_NO_OP_STRING("tick")
    recurse(limit - 1)

recurse(5) # prints "tick" 5 times


check CdhCore.cmdDisp.CommandsDispatched > 1 persist {seconds: 1}:
    CdhCore.cmdDisp.CMD_NO_OP_STRING("more than 30 commands for 15 seconds!")
check CdhCore.cmdDisp.CommandsDispatched > 1 timeout now() + {seconds: 60} persist {seconds: 1}:
    CdhCore.cmdDisp.CMD_NO_OP_STRING("more than 30 commands for 2 seconds!")
check CdhCore.cmdDisp.CommandsDispatched > 1 timeout now() + {seconds: 60} persist {seconds: 1}:
    CdhCore.cmdDisp.CMD_NO_OP_STRING("more than 30 commands for 2 seconds!")
timeout:
    CdhCore.cmdDisp.CMD_NO_OP_STRING("took more than 60 seconds :(")
check CdhCore.cmdDisp.CommandsDispatched > 1 freq {seconds: 1}: # check every 1 second
    CdhCore.cmdDisp.CMD_NO_OP_STRING("more than 30 commands!")
check CdhCore.cmdDisp.CommandsDispatched > 1
    timeout now() + {seconds: 60}
    persist {seconds: 1}
    freq {seconds: 1}:
    CdhCore.cmdDisp.CMD_NO_OP_STRING("more than 30 commands for 2 seconds!")
timeout:
    CdhCore.cmdDisp.CMD_NO_OP_STRING("took more than 60 seconds :(")

# Time functions examples
current_time: Fw.Time = now()
t1: Fw.Time = now()
sleep(seconds=1)
t2: Fw.Time = now()

assert t1 <= t2
interval1: Fw.TimeInterval = {seconds: 5}
interval2: Fw.TimeInterval = {seconds: 10}

assert interval1 < interval2
current: Fw.Time = {timeBase: TimeBase.TB_PROC_TIME, timeContext: 0, seconds: 100, useconds: 500000}
offset: Fw.TimeInterval = {seconds: 60}
assert (current + offset).seconds == 160
start: Fw.Time = {timeBase: TimeBase.TB_PROC_TIME, timeContext: 0, seconds: 100, useconds: 0}
end: Fw.Time = {timeBase: TimeBase.TB_PROC_TIME, timeContext: 0, seconds: 105, useconds: 500000}
assert (end - start).seconds == 5

# Named argument command call
CdhCore.cmdDisp.CMD_NO_OP_STRING(arg1="Hello world!")

# flags.assert_cmd_success = False allows failing commands to proceed
flags.assert_cmd_success = False
Ref.cmdSeq.RUN("", Svc.FpySequencer.BlockState.NO_BLOCK)
# sequence proceeds normally

# Handling the return value suppresses auto-assert even with flag True
flags.assert_cmd_success = True
success: Fw.CmdResponse = Ref.cmdSeq.RUN("", Svc.FpySequencer.BlockState.NO_BLOCK)
# cmd response is handled, sequence proceeds normally
"""
        assert_run_success(
            fprime_test_api,
            seq,
            {"CdhCore.cmdDisp.CommandsDispatched": FpyValue(U32, 45).serialize()},
            timeout_s=20
        )

    def test_readme_bare_cmd_fail_exits(self, fprime_test_api):
        """Unhandled failing command with assert_cmd_success=True exits with error."""
        seq = """
flags.assert_cmd_success = True
Ref.cmdSeq.RUN("", Svc.FpySequencer.BlockState.NO_BLOCK)
# sequence exits with an error
"""
        assert_run_failure(
            fprime_test_api, seq, DirectiveErrorCode.EXIT_WITH_ERROR,
        )
