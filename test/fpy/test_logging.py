import pytest
from fpy.bytecode.directives import Directive, PopEventDirective, PushValDirective
from fpy.test_helpers import (
    assert_compile_failure,
    assert_run_success,
    compile_seq,
)


class TestLog:

    def test_default_severity(self, fprime_test_api):
        seq = '''
log("hello world")
'''
        assert_run_success(fprime_test_api, seq)

    @pytest.mark.skipif("config.getoption('--use-gds')", reason="FATAL severity kills the program on the GDS")
    def test_explicit_severity(self, fprime_test_api):
        seq = '''
log("oh no", Fw.LogSeverity.FATAL)
'''
        assert_run_success(fprime_test_api, seq)

    def test_default_severity_is_activity_hi(self, fprime_test_api):
        seq = '''
log("test message")
'''
        directives, _ = compile_seq(fprime_test_api, seq)
        push_vals = [d for d in directives if isinstance(d, PushValDirective)]
        assert len(push_vals) >= 3
        # ACTIVITY_HI = 5
        assert push_vals[-3].val == bytes([5])
        assert push_vals[-2].val == b"test message"

    def test_explicit_fatal(self, fprime_test_api):
        seq = '''
log("critical", Fw.LogSeverity.FATAL)
'''
        directives, _ = compile_seq(fprime_test_api, seq)
        push_vals = [d for d in directives if isinstance(d, PushValDirective)]
        assert len(push_vals) >= 3
        # FATAL = 1
        assert push_vals[-3].val == bytes([1])
        assert push_vals[-2].val == b"critical"

    def test_explicit_warning_hi(self, fprime_test_api):
        seq = '''
log("watch out", Fw.LogSeverity.WARNING_HI)
'''
        directives, _ = compile_seq(fprime_test_api, seq)
        push_vals = [d for d in directives if isinstance(d, PushValDirective)]
        assert len(push_vals) >= 3
        # WARNING_HI = 2
        assert push_vals[-3].val == bytes([2])

    def test_emits_pop_event_directive(self, fprime_test_api):
        seq = '''
log("test")
'''
        directives, _ = compile_seq(fprime_test_api, seq)
        pop_dirs = [d for d in directives if isinstance(d, PopEventDirective)]
        assert len(pop_dirs) == 1
        # message_size should be pushed onto the stack before POP_EVENT
        push_vals = [d for d in directives if isinstance(d, PushValDirective)]
        # Last push before POP_EVENT should be the message size (U32 big-endian)
        assert push_vals[-1].val == len(b"test").to_bytes(4, "big")

    def test_serialization_roundtrip(self, fprime_test_api):
        seq = '''
log("roundtrip test")
'''
        directives, _ = compile_seq(fprime_test_api, seq)
        pop_dirs = [d for d in directives if isinstance(d, PopEventDirective)]
        assert len(pop_dirs) == 1

        original = pop_dirs[0]
        serialized = original.serialize()
        _, deserialized = Directive.deserialize(serialized, 0)
        assert isinstance(deserialized, PopEventDirective)

    def test_multiple_events(self, fprime_test_api):
        seq = '''
log("test")
log("test", Fw.LogSeverity.DIAGNOSTIC)
log("test", Fw.LogSeverity.COMMAND)
log("test", Fw.LogSeverity.ACTIVITY_HI)
log("test", Fw.LogSeverity.ACTIVITY_LO)
log("test", Fw.LogSeverity.WARNING_HI)
log("test", Fw.LogSeverity.WARNING_LO)
'''
        assert_run_success(fprime_test_api, seq)

    def test_non_literal_message_rejected(self, fprime_test_api):
        seq = '''
x: U32 = 42
log(x)
'''
        assert_compile_failure(fprime_test_api, seq)

    def test_empty_string(self, fprime_test_api):
        seq = '''
log("")
'''
        assert_run_success(fprime_test_api, seq)
