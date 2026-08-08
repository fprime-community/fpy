from fpy.bytecode.directives import Directive, PopEventDirective, PushValDirective
from fpy.test_helpers import (
    assert_compile_failure,
    assert_run_success,
    compile_seq,
)


class TestLog:

    def test_default_severity(self):
        seq = """
log("hello world")
"""
        assert_run_success(seq)

    def test_explicit_severity(self):
        seq = """
log("oh no", Fw.LogSeverity.FATAL)
"""
        assert_run_success(seq)

    def test_default_severity_is_activity_hi(self):
        seq = """
log("test message")
"""
        _, directives, _ = compile_seq(seq)
        push_vals = [d for d in directives if isinstance(d, PushValDirective)]
        assert len(push_vals) >= 3
        # ACTIVITY_HI = 5
        assert push_vals[-3].val == bytes([5])
        assert push_vals[-2].val == b"test message"

    def test_explicit_fatal(self):
        seq = """
log("critical", Fw.LogSeverity.FATAL)
"""
        _, directives, _ = compile_seq(seq)
        push_vals = [d for d in directives if isinstance(d, PushValDirective)]
        assert len(push_vals) >= 3
        # FATAL = 1
        assert push_vals[-3].val == bytes([1])
        assert push_vals[-2].val == b"critical"

    def test_explicit_warning_hi(self):
        seq = """
log("watch out", Fw.LogSeverity.WARNING_HI)
"""
        _, directives, _ = compile_seq(seq)
        push_vals = [d for d in directives if isinstance(d, PushValDirective)]
        assert len(push_vals) >= 3
        # WARNING_HI = 2
        assert push_vals[-3].val == bytes([2])

    def test_emits_pop_event_directive(self):
        seq = """
log("test")
"""
        _, directives, _ = compile_seq(seq)
        pop_dirs = [d for d in directives if isinstance(d, PopEventDirective)]
        assert len(pop_dirs) == 1
        # message_size should be pushed onto the stack before POP_EVENT
        push_vals = [d for d in directives if isinstance(d, PushValDirective)]
        # Last push before POP_EVENT should be the message size (U32 big-endian)
        assert push_vals[-1].val == len(b"test").to_bytes(4, "big")

    def test_serialization_roundtrip(self):
        seq = """
log("roundtrip test")
"""
        _, directives, _ = compile_seq(seq)
        pop_dirs = [d for d in directives if isinstance(d, PopEventDirective)]
        assert len(pop_dirs) == 1

        original = pop_dirs[0]
        serialized = original.serialize()
        _, deserialized = Directive.deserialize(serialized, 0)
        assert isinstance(deserialized, PopEventDirective)

    def test_multiple_events(self):
        seq = """
log("test")
log("test", Fw.LogSeverity.DIAGNOSTIC)
log("test", Fw.LogSeverity.COMMAND)
log("test", Fw.LogSeverity.ACTIVITY_HI)
log("test", Fw.LogSeverity.ACTIVITY_LO)
log("test", Fw.LogSeverity.WARNING_HI)
log("test", Fw.LogSeverity.WARNING_LO)
"""
        assert_run_success(seq)

    def test_non_literal_message_rejected(self):
        seq = """
x: U32 = 42
log(x)
"""
        assert_compile_failure(seq)

    def test_empty_string(self):
        seq = """
log("")
"""
        assert_run_success(seq)
