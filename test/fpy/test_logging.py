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

    def test_explicit_severity(self, fprime_test_api):
        seq = '''
log("oh no", Fw.LogSeverity.FATAL)
'''
        assert_run_success(fprime_test_api, seq)

    def test_default_severity_is_activity_hi(self, fprime_test_api):
        seq = '''
log("test message")
'''
        directives = compile_seq(fprime_test_api, seq)
        push_vals = [d for d in directives if isinstance(d, PushValDirective)]
        assert len(push_vals) >= 2
        # ACTIVITY_HI = 5
        assert push_vals[-2].val == bytes([5])
        assert push_vals[-1].val == b"test message"

    def test_explicit_fatal(self, fprime_test_api):
        seq = '''
log("critical", Fw.LogSeverity.FATAL)
'''
        directives = compile_seq(fprime_test_api, seq)
        push_vals = [d for d in directives if isinstance(d, PushValDirective)]
        assert len(push_vals) >= 2
        # FATAL = 1
        assert push_vals[-2].val == bytes([1])
        assert push_vals[-1].val == b"critical"

    def test_explicit_warning_hi(self, fprime_test_api):
        seq = '''
log("watch out", Fw.LogSeverity.WARNING_HI)
'''
        directives = compile_seq(fprime_test_api, seq)
        push_vals = [d for d in directives if isinstance(d, PushValDirective)]
        assert len(push_vals) >= 2
        # WARNING_HI = 2
        assert push_vals[-2].val == bytes([2])

    def test_emits_pop_event_directive(self, fprime_test_api):
        seq = '''
log("test")
'''
        directives = compile_seq(fprime_test_api, seq)
        pop_dirs = [d for d in directives if isinstance(d, PopEventDirective)]
        assert len(pop_dirs) == 1
        assert pop_dirs[0].message_size == len(b"test")

    def test_serialization_roundtrip(self, fprime_test_api):
        seq = '''
log("roundtrip test")
'''
        directives = compile_seq(fprime_test_api, seq)
        pop_dirs = [d for d in directives if isinstance(d, PopEventDirective)]
        assert len(pop_dirs) == 1

        original = pop_dirs[0]
        serialized = original.serialize()
        _, deserialized = Directive.deserialize(serialized, 0)
        assert isinstance(deserialized, PopEventDirective)
        assert deserialized.message_size == original.message_size

    def test_multiple_events(self, fprime_test_api):
        seq = '''
log("step 1")
log("step 2", Fw.LogSeverity.WARNING_HI)
log("step 3", Fw.LogSeverity.FATAL)
'''
        directives = compile_seq(fprime_test_api, seq)
        pop_dirs = [d for d in directives if isinstance(d, PopEventDirective)]
        assert len(pop_dirs) == 3

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
