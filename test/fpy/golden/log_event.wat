	.file	"<string>"
	.functype	exit (i32) -> ()
	.import_module	exit, fprime_v1
	.functype	panic (i32) -> ()
	.import_module	panic, fprime_v1
	.functype	event (i32, i32, i32) -> ()
	.import_module	event, fprime_v1
	.functype	cmd (i32, i32) -> (i32)
	.import_module	cmd, fprime_v1
	.functype	main () -> ()
	.section	.text.main,"",@
	.globl	main
	.type	main,@function
main:
	.functype	main () -> ()
	i32.const	5
	i32.const	.Llog_msg
	i32.const	5
	call	event
	i32.const	2
	i32.const	.Llog_msg.1
	i32.const	3
	call	event
	end_function

	.type	flags,@object
	.section	.data.flags,"",@
	.p2align	3, 0x0
flags:
	.int8	1
	.size	flags, 1

	.type	.Llog_msg,@object
	.section	.rodata..Llog_msg,"",@
.Llog_msg:
	.ascii	"hello"
	.size	.Llog_msg, 5

	.type	.Llog_msg.1,@object
	.section	.rodata..Llog_msg.1,"",@
.Llog_msg.1:
	.ascii	"bad"
	.size	.Llog_msg.1, 3

