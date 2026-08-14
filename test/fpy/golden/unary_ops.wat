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
	i32.const	0
	i64.const	5
	i64.store	x
	block   	
	block   	
	block   	
	i32.const	1
	i32.eqz
	br_if   	0
	i32.const	0
	i64.load	x
	i64.const	5
	i64.ne  
	br_if   	1
	i32.const	0
	i32.const	0
	i32.store8	b
	i32.const	1
	i32.eqz
	br_if   	2
	return
.LBB0_4:
	end_block
	i32.const	7
	call	exit
	unreachable
.LBB0_5:
	end_block
	i32.const	7
	call	exit
	unreachable
.LBB0_6:
	end_block
	i32.const	7
	call	exit
	unreachable
	end_function

	.type	flags,@object
	.section	.data.flags,"",@
	.p2align	3, 0x0
flags:
	.int8	1
	.size	flags, 1

	.type	x,@object
	.section	.bss.x,"",@
	.p2align	3, 0x0
x:
	.int64	0
	.size	x, 8

	.type	b,@object
	.section	.bss.b,"",@
b:
	.int8	0
	.size	b, 1

