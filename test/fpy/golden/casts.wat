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
	i64.const	4618328827877759386
	i64.store	f
	block   	
	block   	
	block   	
	i32.const	1
	i32.eqz
	br_if   	0
	i32.const	0
	i32.const	300
	i32.store	n
	i32.const	1
	i32.eqz
	br_if   	1
	i32.const	0
	i32.load	n
	f64.convert_i32_s
	f64.const	0x1.2cp8
	f64.ne  
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

	.type	f,@object
	.section	.bss.f,"",@
	.p2align	3, 0x0
f:
	.int64	0x0000000000000000
	.size	f, 8

	.type	n,@object
	.section	.bss.n,"",@
	.p2align	2, 0x0
n:
	.int32	0
	.size	n, 4

