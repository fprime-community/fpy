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
	i32.const	1
	i32.store	x
	block   	
	i32.const	1
	i32.eqz
	br_if   	0
	i32.const	0
	i32.const	2
	i32.store	x
.LBB0_2:
	end_block
	end_function

	.type	flags,@object
	.section	.data.flags,"",@
	.p2align	3, 0x0
flags:
	.int8	1
	.size	flags, 1

	.type	x,@object
	.section	.bss.x,"",@
	.p2align	2, 0x0
x:
	.int32	0
	.size	x, 4

