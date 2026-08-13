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
	i32.const	0
	i32.store	x
.LBB0_1:
	block   	
	loop    	
	i32.const	0
	i32.load	x
	i32.const	9
	i32.gt_u
	br_if   	1
	i32.const	0
	i32.const	0
	i64.load32_u	x
	i64.const	1
	i64.add 
	i64.store32	x
	br      	0
.LBB0_3:
	end_loop
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

