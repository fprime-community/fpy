	.file	"<string>"
	.globaltype	__stack_pointer, i32
	.functype	exit (i32) -> ()
	.import_module	exit, fprime_v1
	.functype	panic (i32) -> ()
	.import_module	panic, fprime_v1
	.functype	event (i32, i32, i32) -> ()
	.import_module	event, fprime_v1
	.functype	cmd (i32, i32) -> (i32)
	.import_module	cmd, fprime_v1
	.functype	main () -> ()
	.functype	fib (i64) -> (i64)
	.section	.text.main,"",@
	.globl	main
	.type	main,@function
main:
	.functype	main () -> ()
	block   	
	i64.const	10
	call	fib
	i64.const	89
	i64.eq  
	br_if   	0
	i32.const	7
	call	exit
	unreachable
.LBB0_2:
	end_block
	end_function

	.section	.text.fib,"",@
	.type	fib,@function
fib:
	.functype	fib (i64) -> (i64)
	.local  	i32, i64
	global.get	__stack_pointer
	i32.const	16
	i32.sub 
	local.tee	1
	global.set	__stack_pointer
	local.get	1
	local.get	0
	i64.store	8
	block   	
	local.get	0
	i64.const	1
	i64.gt_u
	br_if   	0
	local.get	1
	i32.const	16
	i32.add 
	global.set	__stack_pointer
	i64.const	1
	return
.LBB1_2:
	end_block
	local.get	1
	i64.load	8
	i64.const	-1
	i64.add 
	call	fib
	local.set	0
	local.get	1
	i64.load	8
	i64.const	-2
	i64.add 
	call	fib
	local.set	2
	local.get	1
	i32.const	16
	i32.add 
	global.set	__stack_pointer
	local.get	0
	local.get	2
	i64.add 
	end_function

	.type	flags,@object
	.section	.data.flags,"",@
	.p2align	3, 0x0
flags:
	.int8	1
	.size	flags, 1

