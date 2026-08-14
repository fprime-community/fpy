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
	.section	.text.main,"",@
	.globl	main
	.type	main,@function
main:
	.functype	main () -> ()
	.local  	i32
	global.get	__stack_pointer
	i32.const	16
	i32.sub 
	local.tee	0
	global.set	__stack_pointer
	i32.const	0
	i64.const	0
	i64.store	total
	local.get	0
	i64.const	0
	i64.store	8
	local.get	0
	i64.const	5
	i64.store	0
.LBB0_1:
	block   	
	loop    	
	local.get	0
	i64.load	8
	local.get	0
	i64.load	0
	i64.ge_s
	br_if   	1
	i32.const	0
	i32.const	0
	i64.load	total
	local.get	0
	i64.load	8
	i64.add 
	i64.store	total
	local.get	0
	local.get	0
	i64.load	8
	i64.const	1
	i64.add 
	i64.store	8
	br      	0
.LBB0_3:
	end_loop
	end_block
	block   	
	i32.const	0
	i64.load	total
	i64.const	10
	i64.eq  
	br_if   	0
	i32.const	7
	call	exit
	unreachable
.LBB0_5:
	end_block
	local.get	0
	i32.const	16
	i32.add 
	global.set	__stack_pointer
	end_function

	.type	flags,@object
	.section	.data.flags,"",@
	.p2align	3, 0x0
flags:
	.int8	1
	.size	flags, 1

	.type	total,@object
	.section	.bss.total,"",@
	.p2align	3, 0x0
total:
	.int64	0
	.size	total, 8

