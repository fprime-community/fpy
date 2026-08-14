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
	.local  	i64
	i32.const	0
	i64.const	0
	i64.store	evens
	i32.const	0
	i64.const	0
	i64.store	n
.LBB0_1:
	block   	
	block   	
	loop    	
	i32.const	0
	i32.const	0
	i64.load	n
	i64.const	1
	i64.add 
	local.tee	0
	i64.store	n
	block   	
	local.get	0
	i64.const	10
	i64.lt_u
	br_if   	0
	i32.const	0
	i64.load	evens
	i64.const	4
	i64.eq  
	br_if   	2
	i32.const	7
	call	exit
	unreachable
.LBB0_4:
	end_block
	i32.const	1
	i32.eqz
	br_if   	2
	i32.const	0
	i64.load	n
	i32.wrap_i64
	i32.const	1
	i32.and 
	br_if   	0
	i32.const	0
	i32.const	0
	i64.load	evens
	i64.const	1
	i64.add 
	i64.store	evens
	br      	0
.LBB0_7:
	end_loop
	end_block
	return
.LBB0_8:
	end_block
	i32.const	10
	call	panic
	unreachable
	end_function

	.type	flags,@object
	.section	.data.flags,"",@
	.p2align	3, 0x0
flags:
	.int8	1
	.size	flags, 1

	.type	n,@object
	.section	.bss.n,"",@
	.p2align	3, 0x0
n:
	.int64	0
	.size	n, 8

	.type	evens,@object
	.section	.bss.evens,"",@
	.p2align	3, 0x0
evens:
	.int64	0
	.size	evens, 8

