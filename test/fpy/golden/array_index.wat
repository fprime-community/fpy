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
	i64.const	34359738375
	i64.store	a:p2align=2
	i32.const	0
	i32.const	1
	i32.store8	i
	block   	
	block   	
	block   	
	block   	
	block   	
	block   	
	i32.const	0
	br_if   	0
	i64.const	1
	i64.const	2
	i64.ge_s
	br_if   	0
	i32.const	0
	i64.load8_s	i
	local.tee	0
	i64.const	0
	i64.lt_s
	br_if   	1
	local.get	0
	i64.const	2
	i64.ge_s
	br_if   	1
	local.get	0
	i32.wrap_i64
	i32.const	2
	i32.shl 
	i32.const	a
	i32.add 
	i64.const	1
	i32.wrap_i64
	i32.const	2
	i32.shl 
	i32.const	a
	i32.add 
	i32.load	0
	i32.const	1
	i32.add 
	i32.store	0
	i32.const	0
	br_if   	2
	i32.const	1
	i32.eqz
	br_if   	2
	i32.const	0
	i32.load	a
	i32.const	7
	i32.ne  
	br_if   	3
	i32.const	0
	br_if   	4
	i32.const	1
	i32.eqz
	br_if   	4
	i32.const	0
	i32.load	a+4
	i32.const	9
	i32.ne  
	br_if   	5
	return
.LBB0_11:
	end_block
	i32.const	11
	call	panic
	unreachable
.LBB0_12:
	end_block
	i32.const	11
	call	panic
	unreachable
.LBB0_13:
	end_block
	i32.const	11
	call	panic
	unreachable
.LBB0_14:
	end_block
	i32.const	7
	call	exit
	unreachable
.LBB0_15:
	end_block
	i32.const	11
	call	panic
	unreachable
.LBB0_16:
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

	.type	a,@object
	.section	.bss.a,"",@
	.p2align	2, 0x0
a:
	.skip	8
	.size	a, 8

	.type	i,@object
	.section	.bss.i,"",@
i:
	.int8	0
	.size	i, 1

