	.file	"<string>"
	.functype	pow (f64, f64) -> (f64)
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
	.local  	i64, i64, i64
	i32.const	0
	i64.const	5
	i64.store	y
	i32.const	0
	i64.const	17
	i64.store	x
	block   	
	block   	
	block   	
	block   	
	block   	
	block   	
	block   	
	block   	
	block   	
	block   	
	block   	
	block   	
	block   	
	i32.const	1
	i32.eqz
	br_if   	0
	i32.const	0
	i64.load	x
	i32.const	0
	i64.load	y
	i64.sub 
	i64.const	12
	i64.ne  
	br_if   	1
	i32.const	0
	i64.load	x
	i32.const	0
	i64.load	y
	i64.mul 
	i64.const	85
	i64.ne  
	br_if   	2
	i32.const	0
	i64.load	y
	local.tee	0
	i64.const	0
	i64.eq  
	br_if   	3
	i32.const	0
	i64.load	x
	local.tee	1
	local.get	0
	i64.div_s
	local.tee	2
	i64.const	-1
	i64.add 
	local.get	2
	local.get	1
	local.get	0
	i64.xor 
	i64.const	0
	i64.lt_s
	i64.select
	local.get	2
	local.get	1
	local.get	0
	i64.rem_s
	i64.const	0
	i64.ne  
	i64.select
	i64.const	3
	i64.ne  
	br_if   	4
	i32.const	0
	i64.load	y
	local.tee	1
	i64.const	0
	i64.eq  
	br_if   	5
	i32.const	0
	i64.load	x
	local.get	1
	i64.rem_s
	local.tee	0
	local.get	1
	i64.add 
	local.get	0
	local.get	0
	local.get	1
	i64.xor 
	i64.const	0
	i64.lt_s
	i64.select
	local.get	0
	local.get	0
	i64.const	0
	i64.ne  
	i64.select
	i64.const	2
	i64.ne  
	br_if   	6
	i32.const	0
	i64.const	-17
	i64.store	neg
	i32.const	0
	i64.load	y
	local.tee	0
	i64.const	0
	i64.eq  
	br_if   	7
	i64.const	-17
	local.get	0
	i64.div_s
	local.tee	1
	i64.const	-1
	i64.add 
	local.get	1
	i64.const	-17
	local.get	0
	i64.xor 
	i64.const	0
	i64.lt_s
	i64.select
	local.get	1
	i64.const	-17
	local.get	0
	i64.rem_s
	i64.const	0
	i64.ne  
	i64.select
	i64.const	-4
	i64.ne  
	br_if   	8
	i32.const	0
	i64.load	y
	local.tee	1
	i64.const	0
	i64.eq  
	br_if   	9
	i32.const	0
	i64.load	neg
	local.get	1
	i64.rem_s
	local.tee	0
	local.get	1
	i64.add 
	local.get	0
	local.get	0
	local.get	1
	i64.xor 
	i64.const	0
	i64.lt_s
	i64.select
	local.get	0
	local.get	0
	i64.const	0
	i64.ne  
	i64.select
	i64.const	3
	i64.ne  
	br_if   	10
	i32.const	0
	i64.const	4619567317775286272
	i64.store	f
	i32.const	1
	i32.eqz
	br_if   	11
	i32.const	0
	f64.load	f
	f64.const	0x1p1
	call	pow
	f64.const	0x1.88p5
	f64.ne  
	br_if   	12
	return
.LBB0_14:
	end_block
	i32.const	7
	call	exit
	unreachable
.LBB0_15:
	end_block
	i32.const	7
	call	exit
	unreachable
.LBB0_16:
	end_block
	i32.const	7
	call	exit
	unreachable
.LBB0_17:
	end_block
	i32.const	10
	call	panic
	unreachable
.LBB0_18:
	end_block
	i32.const	7
	call	exit
	unreachable
.LBB0_19:
	end_block
	i32.const	10
	call	panic
	unreachable
.LBB0_20:
	end_block
	i32.const	7
	call	exit
	unreachable
.LBB0_21:
	end_block
	i32.const	10
	call	panic
	unreachable
.LBB0_22:
	end_block
	i32.const	7
	call	exit
	unreachable
.LBB0_23:
	end_block
	i32.const	10
	call	panic
	unreachable
.LBB0_24:
	end_block
	i32.const	7
	call	exit
	unreachable
.LBB0_25:
	end_block
	i32.const	7
	call	exit
	unreachable
.LBB0_26:
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

	.type	y,@object
	.section	.bss.y,"",@
	.p2align	3, 0x0
y:
	.int64	0
	.size	y, 8

	.type	neg,@object
	.section	.bss.neg,"",@
	.p2align	3, 0x0
neg:
	.int64	0
	.size	neg, 8

	.type	f,@object
	.section	.bss.f,"",@
	.p2align	3, 0x0
f:
	.int64	0x0000000000000000
	.size	f, 8

