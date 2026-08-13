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
	.local  	i32
	i32.const	0
	i64.const	5
	i64.store	x
	i32.const	0
	local.set	0
	block   	
	i32.const	0
	br_if   	0
	i32.const	0
	i64.load	x
	i64.const	10
	i64.gt_u
	local.set	0
.LBB0_2:
	end_block
	i32.const	0
	local.get	0
	i32.store8	ok
	block   	
	block   	
	local.get	0
	br_if   	0
	i32.const	1
	local.set	0
	block   	
	i32.const	0
	i64.load	x
	i64.const	1
	i64.eq  
	br_if   	0
	i32.const	0
	i64.load	x
	i64.const	5
	i64.eq  
	local.set	0
.LBB0_5:
	end_block
	local.get	0
	i32.eqz
	br_if   	1
	return
.LBB0_7:
	end_block
	i32.const	7
	call	exit
	unreachable
.LBB0_8:
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

	.type	ok,@object
	.section	.bss.ok,"",@
ok:
	.int8	0
	.size	ok, 1

