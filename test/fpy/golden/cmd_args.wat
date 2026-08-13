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
	i32.const	1069547520
	i32.store	v2
	i32.const	0
	i32.const	-2
	i32.store	v1
	i32.const	0
	i32.const	8
	i32.store8	v3
	i32.const	0
	i32.const	8
	i32.store8	.Lcmd_buf+12
	i32.const	0
	i64.const	211381093662719
	i64.store	.Lcmd_buf+4:p2align=0
	block   	
	block   	
	block   	
	i32.const	.Lcmd_buf
	i32.const	13
	call	cmd
	i32.const	255
	i32.and 
	i32.eqz
	br_if   	0
	i32.const	0
	i32.load8_u	flags
	br_if   	1
.LBB0_2:
	end_block
	block   	
	i32.const	.Lcmd_buf.1
	i32.const	11
	call	cmd
	i32.const	255
	i32.and 
	i32.eqz
	br_if   	0
	i32.const	0
	i32.load8_u	flags
	br_if   	2
.LBB0_4:
	end_block
	return
.LBB0_5:
	end_block
	i32.const	17
	call	exit
	unreachable
.LBB0_6:
	end_block
	i32.const	17
	call	exit
	unreachable
	end_function

	.type	flags,@object
	.section	.data.flags,"",@
	.p2align	3, 0x0
flags:
	.int8	1
	.size	flags, 1

	.type	v1,@object
	.section	.bss.v1,"",@
	.p2align	2, 0x0
v1:
	.int32	0
	.size	v1, 4

	.type	v2,@object
	.section	.bss.v2,"",@
	.p2align	2, 0x0
v2:
	.int32	0x00000000
	.size	v2, 4

	.type	v3,@object
	.section	.bss.v3,"",@
v3:
	.int8	0
	.size	v3, 1

	.type	.Lcmd_buf,@object
	.section	.data..Lcmd_buf,"",@
.Lcmd_buf:
	.asciz	"\001\000\000\002\000\000\000\000\000\000\000\000"
	.size	.Lcmd_buf, 13

	.type	.Lcmd_buf.1,@object
	.section	.rodata..Lcmd_buf.1,"",@
.Lcmd_buf.1:
	.ascii	"\001\000\000\001\000\005hello"
	.size	.Lcmd_buf.1, 11

