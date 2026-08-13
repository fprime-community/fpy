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
	i32.const	.Lcmd_buf
	i32.const	4
	call	cmd
	local.tee	0
	i32.store8	ret
	block   	
	local.get	0
	i32.const	255
	i32.and 
	br_if   	0
	i32.const	0
	call	exit
	unreachable
.LBB0_2:
	end_block
	i32.const	1
	call	exit
	unreachable
	end_function

	.type	flags,@object
	.section	.data.flags,"",@
	.p2align	3, 0x0
flags:
	.int8	1
	.size	flags, 1

	.type	ret,@object
	.section	.bss.ret,"",@
ret:
	.int8	0
	.size	ret, 1

	.type	.Lcmd_buf,@object
	.section	.rodata..Lcmd_buf,"",@
.Lcmd_buf:
	.asciz	"\001\000\000"
	.size	.Lcmd_buf, 4

