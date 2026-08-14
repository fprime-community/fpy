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
	.functype	add_one (i32) -> (i32)
	.section	.text.main,"",@
	.globl	main
	.type	main,@function
main:
	.functype	main () -> ()
	i32.const	0
	i32.const	5
	call	add_one
	i32.store	result
	end_function

	.section	.text.add_one,"",@
	.type	add_one,@function
add_one:
	.functype	add_one (i32) -> (i32)
	global.get	__stack_pointer
	i32.const	16
	i32.sub 
	local.get	0
	i32.store	12
	local.get	0
	i32.const	1
	i32.add 
	end_function

	.type	flags,@object
	.section	.data.flags,"",@
	.p2align	3, 0x0
flags:
	.int8	1
	.size	flags, 1

	.type	result,@object
	.section	.bss.result,"",@
	.p2align	2, 0x0
result:
	.int32	0
	.size	result, 4

