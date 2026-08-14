	.file	"<string>"
	.functype	pow (f64, f64) -> (f64)
	.functype	fpy_main () -> (i32)
	.section	.text.fpy_main,"",@
	.globl	fpy_main
	.type	fpy_main,@function
fpy_main:
	.functype	fpy_main () -> (i32)
	i32.const	0
	i32.const	1073741824
	i32.store	val
	i32.const	0
	i32.const	1123418112
	i32.store	val0
	i32.const	0
	f64.const	0x1.ecp6
	f64.const	0x1p1
	call	pow
	f64.store	val3
	i32.const	0
	end_function

	.type	flags,@object
	.section	.data.flags,"",@
	.p2align	3, 0x0
flags:
	.int8	1
	.size	flags, 1

	.type	val0,@object
	.section	.bss.val0,"",@
	.p2align	2, 0x0
val0:
	.int32	0x00000000
	.size	val0, 4

	.type	val,@object
	.section	.bss.val,"",@
	.p2align	2, 0x0
val:
	.int32	0x00000000
	.size	val, 4

	.type	val3,@object
	.section	.bss.val3,"",@
	.p2align	3, 0x0
val3:
	.int64	0x0000000000000000
	.size	val3, 8

