	.text
	.attribute	4, 16
	.attribute	5, "rv32i2p0_m2p0_a2p0_f2p0_d2p0_zfh1p0_xcopift0p1_xdma0p1_xfalthalf0p1_xfaltquarter0p1_xfauxalthalf0p1_xfauxaltquarter0p1_xfauxhalf0p1_xfauxquarter0p1_xfauxvecalthalf0p1_xfauxvecaltquarter0p1_xfauxvechalf0p1_xfauxvecquarter0p1_xfauxvecsingle0p1_xfexpauxvecalthalf0p1_xfexpauxvecaltquarter0p1_xfexpauxvechalf0p1_xfexpauxvecquarter0p1_xfquarter0p1_xfrep0p1_xfvecalthalf0p1_xfvecaltquarter0p1_xfvechalf0p1_xfvecquarter0p1_xfvecsingle0p1_xssr0p1"
	.file	"harness.c"
	.globl	matmul_library_kernel           # -- Begin function matmul_library_kernel
	.p2align	2
	.type	matmul_library_kernel,@function
matmul_library_kernel:                  # @matmul_library_kernel
# %bb.0:
	slli	a6, a1, 3
	li	a7, 72
	#APP
	scfgwi	a7, 64

	#NO_APP
	li	a1, 4
	#APP
	scfgwi	a1, 96

	#NO_APP
	li	t0, 3
	#APP
	scfgwi	t0, 128

	#NO_APP
	li	t1, 8
	#APP
	scfgwi	t1, 192

	#NO_APP
	li	t2, -576
	#APP
	scfgwi	t2, 224

	#NO_APP
	addi	a6, a6, -576
	#APP
	scfgwi	a6, 256

	#NO_APP
	li	a6, 7
	#APP
	scfgwi	a6, 32

	#NO_APP
	slli	t2, a3, 3
	#APP
	scfgwi	a6, 65

	#NO_APP
	#APP
	scfgwi	a7, 97

	#NO_APP
	#APP
	scfgwi	a1, 129

	#NO_APP
	#APP
	scfgwi	t0, 161

	#NO_APP
	#APP
	scfgwi	t1, 193

	#NO_APP
	addi	a7, t2, -56
	#APP
	scfgwi	a7, 225

	#NO_APP
	li	a7, 576
	mul	a3, a3, a7
	sub	a7, t1, a3
	#APP
	scfgwi	a7, 257

	#NO_APP
	li	a7, -312
	sub	a3, a7, a3
	#APP
	scfgwi	a3, 289

	#NO_APP
	slli	a3, a5, 3
	#APP
	scfgwi	a6, 66

	#NO_APP
	#APP
	scfgwi	a1, 98

	#NO_APP
	#APP
	scfgwi	t0, 130

	#NO_APP
	#APP
	scfgwi	t1, 194

	#NO_APP
	#APP
	scfgwi	t1, 226

	#NO_APP
	addi	a5, a3, -312
	#APP
	scfgwi	a5, 258

	#NO_APP
	#APP
	scfgwi	a0, 864

	#NO_APP
	#APP
	scfgwi	a2, 865

	#NO_APP
	#APP
	scfgwi	a4, 962

	#NO_APP
	#APP
	csrsi	1984, 1

	#NO_APP
	li	a0, 70
.LBB0_1:                                # =>This Inner Loop Header: Depth=1
	#APP
	fld	ft3, 0(a4)
	fld	ft4, 8(a4)
	fld	ft5, 16(a4)
	fld	ft6, 24(a4)
	fld	ft7, 32(a4)
	fld	fa0, 40(a4)
	fld	fa1, 48(a4)
	fld	fa2, 56(a4)
	fmadd.d	ft3, ft0, ft1, ft3
	fmadd.d	ft4, ft0, ft1, ft4
	fmadd.d	ft5, ft0, ft1, ft5
	fmadd.d	ft6, ft0, ft1, ft6
	fmadd.d	ft7, ft0, ft1, ft7
	fmadd.d	fa0, ft0, ft1, fa0
	fmadd.d	fa1, ft0, ft1, fa1
	fmadd.d	fa2, ft0, ft1, fa2

	#NO_APP
	#APP
	frep.o	a0, 8, 0, 0
	fmadd.d	ft3, ft0, ft1, ft3
	fmadd.d	ft4, ft0, ft1, ft4
	fmadd.d	ft5, ft0, ft1, ft5
	fmadd.d	ft6, ft0, ft1, ft6
	fmadd.d	ft7, ft0, ft1, ft7
	fmadd.d	fa0, ft0, ft1, fa0
	fmadd.d	fa1, ft0, ft1, fa1
	fmadd.d	fa2, ft0, ft1, fa2
	fmadd.d	ft2, ft0, ft1, ft3
	fmadd.d	ft2, ft0, ft1, ft4
	fmadd.d	ft2, ft0, ft1, ft5
	fmadd.d	ft2, ft0, ft1, ft6
	fmadd.d	ft2, ft0, ft1, ft7
	fmadd.d	ft2, ft0, ft1, fa0
	fmadd.d	ft2, ft0, ft1, fa1
	fmadd.d	ft2, ft0, ft1, fa2

	#NO_APP
	addi	a2, a4, 64
	#APP
	fld	ft3, 0(a2)
	fld	ft4, 8(a2)
	fld	ft5, 16(a2)
	fld	ft6, 24(a2)
	fld	ft7, 32(a2)
	fld	fa0, 40(a2)
	fld	fa1, 48(a2)
	fld	fa2, 56(a2)
	fmadd.d	ft3, ft0, ft1, ft3
	fmadd.d	ft4, ft0, ft1, ft4
	fmadd.d	ft5, ft0, ft1, ft5
	fmadd.d	ft6, ft0, ft1, ft6
	fmadd.d	ft7, ft0, ft1, ft7
	fmadd.d	fa0, ft0, ft1, fa0
	fmadd.d	fa1, ft0, ft1, fa1
	fmadd.d	fa2, ft0, ft1, fa2

	#NO_APP
	#APP
	frep.o	a0, 8, 0, 0
	fmadd.d	ft3, ft0, ft1, ft3
	fmadd.d	ft4, ft0, ft1, ft4
	fmadd.d	ft5, ft0, ft1, ft5
	fmadd.d	ft6, ft0, ft1, ft6
	fmadd.d	ft7, ft0, ft1, ft7
	fmadd.d	fa0, ft0, ft1, fa0
	fmadd.d	fa1, ft0, ft1, fa1
	fmadd.d	fa2, ft0, ft1, fa2
	fmadd.d	ft2, ft0, ft1, ft3
	fmadd.d	ft2, ft0, ft1, ft4
	fmadd.d	ft2, ft0, ft1, ft5
	fmadd.d	ft2, ft0, ft1, ft6
	fmadd.d	ft2, ft0, ft1, ft7
	fmadd.d	ft2, ft0, ft1, fa0
	fmadd.d	ft2, ft0, ft1, fa1
	fmadd.d	ft2, ft0, ft1, fa2

	#NO_APP
	addi	a2, a4, 128
	#APP
	fld	ft3, 0(a2)
	fld	ft4, 8(a2)
	fld	ft5, 16(a2)
	fld	ft6, 24(a2)
	fld	ft7, 32(a2)
	fld	fa0, 40(a2)
	fld	fa1, 48(a2)
	fld	fa2, 56(a2)
	fmadd.d	ft3, ft0, ft1, ft3
	fmadd.d	ft4, ft0, ft1, ft4
	fmadd.d	ft5, ft0, ft1, ft5
	fmadd.d	ft6, ft0, ft1, ft6
	fmadd.d	ft7, ft0, ft1, ft7
	fmadd.d	fa0, ft0, ft1, fa0
	fmadd.d	fa1, ft0, ft1, fa1
	fmadd.d	fa2, ft0, ft1, fa2

	#NO_APP
	#APP
	frep.o	a0, 8, 0, 0
	fmadd.d	ft3, ft0, ft1, ft3
	fmadd.d	ft4, ft0, ft1, ft4
	fmadd.d	ft5, ft0, ft1, ft5
	fmadd.d	ft6, ft0, ft1, ft6
	fmadd.d	ft7, ft0, ft1, ft7
	fmadd.d	fa0, ft0, ft1, fa0
	fmadd.d	fa1, ft0, ft1, fa1
	fmadd.d	fa2, ft0, ft1, fa2
	fmadd.d	ft2, ft0, ft1, ft3
	fmadd.d	ft2, ft0, ft1, ft4
	fmadd.d	ft2, ft0, ft1, ft5
	fmadd.d	ft2, ft0, ft1, ft6
	fmadd.d	ft2, ft0, ft1, ft7
	fmadd.d	ft2, ft0, ft1, fa0
	fmadd.d	ft2, ft0, ft1, fa1
	fmadd.d	ft2, ft0, ft1, fa2

	#NO_APP
	addi	a2, a4, 192
	#APP
	fld	ft3, 0(a2)
	fld	ft4, 8(a2)
	fld	ft5, 16(a2)
	fld	ft6, 24(a2)
	fld	ft7, 32(a2)
	fld	fa0, 40(a2)
	fld	fa1, 48(a2)
	fld	fa2, 56(a2)
	fmadd.d	ft3, ft0, ft1, ft3
	fmadd.d	ft4, ft0, ft1, ft4
	fmadd.d	ft5, ft0, ft1, ft5
	fmadd.d	ft6, ft0, ft1, ft6
	fmadd.d	ft7, ft0, ft1, ft7
	fmadd.d	fa0, ft0, ft1, fa0
	fmadd.d	fa1, ft0, ft1, fa1
	fmadd.d	fa2, ft0, ft1, fa2

	#NO_APP
	#APP
	frep.o	a0, 8, 0, 0
	fmadd.d	ft3, ft0, ft1, ft3
	fmadd.d	ft4, ft0, ft1, ft4
	fmadd.d	ft5, ft0, ft1, ft5
	fmadd.d	ft6, ft0, ft1, ft6
	fmadd.d	ft7, ft0, ft1, ft7
	fmadd.d	fa0, ft0, ft1, fa0
	fmadd.d	fa1, ft0, ft1, fa1
	fmadd.d	fa2, ft0, ft1, fa2
	fmadd.d	ft2, ft0, ft1, ft3
	fmadd.d	ft2, ft0, ft1, ft4
	fmadd.d	ft2, ft0, ft1, ft5
	fmadd.d	ft2, ft0, ft1, ft6
	fmadd.d	ft2, ft0, ft1, ft7
	fmadd.d	ft2, ft0, ft1, fa0
	fmadd.d	ft2, ft0, ft1, fa1
	fmadd.d	ft2, ft0, ft1, fa2

	#NO_APP
	addi	a2, a4, 256
	#APP
	fld	ft3, 0(a2)
	fld	ft4, 8(a2)
	fld	ft5, 16(a2)
	fld	ft6, 24(a2)
	fld	ft7, 32(a2)
	fld	fa0, 40(a2)
	fld	fa1, 48(a2)
	fld	fa2, 56(a2)
	fmadd.d	ft3, ft0, ft1, ft3
	fmadd.d	ft4, ft0, ft1, ft4
	fmadd.d	ft5, ft0, ft1, ft5
	fmadd.d	ft6, ft0, ft1, ft6
	fmadd.d	ft7, ft0, ft1, ft7
	fmadd.d	fa0, ft0, ft1, fa0
	fmadd.d	fa1, ft0, ft1, fa1
	fmadd.d	fa2, ft0, ft1, fa2

	#NO_APP
	#APP
	frep.o	a0, 8, 0, 0
	fmadd.d	ft3, ft0, ft1, ft3
	fmadd.d	ft4, ft0, ft1, ft4
	fmadd.d	ft5, ft0, ft1, ft5
	fmadd.d	ft6, ft0, ft1, ft6
	fmadd.d	ft7, ft0, ft1, ft7
	fmadd.d	fa0, ft0, ft1, fa0
	fmadd.d	fa1, ft0, ft1, fa1
	fmadd.d	fa2, ft0, ft1, fa2
	fmadd.d	ft2, ft0, ft1, ft3
	fmadd.d	ft2, ft0, ft1, ft4
	fmadd.d	ft2, ft0, ft1, ft5
	fmadd.d	ft2, ft0, ft1, ft6
	fmadd.d	ft2, ft0, ft1, ft7
	fmadd.d	ft2, ft0, ft1, fa0
	fmadd.d	ft2, ft0, ft1, fa1
	fmadd.d	ft2, ft0, ft1, fa2

	#NO_APP
	addi	a1, a1, -1
	add	a4, a4, a3
	bnez	a1, .LBB0_1
# %bb.2:
	#APP
	fmv.x.w	a0, fa0
	mv	a0, a0

	#NO_APP
	#APP
	csrci	1984, 1

	#NO_APP
	ret
.Lfunc_end0:
	.size	matmul_library_kernel, .Lfunc_end0-matmul_library_kernel
                                        # -- End function
	.ident	"clang version 15.0.0"
	.section	".note.GNU-stack","",@progbits
	.addrsig
