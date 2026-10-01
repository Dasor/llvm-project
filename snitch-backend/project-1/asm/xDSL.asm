.text
.globl matmul$xdsl_kernel0
.p2align 2
matmul$xdsl_kernel0:
    mv t0, a0
    mv t4, a1
    mv t2, a2
    li t1, 72
    scfgwi t1, 64                                # dm 0 dim 0 bound
    li t1, 9
    scfgwi t1, 96                                # dm 0 dim 1 bound
    li t1, 3
    scfgwi t1, 128                               # dm 0 dim 2 bound
    li t1, 8
    scfgwi t1, 192                               # dm 0 dim 0 stride
    li t1, -576
    scfgwi t1, 224                               # dm 0 dim 1 stride
    li t1, 8
    scfgwi t1, 256                               # dm 0 dim 2 stride
    li t1, 3
    scfgwi t1, 32                                # dm 0 repeat
    li t1, 3
    scfgwi t1, 65                                # dm 1 dim 0 bound
    li t1, 72
    scfgwi t1, 97                                # dm 1 dim 1 bound
    li t1, 9
    scfgwi t1, 129                               # dm 1 dim 2 bound
    li t1, 3
    scfgwi t1, 161                               # dm 1 dim 3 bound
    li t1, 8
    scfgwi t1, 193                               # dm 1 dim 0 stride
    li t1, 296
    scfgwi t1, 225                               # dm 1 dim 1 stride
    li t1, -23032
    scfgwi t1, 257                               # dm 1 dim 2 stride
    li t1, -23352
    scfgwi t1, 289                               # dm 1 dim 3 stride
    scfgwi zero, 33                              # dm 1 repeat
    scfgwi t0, 832                               # dm 0 dim 2 source
    scfgwi t4, 865                               # dm 1 dim 3 source
    csrrsi zero, 1984, 1                         # SSR enable
    li t4, 4
    li t0, 10
    mv t3, zero
    # Constant folded riscv_cf.bge
scf_body_1_for:
    mv t5, zero
    # Constant folded riscv_cf.bge
scf_body_0_for:
    li t6, 4
    mul t6, t5, t6
    li a3, 40
    mul a3, t3, a3
    add a3, a3, t6
    li t6, 8
    mul a3, a3, t6                               # multiply by element size
    add a3, t2, a3
    fld ft6, 0(a3)                               # load double from memref of shape (4, 40)
    li a3, 4
    mul a3, t5, a3
    addi a3, a3, 1
    li t6, 40
    mul t6, t3, t6
    add t6, t6, a3
    li a3, 8
    mul t6, t6, a3                               # multiply by element size
    add t6, t2, t6
    fld ft5, 0(t6)                               # load double from memref of shape (4, 40)
    li t6, 4
    mul t6, t5, t6
    addi t6, t6, 2
    li a3, 40
    mul a3, t3, a3
    add a3, a3, t6
    li t6, 8
    mul a3, a3, t6                               # multiply by element size
    add a3, t2, a3
    fld ft4, 0(a3)                               # load double from memref of shape (4, 40)
    li a3, 4
    mul a3, t5, a3
    addi a3, a3, 3
    li t6, 40
    mul t6, t3, t6
    add t6, t6, a3
    li a3, 8
    mul t6, t6, a3                               # multiply by element size
    add t6, t2, t6
    fld ft3, 0(t6)                               # load double from memref of shape (4, 40)
    li t6, 72
    frep.o t6, 4, 0, 0
    fmadd.d ft6, ft0, ft1, ft6
    fmadd.d ft5, ft0, ft1, ft5
    fmadd.d ft4, ft0, ft1, ft4
    fmadd.d ft3, ft0, ft1, ft3
    li t6, 4
    mul t6, t5, t6
    li a3, 40
    mul a3, t3, a3
    add a3, a3, t6
    li t6, 8
    mul a3, a3, t6                               # multiply by element size
    add a3, t2, a3
    fsd ft6, 0(a3)                               # store double value to memref of shape (4, 40)
    li a3, 4
    mul a3, t5, a3
    addi a3, a3, 1
    li t6, 40
    mul t6, t3, t6
    add t6, t6, a3
    li a3, 8
    mul t6, t6, a3                               # multiply by element size
    add t6, t2, t6
    fsd ft5, 0(t6)                               # store double value to memref of shape (4, 40)
    li t6, 4
    mul t6, t5, t6
    addi t6, t6, 2
    li a3, 40
    mul a3, t3, a3
    add a3, a3, t6
    li t6, 8
    mul a3, a3, t6                               # multiply by element size
    add a3, t2, a3
    fsd ft4, 0(a3)                               # store double value to memref of shape (4, 40)
    li a3, 4
    mul a3, t5, a3
    addi a3, a3, 3
    li t6, 40
    mul t6, t3, t6
    add t6, t6, a3
    li a3, 8
    mul t6, t6, a3                               # multiply by element size
    add t6, t2, t6
    fsd ft3, 0(t6)                               # store double value to memref of shape (4, 40)
    addi t5, t5, 1
    blt t5, t0, scf_body_0_for
scf_body_end_0_for:
    addi t3, t3, 1
    blt t3, t4, scf_body_1_for
scf_body_end_1_for:
    csrrci zero, 1984, 1                         # SSR disable
    ret
