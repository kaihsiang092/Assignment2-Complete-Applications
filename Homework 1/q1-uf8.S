.data
str1:     .string ": produces value "
str2:     .string " but encodes back to "
str3:     .string ": value "
str4:     .string " <= previous_value "
str5:     .string "All tests passed.\n"
str6:     .string " tests failed.\n"
.text
.globl main
main:
   jal ra, test
   beq a0, x0, fail
   la a0, str5 # print str5 
   li a7, 4
   ecall
   li a7, 93         # ecall: exit
   li a0, 0 # exit code is 0, 
   ecall
fail:
    la a0, str6 # print str6
    li a7, 4
    ecall
    li a7, 93         # ecall: exit
    li a0, 1 # exit code is 1
    ecall
# clz(a0 = x) -> a0 = result
clz:
    li      t0, 32        # n = 32
    li      t1, 16        # c = 16
clz_loop:
    srl     t2, a0, t1    # y = x >> c
    beq     t2, x0, clz_skip    # if (y == 0) skip
    sub     t0, t0, t1    # n -= c
    add     a0, t2, x0    # x = y
clz_skip:  
    srli    t1, t1, 1     # c >>= 1
    bne     t1, x0, clz_loop     # while(c)
    sub     a0, t0, a0    # return n - x
    jr      ra
    
# a0 = fl (uint8)
# return a0 = value (uint32)
uf8_decode:
    andi    t0, a0, 0x0F       # t0 = mantissa,mantissa = fl & 0x0f;
    srli    t1, a0, 4          # t1 = exponent,exponent = fl >> 4;
    li      t2, 15
    sub     t2, t2, t1 # 15 - exponent
    li      t3, 0x7FFF
    srl     t3, t3, t2 # 0x7FFF >> (15 - exponent)
    slli    t3, t3, 4 # t3 = offset
    sll     t2, t0, t1 # mantissa << exponent
    add     a0, t2, t3 # a0 = (mantissa << exponent) + offset
    jr      ra   # jump to ra

uf8_encode:
    # if (value < 16) return value;
    addi sp, sp, -4
    sw ra, 0(sp)
    add t6, a0, x0 # t6 = value 
    addi t0, x0, 16
    blt a0, t0, return1
    jal ra, clz
     # msb = 31 - lz
     addi    t1, x0, 31  
     sub     t1, t1, a0  # t1 = msb = 31 - lz;
     addi    t2, x0, 0  # t2 = exponent = 0;
     addi    t3, x0, 0  # t3 = overflow = 0;
     addi    t4, t4, 5
     blt     t1, t4, exact_exponent 
     addi    t2, t1, -4  # exponent = msb - 4;
     addi    t4, x0, 15
     bge     t4, t2, Calculate_overflow # if (exponent <= 15) Calculate overflow for estimated exponent
     li      t2, 15
Calculate_overflow:
    addi t4, x0, 0 # e = 0
for1:
    slli t5, t3, 1
    addi t3, t5, 16
    addi t4, t4, 1
    blt  t4, t2, for1
while1:
    blez t2, exact_exponent # if (exponent <= 0) break;
    bge t6, t3, exact_exponent # if (value >= overflow) break;
    addi t5, t3, -16 # (overflow - 16)
    srli t3, t5, 1 # t3 = overflow = (overflow - 16) >> 1;
    addi t2, t2, -1
    j while1
exact_exponent:
    addi    t5, x0, 15
while2:
    bge t2, t5, return0 
    slli t4, t3, 1 # (overflow << 1)
    addi t4, t4, 16 # t4 = next_overflow = (overflow << 1) + 16;
    blt t6, t4, return0 # if (value < next_overflow) break;
    add t3, t4, x0
    addi t2, t2, 1
    j while2
return0:
   sub t1, t6, t3 # (value - overflow)
   srl t1, t1, t2 # t1 = mantissa = (value - overflow) >> exponent;
   slli t0, t2, 4 # (exponent << 4)
   or a0, t0, t1 # return (exponent << 4) | mantissa;
return1:
    lw ra, 0(sp)
    addi sp, sp, 4
    jr ra   # jump to ra
test:
    addi sp, sp, -4
    sw ra, 0(sp)
    addi s0, x0, -1 # s0 = previous_value
    addi s1, x0,  1 # s1 = passed
    addi s2, x0,  0 # fl = 0
    addi s3, x0,256 # fl = 256
for2:
    add a0, s2, x0
    jal ra, uf8_decode
    add s4, a0, x0 # s4 = value
    add a0, s4, x0
    jal ra, uf8_encode
    add s5, a0, x0 # s5 = fl2
if1:
    beq s2, s5, if2
    mv a0, s2   
    li a7, 34    
    ecall
    li a0, 1
    la a1, str1 # print str1
    la a2, 17
    li a7, 64
    ecall
    mv a0, s4 # print value 
    li a7, 1
    ecall
    la a0, str2 # print str2
    li a7, 4
    ecall
    mv a0, s5    
    li a7, 34     
    ecall
    li s1, 0 # passed = false
if2:
    blt s0, s4, afterif
    mv a0, s2       # print f1
    li a7, 34     
    ecall
    la a0, str3 # print str3
    li a7, 4
    ecall
    mv a0, s4 # print value 
    li a7, 1
    ecall
    la a0, str4 # print str4
    li a7, 4
    ecall
    mv a0, s0   # prepare to print previous_value
    li a7, 34    
    ecall
    li s1, 0  # passed = false  
afterif:
    mv s0, s4 # previous_value = value;
    addi s2, s2, 1
    blt s2, s3, for2
    mv a0, s1 # return passed
    lw ra, 0(sp)
    addi sp, sp, 4
    jr ra   # jump to ra
