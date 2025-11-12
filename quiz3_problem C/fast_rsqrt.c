#include <stdint.h>
#include <stddef.h>

// Count leading zeros of a 32-bit value (returns 32 if x is 0)
static inline uint32_t clz32(uint32_t x) {
    if (x == 0) return 32u;
    uint32_t n = 0;
    if ((x >> 16) == 0) { n += 16; x <<= 16; }
    if ((x >> 24) == 0) { n += 8;  x <<= 8;  }
    if ((x >> 28) == 0) { n += 4;  x <<= 4;  }
    if ((x >> 30) == 0) { n += 2;  x <<= 2;  }
    if ((x >> 31) == 0) { n += 1; }
    return n;
}

// Lookup table (32 entries) for initial 1/sqrt estimate at powers of two (Q16 fixed-point).
// Values ≈ ⌊2^16 / sqrt(2^e)⌋ for e = 0..31. (Using 65536 for e=0 to represent 1.0 exactly)
static const uint32_t rsqrt_table[32] = {
    65536, 46341, 32768, 23170, 16384, 11585, 8192, 5793,
    4096,  2896,  2048,  1448, 1024,   724,  512,  362,
    256,   181,   128,    90,   64,    45,   32,   23,
     16,    11,     8,     6,    4,     3,    2,    1
};

// Perform one Newton-Raphson iteration for y = 1/√x in Q16 fixed-point.
// Uses: y_new = y * (3 - x * y^2) / 2, all in integer arithmetic.
static inline uint32_t q16_newton_step(uint32_t y, uint32_t x) {
    // 64-bit intermediates to avoid overflow on 32-bit multiplies
    uint64_t y2   = (uint64_t)y * y;                       // y^2 (Q32 format)
    uint64_t term = (3ull << 32) - (uint64_t)x * y2;       // (3 * 2^32) - x*y^2
    uint64_t prod = (uint64_t)y * term;                    // y * term
    return (uint32_t)(prod >> 33);  // divide by 2 * 2^32 (i.e., >>33) to get Q16 result
}

// Compute fast reciprocal square root in Q16 fixed-point (returns ⌊2^16/√x⌋).
uint32_t fast_rsqrt(uint32_t x) {
    if (x == 0u) {
        // For x=0, return max value (saturate) as reciprocal sqrt tends to infinity
        return 0xFFFFFFFFu;
    }
    // 1) Find exponent bucket: exp = floor(log2(x))
    uint32_t exp = 31u - clz32(x);

    // 2) Get lookup table base and next values for this exponent
    uint32_t y_base = rsqrt_table[exp];
    uint32_t y_next = (exp < 31u) ? rsqrt_table[exp + 1u] : 1u;  // for exp=31, next is set to 1 (minimum)

    // 3) Linear interpolation between y_base and y_next.
    //    Compute fractional position of x within [2^exp, 2^(exp+1)):
    uint32_t one_exp = (exp < 31u) ? (1u << exp) : 0u;       // 2^exp (avoid shifting 1<<31 in 32-bit)
    uint64_t diff   = (uint64_t)x - one_exp;                 // difference (64-bit to avoid overflow before shift)
    uint64_t frac64 = (diff << 16);                          // fraction * 2^16 (Q16) before division
    uint32_t frac   = (uint32_t)(frac64 >> exp);             // (x - 2^exp) / 2^exp * 2^16 (Q16 fractional part)
    // Interpolate: y ≈ y_base - (y_base - y_next) * frac / 2^16
    uint32_t delta = y_base - y_next;
    uint64_t interp = (uint64_t)delta * frac;
    uint32_t y = y_base - (uint32_t)(interp >> 16);

    // 4) Apply two Newton-Raphson iterations to refine the result
    y = q16_newton_step(y, x);
    y = q16_newton_step(y, x);
    return y;
}

// Compute 3D distance = √(x^2 + y^2 + z^2) using fast_rsqrt (integer approximation).
uint32_t dist3(int32_t x, int32_t y, int32_t z) {
    // Use 64-bit for sum of squares to avoid overflow
    uint64_t sum_sq = (uint64_t)x * x + (uint64_t)y * y + (uint64_t)z * z;
    if (sum_sq > 0xFFFFFFFFull) {
        sum_sq = 0xFFFFFFFFull; // saturate if overflow beyond 32-bit range
    }
    uint32_t inv_sqrt = fast_rsqrt((uint32_t)sum_sq);  // Q16 reciprocal sqrt of sum
    // Distance ≈ (inv_sqrt * sum) >> 16  (because inv_sqrt ≈ 2^16/√(sum))
    uint64_t dist = inv_sqrt * (uint64_t)(uint32_t)sum_sq;
    return (uint32_t)(dist >> 16);
}

// Minimal syscall interface for writing to stdout (fd=1) on rv32emu
static inline long write_syscall(int fd, const void *buf, size_t count) {
    register int a0 __asm__("a0") = fd;
    register const void *a1 __asm__("a1") = buf;
    register size_t a2 __asm__("a2") = count;
    register int syscall_nr __asm__("a7") = 64;  // sys_write call number (64) in RISC-V
    __asm__ volatile ("ecall"
                      : "+r"(a0)                // return value in a0
                      : "r"(syscall_nr), "r"(a1), "r"(a2)
                      : "memory");
    return a0;  // return number of bytes written (or error code)
}

// Helper: output a null-terminated string to stdout
static void print_str(const char *s) {
    // Calculate length (stop at first null)
    size_t len = 0;
    while (s[len] != '\0') len++;
    if (len) {
        write_syscall(1, s, len);
    }
}

// Helper: output an unsigned 32-bit integer in decimal
static void print_uint(uint32_t val) {
    char buf[11]; // up to 10 digits + null terminator (not printing null)
    int pos = 0;
    if (val == 0) {
        buf[pos++] = '0';
    } else {
        // Convert number to decimal string (in reverse order)
        while (val != 0 && pos < 10) {
            buf[pos++] = '0' + (val % 10);
            val /= 10;
        }
    }
    // Output the digits in correct order
    for (int i = pos - 1; i >= 0; --i) {
        write_syscall(1, &buf[i], 1);
    }
}

// Example main function demonstrating usage of fast_rsqrt and dist3
int main(void) {
    // Print a header
    print_str("===== Fast Reciprocal Square Root Demo =====\n");

    // Demonstrate fast_rsqrt on some values
    uint32_t values[] = {1, 5, 16, 1000000};
    for (unsigned i = 0; i < sizeof(values)/sizeof(values[0]); ++i) {
        uint32_t n = values[i];
        print_str("fast_rsqrt(");
        print_uint(n);
        print_str(") = ");
        print_uint(fast_rsqrt(n));
        print_str("\n");
    }

    // Demonstrate computing 3D distance using fast_rsqrt
    int32_t ax = 1, ay = 2, az = 3;
    print_str("Distance of (");
    print_uint(ax); print_str(", ");
    print_uint(ay); print_str(", ");
    print_uint(az); print_str(") = ");
    print_uint(dist3(ax, ay, az));
    print_str("\n");

    return 0;
}
void *memcpy(void *dest, const void *src, size_t n) {
    char *d = dest;
    const char *s = src;
    while (n--) {
        *d++ = *s++;
    }
    return dest;
}

// Bare-metal program entry point: set up stack and call main(), then exit.
__attribute__((naked)) void _start(void) {
    __asm__ volatile (
        "la sp, _stack_top\n"    // set stack pointer to top of stack area
        "call main\n"            // call main()
        // upon return, use exit syscall (93) to terminate program
         "li a0, 0\n"         // exit code 0
        "li a7, 93\n"
        "ecall\n"
    );
}
