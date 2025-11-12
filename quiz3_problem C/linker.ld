/* Linker script for bare-metal RV32I (ILP32) */
ENTRY(_start)
SECTIONS {
    /* Start code at address 0x00000000 */
    . = 0x00000000;
    /* Code and read-only data section */
    .text : {
        *(.text*)           /* all .text sections */
        *(.rodata*)         /* all .rodata (constants, strings) */
    }
    /* Initialized data section */
    .data : {
        *(.data*)
    }
    /* Uninitialized data (BSS) section */
    .bss (NOLOAD) : {
        *(.bss*)
        *(COMMON)
    }
    /* End of program sections */
    _end = .;
    /* Define a stack region of 16KB after end of program */
    _stack_top = _end + 0x4000;
}
