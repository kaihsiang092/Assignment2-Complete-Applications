# Cross-compiler prefix (ensure the RISC-V GCC toolchain is in PATH)
RISCV_PREFIX ?= riscv-none-elf-
EMU ?= ../../../build/rv32emu
CC      = $(RISCV_PREFIX)gcc
CFLAGS  = -march=rv32i -mabi=ilp32 -Os -ffreestanding -Wall -Wextra
LDFLAGS = -T linker.ld -nostartfiles -nostdlib -lgcc

TARGET  = fast_rsqrt.elf
SRCS    = fast_rsqrt.c
OBJS    = $(SRCS:.c=.o)

$(TARGET): $(OBJS) linker.ld
	$(CC) $(CFLAGS) $(OBJS) -o $(TARGET) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

run: $(TARGET)
	$(EMU) ./$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
