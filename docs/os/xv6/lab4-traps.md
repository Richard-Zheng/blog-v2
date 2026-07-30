# Lab 4: Traps

## RISC-V assembly

> Which registers contain arguments to functions? For example, which register holds 13 in main's call to printf?

```
  printf("%d %d\n", f(8)+1, 13);
  2c:	4635                	li	a2,13
  2e:	45b1                	li	a1,12
  30:	00001517          	auipc	a0,0x1
  34:	8d050513          	addi	a0,a0,-1840 # 900 <malloc+0xf6>
  38:	716000ef          	jal	74e <printf>
```

`a2` holds the value 13 in main's call to printf.

The first 8 integer arguments: registers a0 through a7.

```c
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat"

void main(void) {
    unsigned int i = 0x00646c72;
    printf("H%x Wo%s\n", 57616, (char *) &i);
    printf("x=%d y=%d\n", 3);
    exit(0);
}

#pragma GCC diagnostic pop
```

Output:

```
HE110 World
x=3 y=1
```

Explanation:

57616 in hexadecimal is 0xE110

0x00646c72 is stored in memory in little-endian format:

```
low address | 72 6c 64 00 | high address
```

So when it's interpreted as a string, it is "rld\0",

## Backtrace

```
0000000080005590 <printfinit>:
    80005590:   1141                    addi    sp,sp,-16
    80005592:   e406                    sd      ra,8(sp)
    80005594:   e022                    sd      s0,0(sp)
    80005596:   0800                    addi    s0,sp,16
    80005598:   00002597                auipc   a1,0x2
    8000559c:   13858593                addi    a1,a1,312 # 800076d0 <etext+0x6d0>
    800055a0:   0001b517                auipc   a0,0x1b
    800055a4:   59850513                addi    a0,a0,1432 # 80020b38 <pr>
    800055a8:   1e0000ef                jal     80005788 <initlock>
    800055ac:   60a2                    ld      ra,8(sp)
    800055ae:   6402                    ld      s0,0(sp)
    800055b0:   0141                    addi    sp,sp,16
    800055b2:   8082                    ret
```

When a standard function prologue runs, it pushes the Return Address (ra) and the previous Frame Pointer (fp) onto the stack. Immediately after saving these registers, fp is set to equal the current Stack Pointer (sp).

[Lecture note](https://pdos.csail.mit.edu/6.1810/2023/lec/l-riscv.txt) explains it well:

```
Stack
                   .
                   .
      +->          .
      |   +-----------------+   |
      |   | return address  |   |
      |   |   previous fp ------+
      |   | saved registers |
      |   | local variables |
      |   |       ...       | <-+
      |   +-----------------+   |
      |   | return address  |   |
      +------ previous fp   |   |
          | saved registers |   |
          | local variables |   |
      +-> |       ...       |   |
      |   +-----------------+   |
      |   | return address  |   |
      |   |   previous fp ------+
      |   | saved registers |
      |   | local variables |
      |   |       ...       | <-+
      |   +-----------------+   |
      |   | return address  |   |
      +------ previous fp   |   |
          | saved registers |   |
          | local variables |   |
  $fp --> |       ...       |   |
          +-----------------+   |
          | return address  |   |
          |   previous fp ------+
          | saved registers |
  $sp --> | local variables |
          +-----------------+
```