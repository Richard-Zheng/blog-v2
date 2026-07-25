# Lab 3: Page Table

## print_pgtbl

output:

```text
print_pgtbl starting
va 0x0 pte 0x21FC885B pa 0x87F22000 perm 0x5B
va 0x1000 pte 0x21FC7C5B pa 0x87F1F000 perm 0x5B
va 0x2000 pte 0x21FC7817 pa 0x87F1E000 perm 0x17
va 0x3000 pte 0x21FC7407 pa 0x87F1D000 perm 0x7
va 0x4000 pte 0x21FC70D7 pa 0x87F1C000 perm 0xD7
va 0x5000 pte 0x0 pa 0x0 perm 0x0
va 0x6000 pte 0x0 pa 0x0 perm 0x0
va 0x7000 pte 0x0 pa 0x0 perm 0x0
va 0x8000 pte 0x0 pa 0x0 perm 0x0
va 0x9000 pte 0x0 pa 0x0 perm 0x0
va 0x3FFFFF6000 pte 0x0 pa 0x0 perm 0x0
va 0x3FFFFF7000 pte 0x0 pa 0x0 perm 0x0
va 0x3FFFFF8000 pte 0x0 pa 0x0 perm 0x0
va 0x3FFFFF9000 pte 0x0 pa 0x0 perm 0x0
va 0x3FFFFFA000 pte 0x0 pa 0x0 perm 0x0
va 0x3FFFFFB000 pte 0x0 pa 0x0 perm 0x0
va 0x3FFFFFC000 pte 0x0 pa 0x0 perm 0x0
va 0x3FFFFFD000 pte 0x0 pa 0x0 perm 0x0
va 0x3FFFFFE000 pte 0x21FD08C7 pa 0x87F42000 perm 0xC7
va 0x3FFFFFF000 pte 0x2000184B pa 0x80006000 perm 0x4B
print_pgtbl: OK
```

Use readelf to see the program headers:

```text
$ readelf -l user/_pgtbltest

Elf file type is EXEC (Executable file)
Entry point 0x518
There are 4 program headers, starting at offset 64

Program Headers:
  Type           Offset             VirtAddr           PhysAddr
                 FileSiz            MemSiz              Flags  Align
  RISCV_ATTRIBUT 0x000000000000ab84 0x0000000000000000 0x0000000000000000
                 0x0000000000000074 0x0000000000000000  R      0x1
  LOAD           0x0000000000001000 0x0000000000000000 0x0000000000000000
                 0x0000000000001109 0x0000000000001109  R E    0x1000
  LOAD           0x0000000000003000 0x0000000000002000 0x0000000000002000
                 0x0000000000000010 0x0000000000000030  RW     0x1000
  GNU_STACK      0x0000000000000000 0x0000000000000000 0x0000000000000000
                 0x0000000000000000 0x0000000000000000  RW     0x10

 Section to Segment mapping:
  Segment Sections...
   00     .riscv.attributes
   01     .text .rodata
   02     .data .bss
   03
```

As its indicated in the ELF program headers, `0x0` to `0x1109` is the `.text .rodata` segment, corresponds to

```text
va 0x0 pte 0x21FC885B pa 0x87F22000 perm 0x5B
va 0x1000 pte 0x21FC7C5B pa 0x87F1F000 perm 0x5B
```

After that, `0x2000` to `0x2000 + 0x10` is the `.data .bss` segment, corresponds to

```text
va 0x2000 pte 0x21FC7817 pa 0x87F1E000 perm 0x17
```

Then it's the guard page and user stack page. Allocated at `exec.c`

```c
#define USERSTACK 1     // user stack pages

// Allocate some pages at the next page boundary.
// Make the first inaccessible as a stack guard.
// Use the rest as the user stack.
sz = PGROUNDUP(sz);
uint64 sz1;
if((sz1 = uvmalloc(pagetable, sz, sz + (USERSTACK+1)*PGSIZE, PTE_W)) == 0)
goto bad;
sz = sz1;
uvmclear(pagetable, sz-(USERSTACK+1)*PGSIZE);
sp = sz;
stackbase = sp - USERSTACK*PGSIZE;
```

It first use `uvmalloc` to allocate 2 pages, then use `uvmclear` to clear the first page, which is the guard page. What `uvmclear` does is:

```c
*pte &= ~PTE_U;
```

That removes the `PTE_U` flag.

The next page is the user stack page.

```text
va 0x3000 pte 0x21FC7407 pa 0x87F1D000 perm 0x7   // guard page
va 0x4000 pte 0x21FC70D7 pa 0x87F1C000 perm 0xD7  // user stack page
```

The least significant byte of the PTE is the permission bits.

```
bit 7  D  Dirty
bit 6  A  Accessed
bit 5  G  Global
bit 4  U  User
bit 3  X  Execute
bit 2  W  Write
bit 1  R  Read
bit 0  V  Valid
```

D and A are set by the hardware, G is not used in xv6, the rest are set by the OS.

So for the last 4 bit:

```
0x7 = 0111 = W, R, V (rw-)
0xB = 1011 = X, R, V (r-x)
```

Near the top of user address space:

- `0x3fffffe000` is the trapframe page
- `0x3ffffff000` is the trampoline page

```
va 0x3FFFFFE000 pte 0x21FD08C7 pa 0x87F42000 perm 0xC7 // trapframe page
va 0x3FFFFFF000 pte 0x2000184B pa 0x80006000 perm 0x4B // trampoline page
```

They are set up in `proc.c`:

```c
// Create a user page table for a given process, with no user memory,
// but with trampoline and trapframe pages.
pagetable_t
proc_pagetable(struct proc *p)
{
  pagetable_t pagetable;

  // An empty page table.
  pagetable = uvmcreate();
  if(pagetable == 0)
    return 0;

  // map the trampoline code (for system call return)
  // at the highest user virtual address.
  // only the supervisor uses it, on the way
  // to/from user space, so not PTE_U.
  if(mappages(pagetable, TRAMPOLINE, PGSIZE,
              (uint64)trampoline, PTE_R | PTE_X) < 0){
    uvmfree(pagetable, 0);
    return 0;
  }

  // map the trapframe page just below the trampoline page, for
  // trampoline.S.
  if(mappages(pagetable, TRAPFRAME, PGSIZE,
              (uint64)(p->trapframe), PTE_R | PTE_W) < 0){
    uvmunmap(pagetable, TRAMPOLINE, 1, 0);
    uvmfree(pagetable, 0);
    return 0;
  }

  return pagetable;
}
```
