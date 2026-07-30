# Lab 5: CoW fork

Pitfalls I've fallen into:

1. When adding reference count to `kalloc.c`, the `kfree` function should behave like this:

```
void kfree(void *pa) {
    acquire(&kmem.lock);
    --refcount[pa];
    if refcount[pa] == 0 {
        memset to 1
        add the page to freelist
    }
    release(&kmem.lock);
}
```

but I'm afraid that `memset` is time-consuming, not good for doing while holding the lock. So I did it like this:

```
void kfree(void *pa) {
    int is_released = 0;
    acquire(&kmem.lock);
    --refcount[pa];
    if refcount[pa] == 0 {
        is_released = 1;
        add the page to freelist
    }
    release(&kmem.lock);
    if is_released {
        memset to 1
    }
}
```

which is wrong because the freed page need to hold the pointer to the next free page, but `memset` will overwrite it. So the correct way is to `memset` before adding to the freelist.

2. The original `uvmcopy` function is:

```c
int
uvmcopy(pagetable_t old, pagetable_t new, uint64 sz)
{
  pte_t *pte;
  uint64 pa, i;
  uint flags;
  char *mem;

  for(i = 0; i < sz; i += PGSIZE){
    if((pte = walk(old, i, 0)) == 0)
      continue;   // page table entry hasn't been allocated
    if((*pte & PTE_V) == 0)
      continue;   // physical page hasn't been allocated
    pa = PTE2PA(*pte);
    flags = PTE_FLAGS(*pte);
    if((mem = kalloc()) == 0)
      goto err;
    memmove(mem, (char*)pa, PGSIZE);
    if(mappages(new, i, PGSIZE, (uint64)mem, flags) != 0){
      kfree(mem);
      goto err;
    }
  }
  return 0;

 err:
  uvmunmap(new, 0, i / PGSIZE, 1);
  return -1;
}
```

I modified `*pte` to `*pte & ~PTE_W | PTE_COW`, thinking the flag will pass to the new page table's page, but no, I mistakenly added the CoW logic under `flags = PTE_FLAGS(*pte)`, so the new page will still use the original flag.
