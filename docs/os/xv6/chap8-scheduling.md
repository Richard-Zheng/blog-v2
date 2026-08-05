# Chapter 8: Scheduling

:::info

Refer to the [xv6 book](https://pdos.csail.mit.edu/6.1810/2026/xv6/book-riscv-rev5.pdf) chapter 8 for more details.

:::

## Why `intena` is a field in `struct CPU` instead of `struct proc`?

`intena` records if the interrupts were enabled before first `push_off()`.

From source code:

```c
// Switch to scheduler.  Must hold only p->lock
// and have changed proc->state. Saves and restores
// intena because intena is a property of this
// kernel thread, not this CPU. It should
// be proc->intena and proc->noff, but that would
// break in the few places where a lock is held but
// there's no process.
void
sched(void)
{
  int intena;
  struct proc *p = myproc();

  if(!holding(&p->lock))
    panic("sched p->lock");
  if(mycpu()->noff != 1)
    panic("sched locks");
  if(p->state == RUNNING)
    panic("sched RUNNING");
  if(intr_get())
    panic("sched interruptible");

  intena = mycpu()->intena;
  swtch(&p->context, &mycpu()->context);
  mycpu()->intena = intena;
}
```

`intena` is saved to the stack (local variable) right before `swtch(&p->context, &mycpu()->context)` transfers control to `scheduler()` process, and the next process does `mycpu()->intena = intena` to recover it's `intena` from it's stack. So `intena` should be a property of the current process, not the CPU.

So why it's in `struct CPU`? It's because we need to acquire/release spinlock(which does `push_off()` / `pop_off()`) when there's no `c->proc`, specifically in `scheduler()`.

```c
// Per-CPU process scheduler.
// Each CPU calls scheduler() after setting itself up.
// Scheduler never returns.  It loops, doing:
//  - choose a process to run.
//  - swtch to start running that process.
//  - eventually that process transfers control
//    via swtch back to the scheduler.
void
scheduler(void)
{
  struct proc *p;
  struct cpu *c = mycpu();

  c->proc = 0;
  for(;;){
    // The most recent process to run may have had interrupts
    // turned off; enable them to avoid a deadlock if all
    // processes are waiting. Then turn them back off
    // to avoid a possible race between an interrupt
    // and wfi.
    intr_on();
    intr_off();

    int nproc = 0;
    for(p = proc; p < &proc[NPROC]; p++) {
      acquire(&p->lock);
      if(p->state != UNUSED) {
        nproc++;
      }
#ifdef LAB_LOCK
      if(p->pincpu && p->pincpu != c) {
        release(&p->lock);
        continue;
      }
#endif
      if(p->state == RUNNABLE) {
        // Switch to chosen process.  It is the process's job
        // to release its lock and then reacquire it
        // before jumping back to us.
        p->state = RUNNING;
        c->proc = p;
        
        swtch(&c->context, &p->context);

        // Process is done running for now.
        // It should have changed its p->state before coming back.
        c->proc = 0;
      }
      release(&p->lock);
    }
    if(nproc <= 2) {   // only init and sh exist
      // nothing to run; stop running on this core until an interrupt.
      intr_on();
#ifndef LAB_FS
      asm volatile("wfi");
#endif
    }
  }
}
```

At the start, `c->proc` is set to `0`, so we cannot use `c->proc->intena` and `c->proc->noff`. Actually we are in kernel scheduler process, and it doesn't have a `proc` structure.

What about `noff`? The logic is the same, but `noff` is always 1 in `sched()`:

```c
void
sched(void)
{
  int intena;
  struct proc *p = myproc();

  if(!holding(&p->lock))
    panic("sched p->lock");
  if(mycpu()->noff != 1)
    panic("sched locks");
  ...
}
```

So `noff` doesn't need to be saved/restored.
