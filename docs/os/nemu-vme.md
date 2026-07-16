# Nanos-lite Context Switch

:::warning

内容包含剧透，如果你还没有完成 [ICS PA](https://nju-projectn.github.io/ics-pa-gitbook/) 一周目，请不要阅读本篇文章。

:::

## 核心概念

在实现时钟中断前，上下文切换完全由 CTE 的 `yield()` 发起，在 riscv 中为

```c
void yield() {
#ifdef __riscv_e
  asm volatile("li a5, -1; ecall");
#else
  asm volatile("li a7, -1; ecall");
#endif
}
```

`ecall` 后会跳转到 `mtvec` 寄存器保存的异常处理入口地址. 其值由

```c
bool cte_init(Context*(*handler)(Event, Context*)) {
  // initialize exception entry
  asm volatile("csrw mtvec, %0" : : "r"(__am_asm_trap));

  // register event handler
  user_handler = handler;

  return true;
}
```

初始化。可知 `__am_asm_trap` 是 CTE 的入口函数。位于 `trap.S` 中。伪代码如下：

```c
void __am_asm_trap() {
  判断当前是否为用户态，如是则把 sp 切换到内核栈, 并保存用户栈指针
  保存当前上下文到内核栈
  调用 __am_irq_handle() 来处理异常/中断, 接受返回的 Context 指针
  从返回的 context 中恢复上下文
}
```

保存和恢复页表寄存器分别是在 `__am_irq_handle()` 的开头和结尾完成的。

如果要创建新的进程，只需要创建对应的栈空间并在其顶部“伪造”一个 Context 结构体即可。后续调度到此新进程时，就好像是从中断返回一样，直接恢复这个上下文。

```c
Context *ucontext(AddrSpace *as, Area kstack, void *entry, uintptr_t user_stack_top) {
  Context *c = (Context *)(kstack.end - sizeof(Context));
  c->pdir = as->ptr;
  c->mepc = (uintptr_t) entry;
  c->gpr[2] = (uintptr_t) user_stack_top; // sp
  c->GPRx = (uintptr_t) user_stack_top; // sp, set by _start in navy
  c->mstatus = 0x1800 | (1u << 7); // MPP=11b, MPIE=1
  c->next_privilege = PRIV_USER;
  return c;
}
```

## 思考题

> 最后, 为了让这一地址空间生效, 我们还需要将它落实到MMU中. 具体地, 我们希望在CTE恢复进程上下文的时候来切换地址空间. 为此, 我们需要将进程的地址空间描述符指针`as->ptr`加入到上下文中, 框架代码已经实现了这一功能(见`abstract-machine/am/include/arch/$ISA-nemu.h`), 在x86中这一成员为`cr3`, 而在mips32/riscv32中则为`pdir`. 你还需要
>
> - 修改`ucontext()`的实现, 在创建的用户进程上下文中设置地址空间描述符指针
> - 在`__am_irq_handle()`的开头调用`__am_get_cur_as()` (在`abstract-machine/am/src/$ISA/nemu/vme.c`中定义), 来将当前的地址空间描述符指针保存到上下文中
> - 在`__am_irq_handle()`返回前调用`__am_switch()` (在`abstract-machine/am/src/$ISA/nemu/vme.c`中定义)来切换地址空间, 将被调度进程的地址空间落实到MMU中

pdir根页表地址被存储到了上下文中，但是上下文被保存到用户栈上了。这就成了先有鸡还是先有蛋的问题：要恢复页表寄存器就得读取上下文，但是要读取上下文又得有页表。

这个问题的答案在讲义中似乎完全没有提到，相关的只有一道蓝框思考题：

> 可以在用户栈里面创建用户进程上下文吗?
>
> `ucontext()`的行为是在内核栈`kstack`中创建用户进程上下文. 我们是否可以对`ucontext()`的行为进行修改, 让它在用户栈上创建用户进程上下文? 为什么?

不行，就跟上面说的原因一样。

最直接的解法应该是想办法把上下文保存到内核栈中。思考半天，翻了下后面的内容，写到了我的疑惑。

> 我们之前把如下问题作为最难的思考题留给大家思考:
>
> ```text
> 为什么目前不支持并发执行多个用户进程?
> ```
>
> 现在我们就来揭晓问题的答案: 这是因为用户栈的访问造成的.
>
> ...
>
> 不过作为一个地址空间描述符指针, 其值在创建用户进程上下文的时候就已经确定, 并在每次进入CTE时, 其值都是一致的. 因此,  我们完全不必在中断异常到来时保存它, 只要在创建用户进程上下文时将其存放在PCB中, 需要切换虚拟地址空间时,  就直接从B的PCB中读出地址空间描述符指针即可. 当然, PCB是操作系统的概念, AM并不了解, 因此还需要VME提供一个新的API `switch_addrspace()`: 操作系统在`schedule()`中选择进程B之后, 先通过`switch_addrspace()`切换到B的虚拟地址空间, 再返回到CTE并恢复B的上下文.
>
> 打破循环依赖的方法
>
> 如上文所述, 将地址空间描述符指针存放在PCB中, 并在VME中添加一个新API `switch_addrspace()`, 从正确性来考虑, 这一方案是否可行?

不太行，因为schedule里sp还指向上一个进程的用户栈呢，切换地址空间后会覆盖/误读下一个进程的栈。

参考讲义实现了上下文保存到内核栈，核心代码都在汇编 `__am_asm_trap` 中

> 而为了实现上述功能, 我们又需要解决如下问题:
>
> - 如何识别进入CTE之前处于用户态还是内核态? - `pp` (Previous Privilege)
> - CTE的代码如何知道内核栈在什么位置? - `ksp` (Kernel Stack Pointer)
> - 如何知道将要返回的是用户态还是内核态? - `np` (Next Privilege)
> - CTE的代码如何知道用户栈在什么位置? - `usp` (User Stack Pointer)

这里的 previous 指的是 trap 之前的状态，next 就是 trap 返回之后的状态。如果pp是内核态，那么sp已经在内核地址空间，不需要切换到内核栈。如果np是内核态，那么也不需要从内核栈恢复到用户栈。用户态的情况就都需要保存和恢复。

考虑到调用`__am_irq_handle`后返回的栈顶可能跟之前的不属于同一个进程，所以np、usp都得和特定进程上下文绑定，每个上下文各一个。这里有意思的是ksp是不需要和特定上下文绑定的，可以是一个全局唯一变量。这是因为`__am_irq_handle`会负责返回一个正确的内核栈ksp，无论是内核线程的栈还是用户线程的内核栈。保存这个ksp直到下一次调用`__am_asm_trap`就足够了。pp也是同理。

对于重入问题，我们只关心在进入`__am_irq_handle`后可以正常自陷和返回。观察伪代码在进入`__am_irq_handle`前已经切换到内核栈保存了上下文，那么此时再自陷就要当成内核态处理，不切换sp直接保存上下文。所以进入`__am_irq_handle`前把pp置为内核态即可。

合并pp和ksp，usp和sp后的伪代码：

```c
void __am_asm_trap() {
  c->sp = $sp;
  if (ksp != 0) {
    swap($sp, ksp); // user stack -> kernel stack
  }
  c->np = ksp;      // kernel: 0, user: non-zero

  ksp = 0;          // reentrancy: now in kernel mode

  push context;
  $sp = __am_irq_handle($sp);
  pop context;

  if (c->np == USER) {
    ksp = $sp;
  }
  $sp = c->sp;

  return_from_trap();
}
```

和讲义伪代码稍微有些出入，主要是为了汇编好写。riscv32中把mscratch寄存器当作ksp，然后

```
c->sp = $sp;
c->np = ksp;      // kernel: 0, user: non-zero
```

这两条都得在栈上申请了Context的空间以后做，`c->np = ksp;`直接当成多保存一个寄存器就可以了。可是`c->sp = $sp;`怎么办呢？这里得分两种情况：

1.  pp为内核态，直接保存`$sp + sizeof(Context)`即可
2. pp为用户态，此时`$sp`已经切换为内核栈，所以要把原先的用户栈临时保存到ksp中，这就是为什么用`swap($sp, ksp);` 而不是`$sp = ksp;`, riscv32中已经有原子交换CSR寄存器和其他寄存器的指令。这里保存`$mscratch`即可

汇编对应伪代码如下：

```c
void __am_asm_trap() {
  swap(sp, mscratch);
  if (sp == 0) {        // CSR reg can't be used in branch
    swap(sp, mscratch); // swap back
  }
    
  $sp -= sizeof(Context);
  // put registers in stack...

  c->np = $mscratch;
  if ($mscratch != 0) {
    c->sp = $mscratch;  // pp = USER, save original sp
  } else {
    c->sp = $sp + sizeof(Context);
  }
  $mscratch = 0;

  $sp = __am_irq_handle($sp);

  if (c->np != 0) {     // return to user mode, save kernel sp
    $mscratch = $sp + sizeof(Context);
  }
  // retrieve registers from stack...

  $sp = c->sp;
  mret();
}
```

此外，Context结构体需要增加next_privilege项，并更改trap汇编中的对应占位大小。kcontext中需要设置正确的c->sp和c->next_privilege，ucontext中目前只需要设置c->next_privilege，因为在_start中会把GPRx复制到sp（是不是应该改一下，不绕这个弯路呢？不过猜测可能是为了和native之类的兼容的设计……）。