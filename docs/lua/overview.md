# Lua VM Overview

使用一段最简单的 Lua 代码来说明 Lua VM 的工作原理：

```lua
local a=10; local b=20; print(a+b)
```

## 调试技巧

写 gdb 脚本，这样不用每次都手动输入：

```gdb
set pagination off
set debuginfod enabled off

# 下文会用到的 GDB Python 脚本
source lua_gdb.py

break lua_newstate
break preinit_thread
run -e "local a=10; local b=20; print(a+b)"
```

编译命令：

```bash
make clean
make linux MYCFLAGS='-O0 -g3 -fno-inline -fno-omit-frame-pointer -fno-optimize-sibling-calls -DLUA_USE_JUMPTABLE=0'
```

| 参数 | 作用 |
|---|---|
| `-O0` | 关闭 GCC 优化，源码行和执行顺序更直观 |
| `-g3` | 生成完整调试信息，包括宏信息 |
| `-fno-inline` | 尽量避免函数被内联 |
| `-fno-omit-frame-pointer` | 保留帧指针，让 C 调用栈更稳定 |
| `-fno-optimize-sibling-calls` | 避免 GCC 自己把尾调用优化掉 |
| `-DLUA_USE_JUMPTABLE=0` | 让 VM 使用普通 `switch` 分派 |

## 初始化

从 lua.c 的 `main` 开始：

```c
int main (int argc, char **argv) {
  int status, result;
  lua_State *L = luaL_newstate();  /* create state */
  if (L == NULL) {
    l_message(argv[0], "cannot create state: not enough memory");
    return EXIT_FAILURE;
  }
  lua_gc(L, LUA_GCSTOP);  /* stop GC while building state */
  lua_pushcfunction(L, &pmain);  /* to call 'pmain' in protected mode */
  lua_pushinteger(L, argc);  /* 1st argument */
  lua_pushlightuserdata(L, argv); /* 2nd argument */
  status = lua_pcall(L, 2, 1, 0);  /* do the call */
  result = lua_toboolean(L, -1);  /* get result */
  report(L, status);
  lua_close(L);
  return (result && status == LUA_OK) ? EXIT_SUCCESS : EXIT_FAILURE;
}
```

### lua_State

lua_State 是一条 Lua thread 的上下文，包含了执行 Lua 字节码所需的所有状态信息。几乎所有 Lua 函数第一个参数都是它。Lua 官方并不支持真正的多线程，但是可以在单个操作系统线程上协作式地并发执行多个 Lua thread.

`luaL_newstate` 是 `lua_newstate` 的包装函数，用于在没内存分配 state 的时候进行错误处理。(普通的错误处理依赖 state)

```c
LUA_API lua_State *lua_newstate (lua_Alloc f, void *ud, unsigned seed) {
  int i;
  lua_State *L;
  // 在堆上分配一个 global_State + lua_State 的内存块
  // struct global_State 包含一个 LX, 也即 lua_State + 前面 8 bytes 额外空间
  global_State *g = cast(global_State*,
                       (*f)(ud, NULL, LUA_TTHREAD, sizeof(global_State)));
  if (g == NULL) return NULL;
  L = &g->mainth.l;
  L->tt = LUA_VTHREAD;

  /* 设置一大堆字段的默认值... */
  /* 值得注意的有这一行： */
  incnny(L);  /* main thread is always non yieldable */

  if (luaD_rawrunprotected(L, f_luaopen, NULL) != LUA_OK) {
    /* memory allocation error: free partial state */
    close_state(L);
    L = NULL;
  }
  return L;
}
```

`inccny` 在 `lstate.h` 中定义:

```c
/*
** About 'nCcalls':  This count has two parts: the lower 16 bits counts
** the number of recursive invocations in the C stack; the higher
** 16 bits counts the number of non-yieldable calls in the stack.
** (They are together so that we can change and save both with one
** instruction.)
*/

/* true if this thread does not have non-yieldable calls in the stack */
#define yieldable(L)		(((L)->nCcalls & 0xffff0000) == 0)

/* real number of C calls */
#define getCcalls(L)	((L)->nCcalls & 0xffff)

/* Increment the number of non-yieldable calls */
#define incnny(L)	((L)->nCcalls += 0x10000)

/* Decrement the number of non-yieldable calls */
#define decnny(L)	((L)->nCcalls -= 0x10000)

/* Non-yieldable call increment */
#define nyci	(0x10000 | 1)
```

它记录了 C stack 上有多少个 frame, 以及其中有多少个 frame 是 non-yieldable 的。注意：只要这些 frame 里有一个是 non-yieldable 的，那么整个 Lua thread 就是 non-yieldable 的。这里明确指定了主线程（第一个创建的 Lua “线程”）是 non-yieldable 的。

`luaD_rawrunprotected` 保护调用了函数 `f_luaopen`，保护调用是什么呢？就是在出现错误的时候可以统一跳转到一段代码进行错误处理。类似于 try-catch.

```c
TStatus luaD_rawrunprotected (lua_State *L, Pfunc f, void *ud) {
  l_uint32 oldnCcalls = L->nCcalls;
  lua_longjmp lj;
  lj.status = LUA_OK;
  // L->errorJmp 是一个链表，每次保护调用都会把本次上下文放在最前面
  lj.previous = L->errorJmp;  /* chain new error handler */
  L->errorJmp = &lj;
  LUAI_TRY(L, &lj, f, ud);  /* call 'f' catching errors */
  // 结束调用，弹出 lj，恢复上一个上下文
  L->errorJmp = lj.previous;  /* restore old error handler */
  L->nCcalls = oldnCcalls;
  return lj.status;
}
```

这里的关键就在 LUAI_TRY

```c
/* ISO C handling with long jumps */
#define LUAI_THROW(L,c)		longjmp((c)->b, 1)
#define LUAI_TRY(L,c,f,ud)	if (setjmp((c)->b) == 0) ((f)(L, ud))
```

这里用 setjmp 把当前的 C 上下文保存到了 `lua_longjmp lj;` 中，以 RISC-V 为例，包括：

- 被调用者保存寄存器（Call-preserved Registers）`s0` - `s11`
- 栈指针 `sp` 和帧指针 `fp`
- 返回地址 `ra`
- 最后，设置返回值为 0 并返回

这时会正常执行 `Pfunc f` 也即 `f_luaopen`，它会初始化 Lua VM 的各种状态。

```c
/*
** open parts of the state that may cause memory-allocation errors.
*/
static void f_luaopen (lua_State *L, void *ud) {
  global_State *g = G(L);  /* 取得当前 Lua 实例中所有线程共享的全局状态。 */
  UNUSED(ud);  /* 本次初始化不需要回调上下文，仅用于匹配 Pfunc 函数签名。 */
  stack_init(L, L);  /* 为主线程分配值栈，并初始化 base_ci 基础调用帧。 */
  init_registry(L, g);  /* 创建 Registry 表，并注册主线程和全局变量表。 */
  luaS_init(L);  /* 初始化全局字符串表，并创建内存错误消息等固定字符串。 */
  luaT_init(L);  /* 创建并固定 __index、__add 等元方法名称字符串。 */
  luaX_init(L);  /* 创建并固定词法分析器使用的 Lua 保留字字符串。 */
  g->gcstp = 0;  /* 清除 GC 停止标志，从现在开始允许垃圾回收运行。 */
  setnilvalue(&g->nilvalue);  /* 用 nil 标记状态已完整初始化，可进行正常清理。 */
  luai_userstateopen(L);  /* 调用用户可自定义的状态创建完成钩子，默认什么也不做。 */
}
```

如最上面的函数注释所说，此函数有可能触发内存分配错误（如内存不足），发生错误的时候会调用 `luaD_throw`

```c
l_noret luaD_throw (lua_State *L, TStatus errcode) {
  if (L->errorJmp) {  /* thread has an error handler? */
    L->errorJmp->status = errcode;  /* set status */
    LUAI_THROW(L, L->errorJmp);  /* 展开为 longjmp(L->errorJmp->b, 1) */
  } else { /* 省略 */ }
}
```

此时会恢复前面保存的上下文跳转回 `luaD_rawrunprotected`，不同的是 setjmp 返回值不再是 0，而是 1，表示发生了错误，返回 `errcode`。

这些初始化函数里比较值得拿出来说的是 `stack_init(L1, L)`, 它会 call `L`(的全局状态) 中的 `frealloc` 函数为 `L1->stack` 分配堆内存，失败时使用 `L->errorJmp` 进行错误处理，并初始化第一个调用帧 `CallInfo`。

这里有必要说明一点：Lua 虚拟机内部有两套函数运行的上下文：

- 栈，每个 `lua_State` 都有一整块连续内存作为栈，每个函数拥有自己的一段栈区域。栈同时还是寄存器。VM 中 `R[A]` 就是指现在运行的函数的栈里的第 A 个位置（如果算上函数栈开头的 `Closure*` 的话，就是第 A + 1 个）。
- `CallInfo` 链表节点。每次调用函数就向链表头部 push 一个节点，函数返回时 pop 掉。

Closure 对象是什么？是一个函数的实例，它既包含函数在编译期就确定的信息（如字节码，用到的寄存器数量等），也包含函数在运行时的状态（如引用的外部变量）。函数是 Lua 的 first-class 类型，在 Lua 中给变量赋值为一个函数时，实际上操作的也是 Closure 对象。

```c
static void resetCI (lua_State *L) {
  CallInfo *ci = L->ci = &L->base_ci;
  ci->func.p = L->stack.p;
  setnilvalue2s(ci->func.p);  /* 'function' entry for basic 'ci' */
  ci->top.p = ci->func.p + 1 + LUA_MINSTACK;  /* +1 for 'function' entry */
  ci->u.c.k = NULL;
  ci->callstatus = CIST_C;
  L->status = LUA_OK;
  L->errfunc = 0;  /* stack unwind can "throw away" the error function */
}

static void stack_init (lua_State *L1, lua_State *L) {
  int i;
  /* initialize stack array */
  L1->stack.p = luaM_newvector(L, BASIC_STACK_SIZE + EXTRA_STACK, StackValue);
  L1->tbclist.p = L1->stack.p;
  for (i = 0; i < BASIC_STACK_SIZE + EXTRA_STACK; i++)
    setnilvalue2s(L1->stack.p + i);  /* erase new stack */
  L1->stack_last.p = L1->stack.p + BASIC_STACK_SIZE;
  /* initialize first ci */
  resetCI(L1);
  L1->top.p = L1->stack.p + 1;  /* +1 for 'function' entry */
}
```

因此，这里既初始化了栈，也初始化了第一个 `CallInfo` 节点。此节点位于 lua_State 的 `base_ci` 字段中，在 `resetCI` 中被设置为 `L->ci` 链表的头节点。`ci->func.p` 指向函数调用栈区域的第一个位置，为这个函数的 Closure 对象。但是第一个函数在概念上表示 `main` C 函数，没有对应的 Closure, 指向的栈里面为 nil。

```c
ci->func.p = L->stack.p;
setnilvalue2s(ci->func.p);
```

我们可以写一个脚本验证：

```python
import gdb

def _field(expr):
  return gdb.parse_and_eval(expr)

def _tvalue_text(tv):
  tag = int(tv["tt_"])
  kind = tag & 0x0f
  variant = (tag >> 4) & 3
  value = tv["value_"]
  try:
    if kind == 0:
      return "nil"
    if kind == 1:
      return "true" if variant == 1 else "false"
    if kind == 2:
      return f"lightuserdata({value['p']})"
    if kind == 3:
      return str(value["n"] if variant == 1 else value["i"])
    if kind == 4:
      ts = value["gc"].cast(gdb.lookup_type("TString").pointer())
      return '"' + ts["contents"].string(errors="replace") + '"'
    if kind == 5:
      return f"table({value['gc']})"
    if kind == 6:
      names = {0: "LuaClosure", 1: "CFunction", 2: "CClosure"}
      ptr = value["f"] if variant == 1 else value["gc"]
      return f"{names.get(variant, 'function')}({ptr})"
    if kind == 7:
      return f"userdata({value['gc']})"
    if kind == 8:
      return f"thread({value['gc']})"
  except gdb.error:
    pass
  return f"tag=0x{tag:x}"

class LuaFrames(gdb.Command):
  def __init__(self):
    super().__init__("lua-frames", gdb.COMMAND_DATA)

  def invoke(self, arg, from_tty):
    try:
      ci = _field("L->ci")
      depth = 0
      while int(ci) != 0 and depth < 32:
        frame = ci.dereference()
        status = int(frame["callstatus"])
        flavor = "C" if status & (1 << 15) else "Lua"
        funcslot = frame["func"]["p"].dereference()["val"]
        gdb.write(
          f"  #{depth} {flavor:<3} ci={ci} func={_tvalue_text(funcslot)}\n"
        )
        ci = frame["previous"]
        depth += 1
    except gdb.error as exc:
      gdb.write(f"lua-frames: cannot inspect frames ({exc})\n")

LuaFrames()
```

在 `stack_init` 的最后一行打断点：

```gdb
break lstate.c:173
```


使用 `lua-frames` 查看当前的 `CallInfo` 链表。

```
Breakpoint 1, stack_init (L1=0x5555555b1598, L=0x5555555b1598) at lstate.c:173
173	  L1->top.p = L1->stack.p + 1;  /* +1 for 'function' entry */
(gdb) lua-frames
  #0 C   ci=0x5555555b15f8 func=nil
```

### pmain

有了 `lua_State` 后，就可以按照标准的 Lua 调用 C 函数那样调用 `pmain` 了：

p 意为 protected，也即有错误处理，上文已提及。

```c
lua_pushcfunction(L, &pmain);  /* to call 'pmain' in protected mode */
lua_pushinteger(L, argc);  /* 1st argument */
lua_pushlightuserdata(L, argv); /* 2nd argument */
```

push C function, push 的是什么呢？就是 Closure!

```c
#define lua_pushcfunction(L,f)	lua_pushcclosure(L, (f), 0)
```

如此一来，`pmain` 就被包装成了一个 `CClosure` 对象，放在栈里的第一个位置。

向 `lua_gdb.py` 添加一个新的命令 `lua-stack`，可以查看栈内容：

```py
class LuaStack(gdb.Command):
  def __init__(self):
    super().__init__("lua-stack", gdb.COMMAND_DATA)

  def invoke(self, arg, from_tty):
    try:
      start = _field("L->stack.p")
      size = int(_field("L->stack_last.p - L->stack.p"))
      top = _field("L->top.p")
      gdb.write(f"Stack: start={start}, top={top}, size={size}\n")
      nil_start = None

      def write_nil_run(end):
        nonlocal nil_start
        if nil_start is None:
          return
        top_marker = ""
        if start + nil_start <= top < start + end:
          top_marker = f" <-- top at R[{int(top - start)}]"
        count = end - nil_start
        if count == 1:
          gdb.write(f"  R[{nil_start:<2}] nil{top_marker}\n")
        else:
          gdb.write(
            f"  R[{nil_start}..{end - 1}] nil ({count} slots){top_marker}\n"
          )
        nil_start = None

      for index in range(size):
        slot = (start + index).dereference()["val"]
        text = _tvalue_text(slot)
        if text == "nil":
          if nil_start is None:
            nil_start = index
          continue
        write_nil_run(index)
        marker = " <-- top" if start + index == top else ""
        gdb.write(f"  R[{index:<2}] {text}{marker}\n")
      write_nil_run(size)
    except gdb.error as exc:
      gdb.write(f"lua-stack: no active Lua VM frame ({exc})\n")

LuaStack()
```

在 push 前打断点，可以单步调试看到完整过程：

```gdb
break lua.c:785
```

输出

```
785	  lua_pushcfunction(L, &pmain);  /* to call 'pmain' in protected mode */
(gdb) lua-stack
Stack: start=0x5555555b1670, top=0x5555555b1680, size=40
  R[0..39] nil (40 slots) <-- top at R[1]


(gdb) n
786	  lua_pushinteger(L, argc);  /* 1st argument */
(gdb) lua-stack
Stack: start=0x5555555b1670, top=0x5555555b1690, size=40
  R[0 ] nil
  R[1 ] CFunction(0x55555555cdaa <pmain>)
  R[2..39] nil (38 slots) <-- top at R[2]


(gdb) n
787	  lua_pushlightuserdata(L, argv); /* 2nd argument */
(gdb) lua-stack
Stack: start=0x5555555b1670, top=0x5555555b16a0, size=40
  R[0 ] nil
  R[1 ] CFunction(0x55555555cdaa <pmain>)
  R[2 ] 3
  R[3..39] nil (37 slots) <-- top at R[3]


(gdb) n
788	  status = lua_pcall(L, 2, 1, 0);  /* do the call */
(gdb) lua-stack
Stack: start=0x5555555b1670, top=0x5555555b16b0, size=40
  R[0 ] nil
  R[1 ] CFunction(0x55555555cdaa <pmain>)
  R[2 ] 3
  R[3 ] lightuserdata(0x7fffffffe2e8)
  R[4..39] nil (36 slots) <-- top at R[4]
```

接下来

```c
status = lua_pcall(L, 2, 1, 0);  /* do the call */
```

`2` 是参数数量，`1` 是返回值数量，`0` 是错误处理函数（这里传了 `0`，表示没有）。

`lua_pcall` 会展开为 `lua_pcallk`，k 应该代表 continuation，有些 Lua 调用的 C 函数可以主动 yield, 实现异步。这里用不上，先不用管了。重点在于：

```c
static void f_call (lua_State *L, void *ud) {
  struct CallS *c = cast(struct CallS *, ud);
  luaD_callnoyield(L, c->func, c->nresults);
}

int lua_pcallk (lua_State *L, int nargs, int nresults, int errfunc,
                lua_KContext ctx, lua_KFunction k /* 这两个参数本次调用都为 NULL */) {
  struct CallS c;
  ...
  c.func = L->top.p - (nargs+1);  /* function to be called */
  if (k == NULL || !yieldable(L)) {  /* no continuation or no yieldable? */
    c.nresults = nresults;  /* do a 'conventional' protected call */
    status = luaD_pcall(L, f_call, &c, savestack(L, c.func), func);
  }
  ...
}
```

`c.func` 是栈顶往下数 `nargs + 1` 个位置，注意栈顶是第一个空闲位置，因此这就是我们之前 push 的 `pmain` C 函数指针。

后面的过程有点复杂，而且深究的意义不大，直接看走到 `pmain` 的时候的调用栈吧：

```
Breakpoint 1, pmain (L=0x5555555b1598) at lua.c:731
731	static int pmain (lua_State *L) {
#0  pmain (L=0x5555555b1598) at lua.c:731
#1  0x00005555555649aa in precallC (L=0x5555555b1598, func=0x5555555b1680, status=2, f=0x55555555cdaa <pmain>) at ldo.c:657
#2  0x0000555555564d19 in luaD_precall (L=0x5555555b1598, func=0x5555555b1680, nresults=1) at ldo.c:726
#3  0x0000555555564f90 in ccall (L=0x5555555b1598, func=0x5555555b1680, nResults=1, inc=65537) at ldo.c:766
#4  0x0000555555565038 in luaD_callnoyield (L=0x5555555b1598, func=0x5555555b1680, nResults=1) at ldo.c:786
#5  0x000055555555fdca in f_call (L=0x5555555b1598, ud=0x7fffffffe160) at lapi.c:1071
#6  0x0000555555563862 in luaD_rawrunprotected (L=0x5555555b1598, f=0x55555555fd95 <f_call>, ud=0x7fffffffe160) at ldo.c:166
#7  0x0000555555565918 in luaD_pcall (L=0x5555555b1598, func=0x55555555fd95 <f_call>, u=0x7fffffffe160, old_top=16, ef=0) at ldo.c:1090
#8  0x000055555555fea2 in lua_pcallk (L=0x5555555b1598, nargs=2, nresults=1, errfunc=0, ctx=0, k=0x0) at lapi.c:1097
#9  0x000055555555d0de in main (argc=3, argv=0x7fffffffe2e8) at lua.c:788
```

`luaD_pcall` 就是保存了当前 `lua_State` 的一些信息到 C 栈上，然后调用 `luaD_rawrunprotected` 并处理异常返回时的收尾工作。`f_call` 和 `luaD_callnoyield` 只是包装函数。

后面的函数就是 C 侧调用 C / Lua 区别对待的逻辑：`ccall` 负责维护调用前后的 `L->nCcalls` 计数，如果执行的是 Lua 函数的话还会额外执行 `luaV_execute`，也就是 VM 的字节码解释器。`luaD_precall` 根据 `func` 的类型（C closure 或 Lua closure）来决定调用哪一条路径。C closure 会走到 `precallC`，准备好 CallInfo 后会直接进行真正的调用（precall 有点名不副实）。Lua closure 会走到 `prepCallInfo`，准备好 CallInfo 后会返回到 `ccall`。

`pmain` 一开始会真正处理 `argc` 和 `argv`，都是一些比较 trivial 的 C 代码。

```c
/*
** Main body of stand-alone interpreter (to be called in protected mode).
** Reads the options and handles them all.
*/
static int pmain (lua_State *L) {
  int argc = (int)lua_tointeger(L, 1);
  char **argv = (char **)lua_touserdata(L, 2);
  /* 省略参数处理 */

  luai_openlibs(L);  /* open standard libraries */
  createargtable(L, argv, argc, script);  /* create table 'arg' */
  lua_gc(L, LUA_GCRESTART);  /* start GC... */
  lua_gc(L, LUA_GCGEN);  /* ...in generational mode */
  if (handle_luainit(L) != LUA_OK)  /* run LUA_INIT */
    return 0;  /* error running LUA_INIT */
  if (!runargs(L, argv, optlim))  /* execute arguments -e, -l, and -W */
    return 0;  /* something failed */

  /* 省略一些别的功能 */
}
```

`luai_openlibs` 是打开所有标准库的宏。

```c
/* The default is to open all standard libraries */
#define luai_openlibs(L)  luaL_openselectedlibs(L, ~0, 0)
```

`luaL_openselectedlibs` 位于 `linit.c`：

```c
/*
** Standard Libraries. (Must be listed in the same ORDER of their
** respective constants LUA_<libname>K.)
*/
static const luaL_Reg stdlibs[] = {
  {LUA_GNAME, luaopen_base},
  {LUA_LOADLIBNAME, luaopen_package},
  {LUA_COLIBNAME, luaopen_coroutine},
  {LUA_DBLIBNAME, luaopen_debug},
  {LUA_IOLIBNAME, luaopen_io},
  {LUA_MATHLIBNAME, luaopen_math},
  {LUA_OSLIBNAME, luaopen_os},
  {LUA_STRLIBNAME, luaopen_string},
  {LUA_TABLIBNAME, luaopen_table},
  {LUA_UTF8LIBNAME, luaopen_utf8},
  {NULL, NULL}
};

/*
** require and preload selected standard libraries
*/
LUALIB_API void luaL_openselectedlibs (lua_State *L, int load, int preload) {
  int mask;
  const luaL_Reg *lib;
  /* LUA_REGISTRYINDEX 是一个很大的负数，这里用于表示 Lua 内部的 registry table */
  /* 实际是取到 G(L)->l_registry["_PRELOAD"] 这个 table 放在栈顶，专门用于存放预加载的库 */
  luaL_getsubtable(L, LUA_REGISTRYINDEX, LUA_PRELOAD_TABLE);
  for (lib = stdlibs, mask = 1; lib->name != NULL; lib++, mask <<= 1) {
    if (load & mask) {  /* selected? */
      luaL_requiref(L, lib->name, lib->func, 1);  /* require library */
      lua_pop(L, 1);  /* remove result from the stack */
    }
    else if (preload & mask) {  /* selected? */
      lua_pushcfunction(L, lib->func);
      lua_setfield(L, -2, lib->name);  /* add library to PRELOAD table */
    }
  }
  lua_assert((mask >> 1) == LUA_UTF8LIBK);
  lua_pop(L, 1);  /* remove PRELOAD table */
}
```
