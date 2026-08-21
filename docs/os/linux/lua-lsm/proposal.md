# Lua-LSM shared dict 项目申请书

郑语秋 yuqiu283@gmail.com

## 需求分析

本次项目分为两部分，下面分别进行分析。

### 1. 支持 RISC-V 架构

项目目前使用了 Lua 解释器的代码，在内核态下解释执行 Lua 代码。内核 Lua 解释器的错误处理依赖架构提供的非局部跳转能力，RISC-V 尚未提供该支持，导致整个机制在该架构上无法启用。

在 [`lib/Kconfig`](https://github.com/openanolis/lua-lsm-kernel/blob/495d06eaaa491b0dcacb5393822cd06ebd42968c/lib/Kconfig#L640-L649) 下可见：

```text
config HAS_LUA
	bool
	depends on ARCH_HAS_SETJMP
	default y

config LUA
	tristate
	depends on HAS_LUA
	select ARCH_SETJMP
	default n
```

而 [arch/riscv/Kconfig](https://github.com/openanolis/lua-lsm-kernel/blob/495d06eaaa491b0dcacb5393822cd06ebd42968c/arch/riscv/Kconfig) 中缺乏 `ARCH_HAS_SETJMP` 的定义，导致 RISC-V 架构下无法启用 Lua-LSM。[arch/x86/Kconfig](https://github.com/openanolis/lua-lsm-kernel/blob/495d06eaaa491b0dcacb5393822cd06ebd42968c/arch/x86/Kconfig#L107) 中则有该定义，故可以正常编译。

Lua 中依赖 `setjmp`/`longjmp` 的代码在 [`include/linux/luaconf.h`](https://github.com/openanolis/lua-lsm-kernel/blob/495d06eaaa491b0dcacb5393822cd06ebd42968c/include/linux/luaconf.h#L620-L623).

```c
#define LUAI_THROW(L,c)	longjmp(& ((c)->b), 1)
#define LUAI_TRY(L,c,a)	if (setjmp(& ((c)->b)) == 0) { a }
```

也即调用 `LUAI_TRY` 时会把当前 call-preserved 的寄存器保存到 `c->b` 中，并执行语句 `a`。如果在执行 `a` 的过程中调用了 `LUAI_THROW`，则会把寄存器恢复到 `c->b` 中的值，并返回到 `LUAI_TRY` 这行语句之后。

### 2. 共享字典用户态接口

目前共享字典是在同一个 Lua module 不同上下文（不同进程）之间共享同一个 KV Store 的机制。用法如下：

```lua
local denylist = shared["denylist"]
denylist:set("/etc/shadow", true)
denylist:incr("open_count", 1)
return {
  name = "shared_dict_demo",
  author = "zhengyuqiu",
  description = "Shared dictionary example",
  license = "MIT",
  version = 1,
  file_open = function(file, cred)
    denylist:incr("open_count", 1)

    local path = file:path()
    if path and denylist:get(path) then
        return false
    end
    return true
  end
}
```

`shared` 全局变量在 `security/lua/lsm.c` 的 [`module_init`](https://github.com/openanolis/lua-lsm-kernel/blob/495d06eaaa491b0dcacb5393822cd06ebd42968c/security/lua/lsm.c#L688-L695) 中被初始化为一个 table 并加入 `__index` 和 `__newindex` 元方法。由 `lua_shared_index` 处理对 `shared` 的访问。

目前有两个痛点：

1. 共享字典完全封闭在内核态，用户态既观测不到策略的运行时状态，也无法向策略下发数据，策略要接受外部输入只能重新加载一份新代码。
2. 字典不接受字符串，难以承载路径、进程名这类最常需要交互的名单数据，策略逻辑与策略数据因此被迫耦合在一起。

由 [kvcache 节点的处理函数](https://github.com/openanolis/lua-lsm-kernel/blob/495d06eaaa491b0dcacb5393822cd06ebd42968c/security/lua/kvcache.c#L276-L286) 可以看出，kvcache value 目前只支持以下三种类型：

- `LUA_TBOOLEAN`
- `LUA_TNUMBER`
- `LUA_TLIGHTUSERDATA`

因此需要为其加入字符串集合或类似物支持。

## 实现方案

### 1. RISC-V 架构支持

这个需求较简单，只需加入 `setjmp` 和 `longjmp` 的 RISC-V 实现即可。

具体来说，可以在 `arch/riscv/lib/` 下新建 `setjmp.S`，实现 `setjmp` 和 `longjmp`。

再在 `arch/riscv/lib/Makefile` 中加入 `setjmp.o` 作为 `CONFIG_ARCH_SETJMP` 选项的库依赖。

最后在 `arch/riscv/Kconfig` 中加入 `select ARCH_HAS_SETJMP`，以便启用 `HAS_LUA`。

实现完成后，可在 `arch/riscv/kernel/tests/` 下加入一个测试用例，验证 `setjmp`/`longjmp` 的正确性。

### 2. 共享字典用户态接口

