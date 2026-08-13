# LSM 初探

LSM（Linux Security Module）是 Linux 内核的安全模块框架。它允许开发者在内核中实现各种安全策略，如访问控制、权限管理等，而无需修改内核的核心代码。

## 编译支持 LSM 的内核

启用并重新编译：

```bash
scripts/config --enable SECURITY
scripts/config --enable SECURITYFS
make olddefconfig
make -j"$(nproc)"
```

然后重新启动新内核。

进入 BusyBox 后创建挂载点并挂载：

```sh
mkdir -p /sys/kernel/security
mount -t securityfs securityfs /sys/kernel/security
```

`/sys/kernel/security` 不是普通 sysfs 目录，它是 **securityfs 的挂载点**。

检查：

```sh
mount
ls -la /sys/kernel/security
cat /sys/kernel/security/lsm
```

`/sys/kernel/security/lsm` 应该列出当前实际启用的 LSM，例如：

```text
capability,landlock,lockdown,yama,...
```

也可以确认内核是否认识 securityfs：

```sh
cat /proc/filesystems | grep securityfs
```

如果显示：

```text
nodev   securityfs
```

表示已经编译支持，只是之前没有挂载。如果没有任何输出，则通常是 `CONFIG_SECURITYFS` 没启用。

建议在 `/init` 中加入：

```sh
mkdir -p /sys/kernel/security
mount -t securityfs securityfs /sys/kernel/security
```

完整关系是：

```text
/sys
  └── kernel/
       ├── debug       ← debugfs 挂载点
       ├── tracing     ← tracefs 挂载点
       └── security    ← securityfs 挂载点
```

## 调试 LSM / SELinux

SELinux 是一个 LSM 模块，它在内核中注册了许多钩子函数，用于在关键操作时进行安全检查。

在 GDB 中设置：

```gdb
break security_file_open
break selinux_file_open
continue
```

然后在虚拟机中执行：

```sh
cat /proc/version
```

命中 `security_file_open()` 后：

```gdb
bt
```

可得

```
#0  selinux_file_open (file=0xffff888004460c00) at security/selinux/hooks.c:4259
#1  0xffffffff817e5a00 in security_file_open (file=0xffff888004460c00) at security/security.c:2739
#2  0xffffffff8157916a in do_dentry_open (f=f@entry=0xffff888004460c00, open=open@entry=0x0) at fs/open.c:924
#3  0xffffffff8157b56b in vfs_open (path=path@entry=0xffffc90000013d80, file=file@entry=0xffff888004460c00) at fs/open.c:1079
#4  0xffffffff815993ce in do_open (nd=0xffffc90000013d80, file=0xffff888004460c00, op=0xffffc90000013eb4) at fs/namei.c:4699
#5  path_openat (nd=nd@entry=0xffffc90000013d80, op=op@entry=0xffffc90000013eb4, flags=257) at fs/namei.c:4862
#6  0xffffffff8159a432 in do_file_open (dfd=dfd@entry=-100, pathname=pathname@entry=0xffff888003c72b40, op=op@entry=0xffffc90000013eb4)
    at fs/namei.c:4891
#7  0xffffffff8157ba56 in do_sys_openat2 (dfd=-100, filename=<optimized out>, how=how@entry=0xffffc90000013ef8) at fs/open.c:1364
#8  0xffffffff8157bedc in do_sys_open (dfd=<optimized out>, filename=<optimized out>, flags=<optimized out>, mode=<optimized out>)
    at fs/open.c:1370
#9  __do_sys_openat (dfd=<optimized out>, filename=<optimized out>, flags=<optimized out>, mode=<optimized out>) at fs/open.c:1386
#10 __se_sys_openat (dfd=<optimized out>, filename=<optimized out>, flags=<optimized out>, mode=<optimized out>) at fs/open.c:1381
#11 __x64_sys_openat (regs=<optimized out>) at fs/open.c:1381
#12 0xffffffff823ffb40 in do_syscall_x64 (regs=0xffffc90000013f58, nr=<optimized out>) at arch/x86/entry/syscall_64.c:63
#13 do_syscall_64 (regs=0xffffc90000013f58, nr=<optimized out>) at arch/x86/entry/syscall_64.c:94
#14 0xffffffff81000130 in entry_SYSCALL_64 () at arch/x86/entry/entry_64.S:121
#15 0x0000000000000007 in ?? ()
#16 0x000000001e6a4710 in ?? ()
#17 0x0000000000000001 in ?? ()
#18 0x0000000000000006 in ?? ()
#19 0x00007ffe1cefcf10 in ?? ()
#20 0x000000001e6a5550 in ?? ()
```

```gdb
p $lx_current()->pid
p $lx_current()->comm
p file->f_path.dentry->d_name.name
p/x file->f_flags
```

继续：

```gdb
continue
```

应该命中：

```c
selinux_file_open(struct file *file)
```

然后查看：

```gdb
bt
p file->f_path.dentry->d_name.name
p file->f_mode
```

由于这个函数在符号表中的类型是小写 `t`：

```text
ffffffff817f49a0 t selinux_file_open
```

表示它是局部文本符号，即源码中的 `static` 函数。GDB 仍然可以正常对它设置断点。

## 为什么 `CONFIG_LSM` 里有 Landlock/TOMOYO，却没启用

你的配置中有：

```text
CONFIG_LSM="landlock,...,selinux,...,tomoyo,..."
```

但这只是：

> 如果这些 LSM 已经被编译进内核，它们应按什么顺序启用。

它不会自动设置：

```text
CONFIG_SECURITY_LANDLOCK=y
CONFIG_SECURITY_TOMOYO=y
```

关系类似：

```text
CONFIG_SECURITY_TOMOYO=y
    决定是否编译 TOMOYO

CONFIG_LSM="...,tomoyo,..."
    决定已编译的 TOMOYO 是否进入启用序列及其顺序
```

必须同时满足，LSM 才会实际运行。

## 如果想启用 TOMOYO 和 Landlock

执行：

```bash
scripts/config --enable SECURITY_TOMOYO
scripts/config --enable SECURITY_LANDLOCK
make olddefconfig
make -j"$(nproc)"
```

确认：

```bash
grep -E 'CONFIG_SECURITY_(TOMOYO|LANDLOCK)=' .config
nm vmlinux | grep -E 'tomoyo_file_open|hook_file_open'
```

然后务必用新生成的 `bzImage` 重新启动 QEMU，并让 GDB 加载同一次构建的 `vmlinux`。

启动后：

```sh
cat /sys/kernel/security/lsm
```

预计会包含：

```text
capability,landlock,selinux,tomoyo
```

不过 TOMOYO 可能还有默认启用及策略加载相关配置；即使 hook 已注册，没有加载有效策略时，它通常不会拒绝普通操作。

## 确保 QEMU 与 GDB 使用同一份构建

这是内核调试非常重要的一点：

```text
QEMU：arch/x86/boot/bzImage
GDB：vmlinux
```

二者必须来自同一次 `make`。

如果 QEMU 运行旧 `bzImage`，而 GDB 加载新 `vmlinux`，函数地址和源码行可能不匹配，表现为：

- 断点不命中
- 命中错误函数
- 调用栈异常
- 单步跳到奇怪的位置

重新构建和启动后，建议完全退出旧 GDB，再运行：

```bash
gdb ./vmlinux
```

## 一个更清晰的首次实验

暂时不必启用更多 LSM，先用现有 SELinux 观察：

```gdb
break security_file_open
commands
  silent
  printf "\nsecurity_file_open: pid=%d comm=%s file=%s\n", \
      $lx_current()->pid, $lx_current()->comm, \
      file->f_path.dentry->d_name.name
  bt 8
  continue
end
```

如果希望停下来进入 SELinux：

```gdb
break selinux_file_open
continue
```

但还有一点需要检查：SELinux “出现在 LSM 列表”不一定表示正在 enforcing。虚拟机中运行：

```sh
cat /sys/fs/selinux/enforce 2>/dev/null
cat /proc/cmdline
```

如果 selinuxfs 尚未挂载：

```sh
mkdir -p /sys/fs/selinux
mount -t selinuxfs selinuxfs /sys/fs/selinux
cat /sys/fs/selinux/enforce
```

结果：

```text
1 → enforcing
0 → permissive
```

即使没有加载策略或者处于 permissive，`selinux_file_open()` 仍可能被调用；区别是它是否最终拒绝访问。

你现在最直接的断点组合是：

```gdb
break security_file_open
break selinux_file_open
```

这已经足够观察完整的“VFS → LSM 框架 → 具体 LSM”分发过程。