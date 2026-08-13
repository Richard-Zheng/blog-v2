# 开发环境搭建

首先获取 Linux 内核源码，参考 [KernelBuild - KernelNewbies](https://kernelnewbies.org/KernelBuild) 在 kernel.org 上下载最新的 Linux 内核源码，我下载起来挺快的。

解压到一个目录下，我解压到 `~/Download/linux/linux-7.1.8`.

## 编译 Busybox

然后获取 busybox 源码

```bash
cd ~/Download/linux
git clone https://github.com/mirror/busybox.git
```

编译

```bash
cd busybox
make defconfig
make menuconfig
```

报错找不到 ncurses 库，修复测试函数解决：

```diff
diff --git a/scripts/kconfig/lxdialog/check-lxdialog.sh b/scripts/kconfig/lxdialog/check-lxdialog.sh
index 5075ebf2d..fcb96f8ed 100755
--- a/scripts/kconfig/lxdialog/check-lxdialog.sh
+++ b/scripts/kconfig/lxdialog/check-lxdialog.sh
@@ -47,7 +47,7 @@ trap "rm -f $tmp" 0 1 2 3 15
 check() {
         $cc -x c - -o $tmp 2>/dev/null <<'EOF'
 #include CURSES_LOC
-main() {}
+int main(void) { return 0; }
 EOF
        if [ $? != 0 ]; then
            echo " *** Unable to find the ncurses libraries or the"       1>&2
```

然后找到 `Settings -> Build Options -> Build static binary (no shared libs)`，按空格键选中。我这边发现 networking/tc.c 编译不过，所以我还需要把 `Networking Utilities -> tc` 取消选中。

```bash
# Compile using all available CPU cores
make -j$(nproc)

# Install into a target directory named 'rootfs'
make CONFIG_PREFIX=../rootfs install
cd ../rootfs

# Create a directory tree
mkdir -p dev proc sys etc root

# Write a simple init script
cat > init << "EOF"
#!/bin/sh

# Mount essential pseudo-filesystems
mount -t proc none /proc
mount -t sysfs none /sys
mount -t devtmpfs none /dev

# Clear the screen and welcome the user
clear
echo "=== Minimal Kernel Dev Environment Ready ==="

# Launch an interactive shell (manages job control)
exec setsid cttyhack /bin/sh
EOF
# Make the init script executable
chmod +x init

# Pack into a CPIO archive
find . | cpio -o -H newc | gzip > ../rootfs.cpio.gz
```

## 编译 Linux 内核

```bash
cd ~/Download/linux/linux-7.1.8
# Configure the kernel
make x86_64_defconfig
make menuconfig
# Compile
make -j"$(nproc)"
```

运行

```bash
qemu-system-x86_64 \
  -kernel arch/x86/boot/bzImage \
  -initrd ../rootfs.cpio.gz \
  -append "console=ttyS0 nokaslr" \
  -nographic
```

生成 `compile_commands.json`，用于 clangd 智能提示：

```bash
python3 ./scripts/clang-tools/gen_compile_commands.py
```

加上调试信息：

直接用 scripts/config：

```bash
scripts/config --enable DEBUG_INFO
scripts/config --disable DEBUG_INFO_NONE
scripts/config --enable DEBUG_INFO_DWARF5
scripts/config --enable GDB_SCRIPTS
scripts/config --enable FRAME_POINTER
scripts/config --disable RANDOMIZE_BASE

make olddefconfig
make -j"$(nproc)"
```

其中：

- `DEBUG_INFO`：生成源码行、类型和局部变量信息
- `DWARF5`：调试信息格式
- `GDB_SCRIPTS`：生成 Linux 专用 GDB 辅助命令
- `FRAME_POINTER`：让调用栈更容易还原
- 关闭 `RANDOMIZE_BASE`：避免 KASLR 改变内核运行地址

终端一启动 QEMU：

```bash
qemu-system-x86_64 \
    -kernel arch/x86/boot/bzImage \
    -initrd /tmp/my-initramfs.cpio.gz \
    -append "console=ttyS0 rdinit=/init nokaslr" \
    -nographic \
    -m 512M \
    -s \
    -S
```

新增的两个参数：

- `-s`：在 TCP 端口 `1234` 开启 GDB server
- `-S`：CPU 上电后暂停，等待 GDB

此时终端看起来没有反应是正常的。

## 连接 GDB

打开第二个终端，进入内核源码目录：

```bash
cd /home/frain/Downloads/linux/linux-7.1.8
gdb vmlinux
```

进入 GDB 后：

```gdb
target remote :1234
break start_kernel
continue
```

命中后可以运行：

```gdb
bt
list
info args
info locals
next
step
```

常用含义：

```text
bt          当前调用栈
frame 2     切换到第 2 层栈帧
up/down     切换到调用者/被调用者
list        显示附近源码
info args   当前函数参数
info locals 当前局部变量
p expr      查看 C 表达式
next        单步，但不进入函数
step        单步并进入函数
continue    继续运行到下个断点
```

不要从 QEMU 上电后的第一条汇编开始单步。x86 启动过程很长，先在 `start_kernel()` 断住更合适。

## 第一个实验：观察内核启动

设置这些断点：

```gdb
break start_kernel
break mm_core_init
break sched_init
break kernel_init
continue
```

每次断住后：

```gdb
bt
info args
list
```

推荐用 `continue` 在几个关键点之间跳转，不要试图逐行走完整个 `start_kernel()`。

你会观察到类似：

```text
x86 架构启动
  → start_kernel()
  → 内存、调度器等子系统初始化
  → rest_init()
  → 创建 kernel_init 内核线程
  → kernel_init()
  → 执行 /init
```

值得注意的是，`start_kernel()` 早期的栈可能比较短，因为它本身接近 C 语言初始化入口。

## 第二个实验：观察页分配

先让系统正常启动到 shell，然后在 GDB 中按 `Ctrl-C` 暂停虚拟 CPU，设置断点：

```gdb
break __alloc_pages_noprof
continue
```

命中后：

```gdb
bt
info args
p order
p/x gfp
p $lx_current().comm
p $lx_current().pid
```

但页分配是高频操作，断点会命中得非常频繁。更好的做法是使用条件断点，例如只观察 BusyBox：

```gdb
break __alloc_pages_noprof if $lx_current().pid > 1
```

或者先在 shell 中确定一个实验程序的 PID，再使用：

```gdb
break __alloc_pages_noprof if $lx_current().pid == 42
```

函数名可能因配置和编译优化有所变化，查找候选符号：

```gdb
info functions alloc_pages
```

一次典型的调用栈可能显示：

```text
__alloc_pages_noprof
  ← alloc_pages_mpol_noprof
  ← folio_alloc_noprof
  ← 某个缺页或文件缓存路径
```

这时从栈底向上读，比从 `mm/page_alloc.c` 猜谁会调用它容易很多。

## 第三个实验：观察用户缺页

这是从 xv6 迁移过来最合适的实验。

在 GDB 中搜索函数：

```gdb
info functions handle_mm_fault
```

设置断点：

```gdb
break handle_mm_fault
continue
```

然后在 BusyBox shell 中运行一个会创建新进程的命令，例如：

```sh
/bin/echo hello
```

命中后：

```gdb
bt
p/x address
p flags
p $lx_current().comm
p $lx_current().pid
```

你会看到概念上的路径：

```text
用户访问尚未映射的地址
  → x86 page fault 异常入口
  → do_user_addr_fault()
  → lock_mm_and_find_vma()
  → handle_mm_fault()
  → 匿名页或文件页处理
  → alloc_pages()
  → 安装 PTE
```

这是非常好的阅读切入点。先保存实际调用栈，再逐层打开相应源码。

## 第四个实验：观察 `exec`

断在：

```gdb
break do_execveat_common
continue
```

然后在 shell 中执行：

```sh
/bin/uname -a
```

断住后查看：

```gdb
bt
p $lx_current().comm
p $lx_current().pid
p filename->name
```

继续执行时，可以再断：

```gdb
break load_elf_binary
continue
```

由于 BusyBox 是 ELF，你会看到：

```text
shell 发起 execve
  → do_execveat_common()
  → 读取 BusyBox 文件头
  → search_binary_handler()
  → load_elf_binary()
  → 建立新的用户地址空间
  → 返回用户态执行 BusyBox
```

而启动 `/init` 时，因为它是 `#!` 脚本，还会经过：

```gdb
break load_script
```

于是能真实观察：

```text
/init
  → load_script()
  → 找到 #!/bin/sh
  → 再执行 /bin/sh
  → /bin/sh 解析到 BusyBox
  → load_elf_binary()
```

这正好验证我们上一轮讨论的启动链。

## Linux 提供的 GDB 命令

启用 `CONFIG_GDB_SCRIPTS` 后，在源码根目录运行：

```bash
make scripts_gdb
```

GDB 启动时可能拒绝自动加载脚本：

```text
warning: File .../vmlinux-gdb.py auto-loading has been declined
```

在 `~/.gdbinit` 中加入：

```gdb
add-auto-load-safe-path /home/frain/Downloads/linux/linux-7.1.8
```

重新打开 GDB 后，可使用：

```gdb
lx-ps
lx-dmesg
lx-symbols
p $lx_current().pid
p $lx_current().comm
```

例如：

```gdb
(gdb) lx-ps
(gdb) p $lx_current().comm
$1 = "sh"
(gdb) lx-dmesg
```

参考 linux-7.1.8/Documentation/translations/zh_CN/dev-tools/gdb-kernel-debugging.rst

## 调用栈不完整怎么办

常见原因有：

- 调试信息未启用
- 编译优化导致函数内联
- 某些汇编入口没有普通栈帧
- CPU 正处于中断或异常入口
- `FRAME_POINTER` 没启用
- 断点设在过于底层或极高频的函数
- GDB 使用的 `vmlinux` 和 QEMU 中的 `bzImage` 不是同一次构建

确保：

```text
CONFIG_DEBUG_INFO_DWARF5=y
CONFIG_FRAME_POINTER=y
nokaslr
```

并且每次重新构建后，退出并重新启动 GDB/QEMU。

还要注意编译优化会使 GDB 出现：

```text
<optimized out>
```

这是正常的。不要一开始就把整个内核编译成 `-O0`，因为内核代码和构建假设通常依赖正常优化。函数参数看不到时，可以：

- 在函数入口断住
- 查看寄存器
- 查看调用者
- 使用反汇编辅助理解

```gdb
disassemble /m function_name
info registers
```

## 推荐的学习循环

每次只研究一个问题：

```text
提出问题
  → 写一个最小用户程序触发它
  → 在通用内核函数设置断点
  → bt 获取真实调用栈
  → 记录关键参数和 current
  → 只阅读栈上的函数
  → 用下一层断点验证理解
```

最推荐依次完成四个实验：

1. `/init` 如何变成 BusyBox
2. `execve()` 如何加载 ELF
3. 用户 page fault 如何分配匿名页
4. `read()` 如何从 fd 进入 VFS

不要先追 `kmalloc()` 或所有页分配，因为它们太高频，容易反复断住。**`exec + page fault` 是从 xv6 进入 Linux 动态调试的最佳起点。**