# Lua-LSM Risc-V

## 编译 Busybox

先安装 toolchains 参考 [xv6 实验手册](https://pdos.csail.mit.edu/6.1810/2026/tools.html)。

```bash
sudo pacman -S riscv64-linux-gnu-binutils riscv64-linux-gnu-gcc riscv64-linux-gnu-gdb qemu-emulators-full bc git base-devel
```

clone 下来 busybox 代码库，开始编译

```bash
git clone --depth=1 https://github.com/mirror/busybox.git
cd busybox
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- defconfig
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- menuconfig
```

`Settings -> Build Options -> Build static binary (no shared libs)`，按空格键选中。还有取消编译 tc（我的系统上编译 tc 会报错，[mailing list 里也有类似的报告](https://lists.busybox.net/pipermail/busybox/2026-August/092410.html)）。

```bash
# Compile using all available CPU cores
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- -j"$(nproc)"

# Install into a target directory named 'rootfs'
rm -rf ../rootfs
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- CONFIG_PREFIX=../rootfs install

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

echo "=== Minimal RISC-V Kernel Dev Environment Ready ==="

# Launch an interactive shell (manages job control)
exec setsid cttyhack /bin/sh
EOF
# Make the init script executable
chmod +x init

# Pack into a CPIO archive
find . | cpio -o -H newc | gzip > /tmp/lua-lsm-initramfs.cpio.gz
```

## 编译

你当前环境已经装好了交叉编译器和 QEMU。建议使用独立输出目录，避免在源码树中产生大量构建文件。

### 1. 生成配置

在仓库根目录执行：

```bash
mkdir -p /tmp/lua-lsm-riscv-build

make O=/tmp/lua-lsm-riscv-build \
    ARCH=riscv \
    CROSS_COMPILE=riscv64-linux-gnu- \
    defconfig
```

启用 Lua-LSM：

```bash
scripts/config \
    --file /tmp/lua-lsm-riscv-build/.config \
    --enable SECURITY \
    --enable SECURITYFS \
    --enable LUA \
    --enable SECURITY_LUA_LSM
```

确保 Lua LSM 在启动顺序中。先检查当前配置：

```bash
grep '^CONFIG_LSM=' /tmp/lua-lsm-riscv-build/.config
```

如果结果里没有 `lua`，可以设置为：

```bash
scripts/config \
    --file /tmp/lua-lsm-riscv-build/.config \
    --set-str LSM "landlock,lockdown,yama,integrity,apparmor,lua,bpf"
```

最后解析配置依赖：

```bash
make O=/tmp/lua-lsm-riscv-build \
    ARCH=riscv \
    CROSS_COMPILE=riscv64-linux-gnu- \
    olddefconfig
```

确认关键配置：

```bash
grep -E '^CONFIG_(ARCH_HAS_SETJMP|ARCH_SETJMP|HAS_LUA|LUA|SECURITY_LUA_LSM|LSM)=' \
    /tmp/lua-lsm-riscv-build/.config
```

### 2. 编译内核

仓库的构建脚本硬编码调用了 `gsed`。如果系统没有 `gsed`，先建立一个临时兼容命令：

```bash
mkdir -p /tmp/lua-lsm-tools
ln -sf /usr/bin/sed /tmp/lua-lsm-tools/gsed
```

另外如果编译的是 x86 架构，需要改一下：

```diff
diff --git a/arch/x86/tools/relocs.c b/arch/x86/tools/relocs.c
index e5a2b9a91..d93107697 100644
--- a/arch/x86/tools/relocs.c
+++ b/arch/x86/tools/relocs.c
@@ -287,16 +287,16 @@ static const char *sym_name(const char *sym_strtab, Elf_Sym *sym)
        return name;
 }

-#if BYTE_ORDER == LITTLE_ENDIAN
+#if __BYTE_ORDER == __LITTLE_ENDIAN
 # define le16_to_cpu(val)      (val)
 # define le32_to_cpu(val)      (val)
 # define le64_to_cpu(val)      (val)
-#endif
-
-#if BYTE_ORDER == BIG_ENDIAN
+#elif __BYTE_ORDER == __BIG_ENDIAN
 # define le16_to_cpu(val)      bswap_16(val)
 # define le32_to_cpu(val)      bswap_32(val)
 # define le64_to_cpu(val)      bswap_64(val)
+#else
+# error "Unknown byte order"
 #endif

 static uint16_t elf16_to_cpu(uint16_t val)
```

编译：

```bash
PATH=/tmp/lua-lsm-tools:$PATH \
make O=/tmp/lua-lsm-riscv-build \
    ARCH=riscv \
    CROSS_COMPILE=riscv64-linux-gnu- \
    -j"$(nproc)" \
    Image
```

生成文件：

```text
/tmp/lua-lsm-riscv-build/arch/riscv/boot/Image
/tmp/lua-lsm-riscv-build/vmlinux
```

确认架构：

```bash
file /tmp/lua-lsm-riscv-build/vmlinux
```

### 3. 只运行 setjmp KUnit 测试

启用测试：

```bash
scripts/config \
    --file /tmp/lua-lsm-riscv-build/.config \
    --enable KUNIT \
    --enable RISCV_SETJMP_KUNIT

make O=/tmp/lua-lsm-riscv-build \
    ARCH=riscv \
    CROSS_COMPILE=riscv64-linux-gnu- \
    olddefconfig

PATH=/tmp/lua-lsm-tools:$PATH \
make O=/tmp/lua-lsm-riscv-build \
    ARCH=riscv \
    CROSS_COMPILE=riscv64-linux-gnu- \
    -j"$(nproc)" \
    Image
```

用 QEMU 启动：

```bash
qemu-system-riscv64 \
    -machine virt \
    -m 1G \
    -smp 2 \
    -nographic \
    -kernel /tmp/lua-lsm-riscv-build/arch/riscv/boot/Image \
    -append "console=ttyS0 earlycon=sbi kunit.filter_glob=riscv_setjmp*"
```

预期看到：

```text
# Subtest: riscv_setjmp
ok 1 setjmp_initial_return_test
ok 2 setjmp_zero_value_test
ok 3 setjmp_cross_stack_test
# riscv_setjmp: pass:3 fail:0 skip:0 total:3
```

测试之后会出现：

```text
VFS: Unable to mount root fs
```

这是因为没有提供 rootfs。KUnit 已经在挂载 rootfs 前运行完毕，所以不影响测试结果。

按 `Ctrl-A`，松开后再按 `X`，可退出 QEMU。

### 4. 启动到可交互 Shell

启动：

```bash
qemu-system-riscv64 \
    -machine virt \
    -m 1G \
    -smp 2 \
    -nographic \
    -kernel /tmp/lua-lsm-riscv-build/arch/riscv/boot/Image \
    -initrd /tmp/lua-lsm-initramfs.cpio.gz \
    -append "console=ttyS0 earlycon=sbi rdinit=/init lsm=landlock,lockdown,yama,integrity,apparmor,lua,bpf"
```

进入 shell 后创建挂载点并挂载：

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

检查：

```sh
cat /sys/kernel/security/lua/version
cat /sys/kernel/security/lua/modules
```

加载策略：

```sh
cat /policy.lua > /sys/kernel/security/lua/register
cat /sys/kernel/security/lua/modules
```

如果只是验证 RISC-V `setjmp/longjmp`，做到第 3 部分即可；如果要验证 Lua 策略动态加载、执行与错误回退，则必须准备第 4 部分的 RISC-V initramfs。