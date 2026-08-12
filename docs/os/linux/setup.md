# 开发环境搭建

首先获取 Linux 内核源码，参考 [KernelBuild - KernelNewbies](https://kernelnewbies.org/KernelBuild) 在 kernel.org 上下载最新的 Linux 内核源码，我下载起来挺快的。

解压到一个目录下，我解压到 `~/Download/linux/linux-7.1.8`.

然后获取 busybox 源码

```bash
cd ~/Download/linux
git clone https://github.com/mirror/busybox.git
```

编译 busybox

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

编译 Linux 内核

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
