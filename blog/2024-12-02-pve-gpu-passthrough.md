---
title: Proxmox VE 直通核显
tags: ['homelab']
---

参考

https://3os.org/infrastructure/proxmox/gpu-passthrough/igpu-passthrough-to-vm

https://3os.org/infrastructure/proxmox/gpu-passthrough/igpu-split-passthrough/

<!-- truncate -->

下面这篇文章底下有人评论

https://github.com/fire1ce/3os.org/discussions/111#discussioncomment-9361937

When installing Proxmox and choosing ZFS, GRUB is not used as a bootloader, systemd-boot is. So editing /etc/default/grub will have no effect. You need to edit /etc/kernel/cmdline instead of /etc/default/grub for the kernel parameters.

这里折腾了我好久（

## macOS guest 特殊设置

参考 http://vfio.blogspot.com/2016/07/intel-graphics-assignment.html

使用 Legacy 方式，具体操作可见 https://imacos.top/2024/04/22/i44fx/

我的 CPU 是 i5-8600K，下载了他提供的 8265U 的 rom, 用 https://github.com/awilliam/rom-parser 改了设备 ID 就可以用了。