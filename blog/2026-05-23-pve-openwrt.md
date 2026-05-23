---
title: 单网口 PVE 小主机利用 VLAN 跑 OpenWrt 作主路由
tags: ['OpenWrt']
---

需求是用 PVE 上的 OpenWrt 虚拟机做主路由，运行 NAT 和透明代理，以及一些家长控制之类的。硬路由当 AP 用。

方案：PVE 单网口走 VLAN trunk，OpenWrt VM 做主路由，硬 OpenWrt 平时只做“交换机 + AP + 救援入口”。

<!-- truncate -->

核心拓扑

```
光猫/ONT
  │
  │ 硬路由 WAN 口，作为 VLAN10 access 口
  ▼
硬 OpenWrt 路由器：只做 VLAN 交换机 + AP
  │
  │ 一根网线，trunk：VLAN10(WAN) + VLAN20(LAN) + VLAN99(救援)
  ▼
PVE 单网口 vmbr0
  └── OpenWrt VM
        ├── eth0 / VLAN20：LAN，192.168.20.1
        └── eth1 / VLAN10：WAN，DHCP/PPPoE
```

## VLAN 和地址规划

| 用途     | VLAN | 网段 / 地址       | 说明                       |
| -------- | ---- | ----------------- | -------------------------- |
| WAN 透传 | 10   | 无内网 IP         | 光猫到 OpenWrt VM 的 WAN   |
| 主 LAN   | 20   | `192.168.20.0/24` | 家里正常网络               |
| 救援管理 | 99   | `192.168.99.0/24` | 软路由炸了还能进硬路由/PVE |

## 准备 OpenWrt 镜像

使用 [Firmware Selector](https://firmware-selector.openwrt.org/?target=x86%2F64&id=generic) 制作镜像

我的配置：

预装软件包

```
apk-mbedtls base-files ca-bundle dnsmasq dropbear e2fsprogs firewall4 fstools grub2-bios-setup kmod-button-hotplug kmod-nft-offload libc libgcc libustream-mbedtls logd mkf2fs mtd netifd nftables odhcp6c odhcpd-ipv6only partx-utils ppp ppp-mod-pppoe procd-ujail uci uclient-fetch urandom-seed urngd kmod-amazon-ena kmod-amd-xgbe kmod-bnx2 kmod-dwmac-intel kmod-e1000e kmod-e1000 kmod-forcedeth kmod-fs-vfat kmod-igb kmod-igc kmod-ixgbe kmod-r8169 kmod-tg3 kmod-drm-i915 luci luci-app-attendedsysupgrade

ddns-scripts ddns-scripts-cloudflare resolveip luci-proto-wireguard openssh-sftp-server
```

首次启动时运行的脚本（uci-defaults）

```
#!/bin/sh

# PVE OpenWrt VM 自动初始化配置
# net0 = eth0 = LAN = VLAN20
# net1 = eth1 = WAN = VLAN10

set -e

# ---------- system ----------
uci set system.@system[0].hostname='pve-openwrt'
uci set system.@system[0].timezone='CST-8'
uci set system.@system[0].zonename='Asia/Shanghai'

# ---------- network ----------
uci -q delete network.lan
uci -q delete network.wan
uci -q delete network.wan6

uci set network.lan='interface'
uci set network.lan.device='eth0'
uci set network.lan.proto='static'
uci set network.lan.ipaddr='192.168.20.1'
uci set network.lan.netmask='255.255.255.0'
uci set network.lan.ip6assign='60'

uci set network.wan='interface'
uci set network.wan.device='eth1'
uci set network.wan.proto='pppoe'
uci set network.wan.username='你的宽带账号'
uci set network.wan.password='你的宽带密码'
uci set network.wan.ipv6='auto'
uci set network.wan.norelease='1'

uci commit network

# ---------- DHCP ----------
uci -q delete dhcp.lan
uci -q delete dhcp.wan

uci set dhcp.lan='dhcp'
uci set dhcp.lan.interface='lan'
uci set dhcp.lan.start='100'
uci set dhcp.lan.limit='150'
uci set dhcp.lan.leasetime='12h'
uci set dhcp.lan.dhcpv4='server'
uci set dhcp.lan.ra='server'
uci set dhcp.lan.dhcpv6='server'
uci set dhcp.lan.ra_preference='medium'
uci set dhcp.lan.ra_default='1'
uci add_list dhcp.lan.ra_flags='other-config'

uci set dhcp.wan='dhcp'
uci set dhcp.wan.interface='wan'
uci set dhcp.wan.ignore='1'

uci commit dhcp

# ---------- firewall ----------
uci -q delete firewall.lan
uci -q delete firewall.wan
uci -q delete firewall.@forwarding[0]

uci set firewall.lan='zone'
uci set firewall.lan.name='lan'
uci add_list firewall.lan.network='lan'
uci set firewall.lan.input='ACCEPT'
uci set firewall.lan.output='ACCEPT'
uci set firewall.lan.forward='ACCEPT'

uci set firewall.wan='zone'
uci set firewall.wan.name='wan'
uci add_list firewall.wan.network='wan'
uci set firewall.wan.input='REJECT'
uci set firewall.wan.output='ACCEPT'
uci set firewall.wan.forward='REJECT'
uci set firewall.wan.masq='1'
uci set firewall.wan.mtu_fix='1'

uci add firewall forwarding
uci set firewall.@forwarding[-1].src='lan'
uci set firewall.@forwarding[-1].dest='wan'

uci commit firewall

# ---------- dropbear / SSH ----------
uci set dropbear.@dropbear[0].PasswordAuth='on'
uci set dropbear.@dropbear[0].RootPasswordAuth='on'
uci set dropbear.@dropbear[0].Port='22'
uci commit dropbear

# ---------- root password ----------
# 第一次启动后请马上用 passwd 修改密码。
# 如果你想预设密码，可以取消下面两行注释：
# echo 'root:你的强密码' | chpasswd

# ---------- services ----------
/etc/init.d/network restart
/etc/init.d/dnsmasq restart
/etc/init.d/firewall restart
/etc/init.d/dropbear restart

exit 0
```

请求构建，构建好了下载第一个 COMBINED-EFI (EXT4)

然后 scp 传到 PVE 上，解压

```
scp openwrt-25.12.4-xxx-x86-64-generic-ext4-combined-efi.img.gz root@pve:~/
ssh root@pve
gunzip openwrt-25.12.4-xxx-x86-64-generic-ext4-combined-efi.img.gz
```

## 创建和启动虚拟机

创建虚拟机

```
qm create 106 \
  --name openwrt-router \
  --memory 512 \
  --cores 2 \
  --machine q35 \
  --bios ovmf \
  --ostype l26 \
  --serial0 socket \
  --vga serial0
qm importdisk 106 openwrt-25.12.4-xxx-x86-64-generic-ext4-combined-efi.img local-zfs
qm set 106 \
  --scsihw virtio-scsi-pci \
  --scsi0 local-zfs:vm-106-disk-0 \
  --boot order=scsi0
```

添加网卡

```
qm set 106 --net0 virtio,bridge=vmbr0,tag=20
qm set 106 --net1 virtio,bridge=vmbr0,tag=10
```

设置开机启动（`up=20` 是启动 OpenWrt 20 秒后再顺序启动别的虚拟机）

```
qm set 106 --onboot 1 --startup order=1,up=20
```

启动虚拟机

```
qm start 106
```

然后去到控制台设置一下 root 密码

## 设置 PVE 网络

PVE 单物理网口假设叫 `enp1s0`，你要改成自己的网卡名。

`/etc/network/interfaces` 改成这样（要把 `enp0s31f6` 换成你的网卡名）：

```
auto lo
iface lo inet loopback

iface enp0s31f6 inet manual

auto vmbr0
iface vmbr0 inet manual
        bridge-ports enp0s31f6
        bridge-stp off
        bridge-fd 0
        bridge-vlan-aware yes
        bridge-vids 2-4094

auto vmbr0.20
iface vmbr0.20 inet static
        address 192.168.20.9/24
        gateway 192.168.20.1

auto vmbr0.99
iface vmbr0.99 inet static
        address 192.168.99.9/24
```

改完后可以用以下命令应用。注意！这么做会暂时断开与 PVE 的连接。等在硬路由配置好 VLAN 后才能连上。

```
ifreload -a
```

## 设置硬路由

硬路由也是 OpenWrt

- wan: 光猫
- lan2: PVE 主机
- lan3: 我房间的另一台路由器，作傻瓜交换机用。最终连到我的 ArchLinux 主机。
- lan4: 网管交换机

```
uci add_list network.@device[0].ports 'wan'

# 1. VLAN 99: rescue, for connectivity to this router
uci add network bridge-vlan
uci set network.@bridge-vlan[-1].device='br-lan'
uci set network.@bridge-vlan[-1].vlan='99'
uci add_list network.@bridge-vlan[-1].ports='lan2:t'
uci add_list network.@bridge-vlan[-1].ports='lan3:t'
uci add_list network.@bridge-vlan[-1].ports='lan4:t'

# 2. VLAN 20: PVE lan, for lan on PVE OpenWrt
uci add network bridge-vlan
uci set network.@bridge-vlan[-1].device='br-lan'
uci set network.@bridge-vlan[-1].vlan='20'
uci add_list network.@bridge-vlan[-1].ports='lan2:t'
uci add_list network.@bridge-vlan[-1].ports='lan3:t'
uci add_list network.@bridge-vlan[-1].ports='lan4:u*'

# 3. VLAN 10: modem, for wan PPPoE on PVE OpenWrt
uci add network bridge-vlan
uci set network.@bridge-vlan[-1].device='br-lan'
uci set network.@bridge-vlan[-1].vlan='10'
uci set network.@bridge-vlan[-1].local='0'
uci add_list network.@bridge-vlan[-1].ports='lan2:t'
uci add_list network.@bridge-vlan[-1].ports='wan:u*'

# Change Interface accordingly
uci set network.wan.disabled='1'
uci set network.lan.device='br-lan.99'
uci del network.lan.ipaddr
uci add_list network.lan.ipaddr='192.168.99.1/24'

# Add 'pvelan' interface configuration
uci set network.pvelan=interface
uci set network.pvelan.proto='dhcp'
uci set network.pvelan.device='br-lan.20'
uci set network.pvelan.multipath='off'

uci commit network
/etc/init.d/network restart
```

PC 这边加一个专门的 VLAN

```
sudo nmcli connection add type vlan con-name "rescue99" ifname "vlan99" vlan.parent "enp2s0" vlan.id 99
```

然后关闭默认的连接，启用这个 rescue99

```
nmcli connection down "Wired connection 1"
nmcli connection up rescue99
```

现在来连通 PVE 里的 Op 虚拟机

```
sudo nmcli connection add type vlan con-name "pve20" ifname "vlan20" vlan.parent "enp2s0" vlan.id 20
nmcli connection up pve20
```

访问 192.168.20.1，设置一下 PPPoE 拨号账号密码。

此时应该可以上网。
