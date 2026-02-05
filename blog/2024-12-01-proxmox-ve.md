---
title: 配置 Proxmox VE
tags: ['homelab']
---

参考 https://songxwn.com/PVE-apt-source/

<!-- truncate -->

删除订阅弹窗

```
sed -Ezi.bak "s/(Ext.Msg.show\(\{\s+title: gettext\('No valid sub)/void\(\{ \/\/\1/g" /usr/share/javascript/proxmox-widget-toolkit/proxmoxlib.js && systemctl restart pveproxy.service
```

修改软件源

```
mv /etc/apt/sources.list.d/pve-enterprise.list /etc/apt/sources.list.d/pve-enterprise.list.bak
# 屏蔽订阅企业源
wget https://mirrors.ustc.edu.cn/proxmox/debian/proxmox-release-bookworm.gpg -O /etc/apt/trusted.gpg.d/proxmox-release-bookworm.gpg
echo "deb https://mirrors.ustc.edu.cn/proxmox/debian bookworm pve-no-subscription" > /etc/apt/sources.list.d/pve-no-subscription.list
# 使用Proxmox非企业版中科大源
sed -i 's|^deb http://ftp.debian.org|deb https://mirrors.ustc.edu.cn|g' /etc/apt/sources.list
sed -i 's|^deb http://security.debian.org|deb https://mirrors.ustc.edu.cn/debian-security|g' /etc/apt/sources.list
# 将Debian官方源替换为中科大源
echo "deb https://mirrors.ustc.edu.cn/proxmox/debian/ceph-quincy bookworm no-subscription" > /etc/apt/sources.list.d/ceph.list
# 替换Ceph源
sed -i 's|http://download.proxmox.com|https://mirrors.ustc.edu.cn/proxmox|g' /usr/share/perl5/PVE/APLInfo.pm
# 替换CT镜像下载源（重启PVE才能生效）
apt update 
#更新源列表
apt upgrade 
#升级软件包
apt dist-upgrade 
#升级系统
#apt install openvswitch-switch
```

设置 DHCP

参考 https://weblog.lkiesow.de/20220223-proxmox-test-machine-self-servic/proxmox-server-dhcp.html

修改 /etc/network/interfaces

```
iface vmbr0 inet dhcp
        bridge-ports enp5s0
        bridge-stp off
        bridge-fd 0
```

新建修改 /etc/dhcp/dhclient-exit-hooks.d/update-etc-hosts 
```
if ([ $reason = "BOUND" ] || [ $reason = "RENEW" ])
then
  sed -i "s/^.*\s$(hostname).lan\s.*$/${new_ip_address} $(hostname).lan $(hostname)/" /etc/hosts
fi
```