# 阿里云 Ubuntu 配置

## 内存优化

配置选的 2C2G, 开机一看

```
# free
               total        used        free      shared  buff/cache   available
Mem:         1651696      393420      418248        2652     1012996     1258276
Swap:              0           0           0
```

内存才 1.6 GiB！搜索了一下看到 [V2EX](https://v2ex.com/t/998120) 有人说是 crashkernel 占用的. 修改 `/etc/default/grub` 删除相关内核选项后，

```bash
grub-mkconfig -o /boot/grub/grub.cfg
```

然后重启。现在再来看内存：

```
# free
               total        used        free      shared  buff/cache   available
Mem:         1913840      404420     1308076        2672      353696     1509420
Swap:              0           0           0
```

约为 1.825 GiB, 使用 `dmesg | grep -i "Memory:"` 查看实际物理内存：

```
# dmesg | grep -i "Memory:"
[    0.045721] Memory: 1830108K/1995668K available (22528K kernel code, 4436K rwdata, 14408K rodata, 4920K init, 4792K bss, 165300K reserved, 0K cma-reserved)
[    0.151181] Freeing SMP alternatives memory: 48K
[    0.597818] Freeing initrd memory: 54708K
[    0.854832] Freeing unused decrypted memory: 2028K
[    0.860629] Freeing unused kernel image (initmem) memory: 4920K
[    0.867992] Freeing unused kernel image (rodata/data gap) memory: 1976K
# free
               total        used        free      shared  buff/cache   available
Mem:         1913840      361872     1305672        2664      399288     1551968
Swap:              0           0           0
```

1995668 KiB 和 2048 MiB 相比还是有 96.7 MiB 左右的差值，可能是 BIOS 等占的内存。最终可用内存为 1.825 GiB.

另外可以关掉 aliyun-service

```bash
systemctl disable --now aliyun.service
```

## Docker/Podman 相关

打开 IP 转发。

```bash
echo "net.ipv4.ip_forward=1" >> /etc/sysctl.conf
sysctl -p
```

设置 Podman 默认 registries.

```bash
cat <<EOF > /etc/containers/registries.conf
unqualified-search-registries = ["docker.io", "quay.io"]
EOF
```

## Vaultwarden

创建数据目录和环境变量文件

```bash
mkdir -p /opt/vaultwarden/data
install -o0 -g0 -m600 /dev/null /etc/vaultwarden.env
cat <<EOF > /etc/vaultwarden.env
ROCKET_PORT=8080
DOMAIN=https://example.com
SIGNUPS_ALLOWED=false
WEBSOCKET_ENABLED=true
EOF
```

设置 container

```bash
mkdir -p /etc/containers/systemd
cat <<EOF > /etc/containers/systemd/vaultwarden.container
[Unit]
Description=Vaultwarden container
After=network-online.target

[Container]
ContainerName=vaultwarden
Image=ghcr.io/dani-garcia/vaultwarden:latest
AutoUpdate=registry
EnvironmentFile=/etc/vaultwarden.env
Volume=/opt/vaultwarden/data:/data
PublishPort=127.0.0.1:8080:8080
Exec=/start.sh

[Service]
Restart=always

[Install]
WantedBy=multi-user.target
EOF
```

由于已经写好了 `WantedBy=multi-user.target` 所以默认就会开机自动启动。

手动启动：

```bash
systemctl start vaultwarden.service
```

Caddy 配置：

```
example.com {
    reverse_proxy 127.0.0.1:8080
}
```
