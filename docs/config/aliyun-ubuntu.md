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

## 每月流量限制

安装 vnstat

```bash
apt install -y vnstat iproute2
systemctl enable --now vnstat
```

查看网卡名：

```bash
ip route get 1.1.1.1
```

比如显示 `dev eth0`，那网卡就是 `eth0`。

先初始化统计：

```bash
vnstat -i eth0
systemctl restart vnstat
```

写脚本

```bash
cat <<EOF > /usr/local/sbin/monthly-egress-limit.sh
#!/bin/bash
set -e

IFACE="eth0"
LIMIT_GB=17
RATE="500kbit"
STATE_FILE="/run/egress-limited"

# vnstat --oneline b:
TX_BYTES=$(vnstat -i "$IFACE" --oneline b 2>/dev/null | awk -F';' '{print $10}')

[ -z "$TX_BYTES" ] && exit 0

LIMIT_BYTES=$((LIMIT_GB * 1000 * 1000 * 1000))

if [ "$TX_BYTES" -ge "$LIMIT_BYTES" ]; then
    if [ ! -f "$STATE_FILE" ]; then
        tc qdisc replace dev "$IFACE" root tbf rate "$RATE" burst 32kbit latency 400ms
        touch "$STATE_FILE"
    fi
else
    if [ -f "$STATE_FILE" ]; then
        tc qdisc del dev "$IFACE" root 2>/dev/null || true
        rm -f "$STATE_FILE"
    fi
fi
EOF
chmod +x /usr/local/sbin/monthly-egress-limit.sh
```

用 systemd timer 每 5 分钟检查一次：

```bash
cat > /etc/systemd/system/monthly-egress-limit.service <<'EOF'
[Unit]
Description=Monthly egress traffic limiter

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/monthly-egress-limit.sh
EOF

cat > /etc/systemd/system/monthly-egress-limit.timer <<'EOF'
[Unit]
Description=Run monthly egress traffic limiter periodically

[Timer]
OnBootSec=1min
OnUnitActiveSec=5min
Persistent=true

[Install]
WantedBy=timers.target
EOF

systemctl daemon-reload
systemctl enable --now monthly-egress-limit.timer
```

## 每月流量超限关机保险

为了最大程度确保流量不超限，设置一个最简单的定时检测脚本，不成功即关机。参考 [为云主机实现网络达量停机](https://tao.zz.ac/unix/vnstat.html) 。

```bash
cat <<EOF > /usr/local/sbin/traffic.sh
#!/usr/bin/env bash

LIMIT="${1:-50000000000}"

result=$(
  vnstat -i eth0 -m --json \
    | jq -r ".interfaces[0].traffic.month[0] | if .tx > $LIMIT then \"err\" else \"ok\" end"
)

if [ "$result" != "ok" ]; then
    systemctl poweroff
fi
EOF
chmod +x /usr/local/sbin/traffic.sh
```

然后配置 crontab

```
*/15 * * * * /usr/local/sbin/traffic.sh 30000000000
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

## Coturn

### 1. 安装 coturn

```bash
sudo apt update
sudo apt install -y coturn openssl
```

启用 coturn：

```bash
sudo sed -i 's/^#\?TURNSERVER_ENABLED=.*/TURNSERVER_ENABLED=1/' /etc/default/coturn
```

### 2. 先让 Caddy 给域名签证书

如果 `turn.example.com` 现在没有被 Caddy 使用，可以在 `/etc/caddy/Caddyfile` 里放一个占位站点：

```caddyfile
turn.example.com {
	respond "coturn cert ok"
}
```

然后：

```bash
sudo systemctl reload caddy
```

确认 Caddy 已经拿到证书：

```bash
sudo find /var/lib/caddy/.local/share/caddy/certificates \
  -type f \( -name 'turn.example.com.crt' -o -name 'turn.example.com.key' \) -print
```

### 3. 把 Caddy 证书复制给 coturn 用

不建议让 `turnserver` 直接读 Caddy 的私钥目录，权限会很别扭。更稳的是复制一份到 coturn 自己的目录。

```bash
sudo tee /usr/local/sbin/sync-caddy-cert-to-coturn.sh >/dev/null <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

DOMAIN="turn.example.com"
SRC_BASE="/var/lib/caddy/.local/share/caddy/certificates"
DST_DIR="/etc/coturn/certs"

CERT="$(find "$SRC_BASE" -type f -path "*/$DOMAIN/$DOMAIN.crt" | sort | tail -n1)"
KEY="$(find "$SRC_BASE" -type f -path "*/$DOMAIN/$DOMAIN.key" | sort | tail -n1)"

if [ -z "$CERT" ] || [ -z "$KEY" ]; then
  echo "Caddy certificate for $DOMAIN not found" >&2
  exit 1
fi

install -d -o turnserver -g turnserver -m 0750 "$DST_DIR"

changed=0

if ! [ -f "$DST_DIR/$DOMAIN.crt" ] || ! cmp -s "$CERT" "$DST_DIR/$DOMAIN.crt"; then
  install -o turnserver -g turnserver -m 0440 "$CERT" "$DST_DIR/$DOMAIN.crt"
  changed=1
fi

if ! [ -f "$DST_DIR/$DOMAIN.key" ] || ! cmp -s "$KEY" "$DST_DIR/$DOMAIN.key"; then
  install -o turnserver -g turnserver -m 0440 "$KEY" "$DST_DIR/$DOMAIN.key"
  changed=1
fi

if [ "$changed" = 1 ]; then
  systemctl restart coturn || true
fi
EOF

sudo chmod +x /usr/local/sbin/sync-caddy-cert-to-coturn.sh
sudo /usr/local/sbin/sync-caddy-cert-to-coturn.sh
```

coturn 的 `cert=` 和 `pkey=` 要用 PEM 文件，支持绝对路径。([GitHub](https://github.com/coturn/coturn/blob/master/examples/etc/turnserver.conf))

再加一个定时同步，防止 Caddy 续期后 coturn 还用旧证书：

```bash
sudo tee /etc/systemd/system/coturn-cert-sync.service >/dev/null <<'EOF'
[Unit]
Description=Sync Caddy certificate to coturn

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/sync-caddy-cert-to-coturn.sh
EOF

sudo tee /etc/systemd/system/coturn-cert-sync.timer >/dev/null <<'EOF'
[Unit]
Description=Periodically sync Caddy certificate to coturn

[Timer]
OnBootSec=5min
OnUnitActiveSec=12h
Persistent=true

[Install]
WantedBy=timers.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now coturn-cert-sync.timer
```

### 4. 确认阿里云 IPv4/IPv6 情况

```bash
PUBLIC4="$(curl -4s https://ifconfig.co)"
PRIVATE4="$(ip -4 -o addr show scope global | awk '!/docker|podman|br-|virbr|wg|tun|tailscale/ {print $4; exit}' | cut -d/ -f1)"

PUBLIC6="$(curl -6s https://ifconfig.co || true)"

echo "PUBLIC4=$PUBLIC4"
echo "PRIVATE4=$PRIVATE4"
echo "PUBLIC6=$PUBLIC6"
ip -6 addr show scope global
```

阿里云 VPC 里，公网 IPv4 / EIP 往往是网关 NAT 映射，不一定直接出现在系统网卡上；阿里云文档也说明，在安全组里控制具体 ECS 时通常用实例的私网 IP，因为公网 IP/EIP 是云网关上的 NAT 地址。([alibabacloud.com](https://www.alibabacloud.com/help/en/ecs/user-guide/security-group-rules))

所以：

如果 `PUBLIC4 != PRIVATE4`，coturn 里用：

```ini
external-ip=公网IPv4/私网IPv4
```

例如：

```ini
external-ip=8.8.8.8/172.16.1.23
```

IPv6 一般是直接分配到网卡上的公网 IPv6，确认 `ip -6 addr show scope global` 里有它即可。阿里云 IPv6 需要 VPC/vSwitch 开 IPv6，系统网卡识别到 global IPv6，并配置 IPv6 安全组规则。

### 5. 写 coturn 配置

生成一个用户密码：

```bash
TURN_USER="turnuser"
TURN_PASS="$(openssl rand -base64 32)"
TURN_KEY="$(turnadmin -k -u "$TURN_USER" -r turn.example.com -p "$TURN_PASS")"

echo "TURN_USER=$TURN_USER"
echo "TURN_PASS=$TURN_PASS"
echo "TURN_KEY=$TURN_KEY"
```

把输出里的 `TURN_PASS` 保存好，客户端要用。`TURN_KEY` 写进配置。coturn 官方示例里也建议可以用 `turnadmin -k` 生成 key，避免把明文密码直接写进配置。([GitHub](https://github.com/coturn/coturn/blob/master/examples/etc/turnserver.conf))

编辑：

```bash
sudo nano /etc/turnserver.conf
```

写入下面配置。注意把 `external-ip=` 和 `user=` 改成你自己的实际值。

```ini
# 基本身份
server-name=turn.example.com
realm=turn.example.com

# 监听端口
listening-port=3478
tls-listening-port=5349

# IPv4 + IPv6 双栈监听
listening-ip=0.0.0.0
listening-ip=::

# 阿里云 IPv4 如果是公网 NAT 映射，写成 公网IPv4/私网IPv4
# 示例：
# external-ip=8.8.8.8/172.16.1.23
external-ip=你的公网IPv4/你的私网IPv4

# 如果你的公网 IPv4 直接在网卡上，也可以写：
# external-ip=你的公网IPv4

# TLS 证书，来自 Caddy 的复制副本
cert=/etc/coturn/certs/turn.example.com.crt
pkey=/etc/coturn/certs/turn.example.com.key

# 认证
fingerprint
lt-cred-mech

# 把这里替换成openssl rand -base64 32生成的密码
user=turnuser:替换成密码

# relay 端口范围，小规模够用；人多就扩大
min-port=49160
max-port=49200

# 配额，防止被滥用
user-quota=12
total-quota=100
stale-nonce=600

# 日志走 syslog / journalctl
syslog

# 安全：不要允许 multicast peer
no-multicast-peers

# 安全：禁止被当成内网代理打私网/链路本地地址
denied-peer-ip=0.0.0.0-0.255.255.255
denied-peer-ip=10.0.0.0-10.255.255.255
denied-peer-ip=100.64.0.0-100.127.255.255
denied-peer-ip=127.0.0.0-127.255.255.255
denied-peer-ip=169.254.0.0-169.254.255.255
denied-peer-ip=172.16.0.0-172.31.255.255
denied-peer-ip=192.168.0.0-192.168.255.255
denied-peer-ip=198.18.0.0-198.19.255.255
```

coturn 支持 `listening-ip=0.0.0.0` 和 `listening-ip=::` 来监听 IPv4/IPv6；默认普通 TURN 端口是 3478，TLS 端口是 5349；relay UDP 端口范围默认是 49152-65535，可以用 `min-port` / `max-port` 收窄。([GitHub](https://github.com/coturn/coturn/blob/master/examples/etc/turnserver.conf))

启动：

```bash
sudo systemctl enable --now coturn
sudo systemctl restart coturn
```

看日志：

```bash
sudo journalctl -u coturn -e --no-pager
```

### 6. 阿里云安全组 / UFW 放行

阿里云安全组入方向至少放这些：

```text
TCP 3478
UDP 3478
TCP 5349
UDP 5349
UDP 49160-49200
```

IPv6 也要单独加 IPv6 入方向规则，来源可以是 `::/0`。阿里云安全组规则本来就是用于控制 ECS 入/出方向流量的，规则按协议、端口和授权对象匹配。

如果系统开了 UFW：

```bash
sudo ufw allow 3478/tcp
sudo ufw allow 3478/udp
sudo ufw allow 5349/tcp
sudo ufw allow 5349/udp
sudo ufw allow 49160:49200/udp
```

确认 `/etc/default/ufw` 里：

```ini
IPV6=yes
```

然后：

```bash
sudo ufw reload
```

### 7. 测试

本机检查监听：

```bash
sudo ss -lntup | grep -E '3478|5349|turnserver'
```

测试 TLS 证书：

```bash
openssl s_client -connect turn.example.com:5349 -servername turn.example.com -brief
```

测试 TURN：

```bash
turnutils_uclient -v -u turnuser -w '你的TURN_PASS' turn.example.com
```

测试 TURNS：

```bash
turnutils_uclient -v -S -p 5349 -u turnuser -w '你的TURN_PASS' turn.example.com
```

客户端配置一般写：

```js
[
  {
    urls: [
      "stun:turn.example.com:3478",
      "turn:turn.example.com:3478?transport=udp",
      "turn:turn.example.com:3478?transport=tcp",
      "turns:turn.example.com:5349?transport=tcp"
    ],
    username: "turnuser",
    credential: "你的密码"
  }
]
```
