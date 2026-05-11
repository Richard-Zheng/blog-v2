# 阿里云 Ubuntu 配置

## 修改 SSH 端口

注意要 daemon reload 才能生效。

```
sudo sed -i 's/^#\?Port.*/Port 23333/' /etc/ssh/sshd_config
sudo systemctl daemon-reload
sudo systemctl restart ssh.socket
```

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
ROCKET_ADDRESS=127.0.0.1
DOMAIN=https://vw.example.com
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
Network=host
Exec=/start.sh

[Service]
Restart=always

[Install]
WantedBy=multi-user.target
EOF
```

根据 [podman-systemd.unit 文档](https://docs.podman.io/en/latest/markdown/podman-systemd.unit.5.html#enabling-unit-files)，由于 transient 的 unit 不支持 `systemd enable`，为了能实现开机自启的效果 systemd generator 在看到 `WantedBy=multi-user.target` 这一行之后就会自动将其添加到开机启动项中。

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

## OpenGist

创建数据目录

```bash
sudo mkdir -p /opt/opengist
```


```bash
sudo vim /etc/containers/systemd/opengist.container
```

写入：

```ini
[Unit]
Description=Opengist container
After=network-online.target
Wants=network-online.target

[Container]
ContainerName=opengist
Image=ghcr.io/thomiceli/opengist:1
AutoUpdate=registry
PublishPort=127.0.0.1:6157:6157
PublishPort=2222:2222
Volume=/opt/opengist:/opengist

[Service]
Restart=always

[Install]
WantedBy=multi-user.target
```

SSH 端口 `2222` 直接暴露，因为 Git SSH clone/push 需要外部访问。官方说明如果不用 SSH，可以删掉 `2222:2222` 这个端口。

Caddy 反代

```caddyfile
gist.example.com {
    reverse_proxy 127.0.0.1:6157
}
```

备份：

```bash
sudo tar czf opengist-backup-$(date +%F).tar.gz /opt/opengist
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
denied-peer-ip=::1
denied-peer-ip=fe80::-fec0::
denied-peer-ip=fc00::-fdff:ffff:ffff:ffff:ffff:ffff:ffff:ffff
denied-peer-ip=::ffff:0.0.0.0-::ffff:255.255.255.255
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

## PairDrop

```text
浏览器
  ├─ HTTPS 443  → Caddy → PairDrop 容器 127.0.0.1:3000
  └─ STUN/TURN → coturn 3478 / 5349 / relay UDP ports
```

PairDrop 官方文档明确说，跨网络传输需要自己的 TURN server；PairDrop 通过 `RTC_CONFIG` 指定给浏览器使用的 STUN/TURN 配置文件。

重点：**PairDrop 容器不需要访问 coturn**。`rtc_config.json` 是发给浏览器的，浏览器会直接连接。

给 PairDrop 写 RTC 配置

创建目录：

```bash
sudo mkdir -p /opt/pairdrop
```

写 `/opt/pairdrop/rtc_config.json`：

```bash
sudo tee /opt/pairdrop/rtc_config.json >/dev/null <<'EOF'
{
  "sdpSemantics": "unified-plan",
  "iceServers": [
    {
      "urls": [
        "stun:cttf.example.com:3478",
        "turn:cttf.example.com:3478?transport=udp",
        "turn:cttf.example.com:3478?transport=tcp",
        "turns:cttf.example.com:5349?transport=tcp"
      ],
      "username": "pairdrop",
      "credential": "换成coturn里user=pairdrop:后面的那个密码"
    }
  ]
}
EOF
```

这个文件会被 PairDrop 提供给浏览器，所以这里的 TURN 用户名密码本来就会暴露给访问你 PairDrop 的客户端。自用没问题；如果公开给很多人用，就要靠长密码、配额、防火墙和日志监控控制滥用。

---

Podman Quadlet 部署 PairDrop

用 system-level Quadlet：

```bash
sudo tee /etc/containers/systemd/pairdrop.container >/dev/null <<'EOF'
[Unit]
Description=PairDrop container
After=network-online.target
Wants=network-online.target

[Container]
ContainerName=pairdrop
Image=lscr.io/linuxserver/pairdrop:latest

PublishPort=127.0.0.1:3000:3000

Volume=/opt/pairdrop/rtc_config.json:/home/node/app/rtc_config.json:ro

Environment=PUID=1000
Environment=PGID=1000
Environment=TZ=Asia/Shanghai
Environment=RATE_LIMIT=false
Environment=WS_FALLBACK=false
Environment=RTC_CONFIG=/home/node/app/rtc_config.json
Environment=DEBUG_MODE=false

AutoUpdate=registry

[Service]
Restart=always
TimeoutStartSec=900

[Install]
WantedBy=multi-user.target
EOF
```

---

4. Caddy 反代 PairDrop

例如 `/etc/caddy/Caddyfile`：

```caddyfile
cttf.example.com {
	reverse_proxy 127.0.0.1:3000
}
```

然后：

```bash
sudo systemctl reload caddy
```

Caddy 的 `reverse_proxy` 默认会设置或追加 `X-Forwarded-For`，并设置 `X-Forwarded-Proto` 和 `X-Forwarded-Host`。([Caddy Web Server][3])

---

6. 测试

看 PairDrop：

```bash
curl -I http://127.0.0.1:3000
curl -I https://cttf.example.com
```

看 coturn：

```bash
sudo ss -lntup | grep -E '3478|5349|turnserver'
```

测 TLS：

```bash
openssl s_client -connect cttf.example.com:5349 -servername cttf.example.com -brief
```

测 TURN：

```bash
turnutils_uclient -v -u pairdrop -w '你的TURN密码' cttf.example.com
```

---

关于 `WS_FALLBACK`

我建议你先保持：

```ini
WS_FALLBACK=false
```

因为你已经有 coturn 了。`WS_FALLBACK=true` 会在 WebRTC 不可用时通过 PairDrop 服务器中转，但官方文档提醒，这样就不是 peer-to-peer 了，流量会走服务器，并且服务器可读这部分 fallback 流量。

## Authelia

建议架构这样分：

```text
Caddy
 ├─ auth.example.com  -> Authelia
 ├─ drop.example.com  -> PairDrop，Caddy forward_auth 到 Authelia
 ├─ gist.example.com  -> Opengist，应用内 OIDC 登录到 Authelia
 └─ hs.example.com    -> Headscale，不要加 forward_auth
```

原因：PairDrop 本身没有完善登录系统，适合用 Caddy + Authelia 反代保护；Opengist 原生支持 OpenID Connect，所以更适合直接接 Authelia 做 SSO；Headscale 也支持 OIDC，但不要用 Caddy forward_auth 套在 Headscale 外面。Authelia 的 Caddy 集成用的是 Caddy 官方 `forward_auth` 机制，Authelia 文档要求 Caddy v2.5.1+；Opengist 官方文档也明确支持 GitHub、Gitea、GitLab 和 OpenID Connect；Headscale 反代必须支持 WebSocket，Caddy 的 `reverse_proxy` 可以正常处理。([Authelia][1])

下面我用这些占位域名，你自己替换：

```text
auth.example.com
drop.example.com
gist.example.com
hs.example.com
example.com
```

部署 Authelia

创建目录：

```bash
sudo mkdir -p /opt/authelia/config /opt/authelia/secrets
sudo chmod 700 /opt/authelia /opt/authelia/config /opt/authelia/secrets
```

生成几个密钥：

```bash
sudo bash -c 'tr -dc A-Za-z0-9 </dev/urandom | head -c 64 > /opt/authelia/secrets/session_secret'
sudo bash -c 'tr -dc A-Za-z0-9 </dev/urandom | head -c 64 > /opt/authelia/secrets/storage_encryption_key'
sudo bash -c 'tr -dc A-Za-z0-9 </dev/urandom | head -c 64 > /opt/authelia/secrets/jwt_secret'
sudo bash -c 'tr -dc A-Za-z0-9 </dev/urandom | head -c 64 > /opt/authelia/secrets/oidc_hmac_secret'
sudo chmod 600 /opt/authelia/secrets/*
```

生成 OIDC RSA 私钥：

```bash
sudo openssl genrsa -out /opt/authelia/secrets/oidc.rsa.pem 2048
sudo openssl pkcs8 -topk8 -inform PEM -outform PEM -nocrypt \
  -in /opt/authelia/secrets/oidc.rsa.pem \
  -out /opt/authelia/secrets/oidc.private.pem
sudo chmod 600 /opt/authelia/secrets/oidc.private.pem
```

Authelia 的 OIDC Provider 需要 HMAC secret 和至少一个用于签名的 JWK；RSA key 最低 2048 bit，`RS256` 是常见配置。([GitHub][2])

生成你的 Authelia 用户密码 hash：

```bash
sudo podman run --rm -it ghcr.io/authelia/authelia:4.39.19 \
  authelia crypto hash generate argon2
```

它会让你输入密码，然后输出类似：

```text
Digest: $argon2id$v=19$...
```

记下 `Digest:` 后面的整段。Authelia 官方建议用 Authelia CLI 或容器生成密码 hash。([Authelia][3])

创建用户文件：

```bash
sudo nano /opt/authelia/config/users_database.yml
```

内容：

```yaml
users:
  fallrain:
    disabled: false
    displayname: "fallrain"
    password: "$argon2id$v=19$这里换成你的hash"
    email: "you@example.com"
    groups:
      - admins
```

---

### 生成 Opengist OIDC secret

这个 secret 有两份：

一份明文给 Opengist 用；一份 hash 放进 Authelia 配置。

```bash
OPENGIST_SECRET="$(tr -dc A-Za-z0-9 </dev/urandom | head -c 64)"
echo "$OPENGIST_SECRET" | sudo tee /opt/authelia/secrets/opengist_oidc_secret_plain
sudo chmod 600 /opt/authelia/secrets/opengist_oidc_secret_plain

sudo podman run --rm ghcr.io/authelia/authelia:4.39.19 \
  authelia crypto hash generate pbkdf2 --variant sha512 --password "$OPENGIST_SECRET"
```

记下输出的 PBKDF2 hash，放到后面的 `client_secret`。

Authelia 的 OIDC client 配置支持 `client_secret` 存 hash；官方 Opengist 集成示例也使用 PBKDF2 hash，并要求 Opengist 回调地址为 `/oauth/openid-connect/callback`。([Authelia][4])

---

### 写 Authelia 配置

先把几个 secret 读出来：

```bash
sudo cat /opt/authelia/secrets/session_secret
sudo cat /opt/authelia/secrets/storage_encryption_key
sudo cat /opt/authelia/secrets/jwt_secret
sudo cat /opt/authelia/secrets/oidc_hmac_secret
sudo cat /opt/authelia/secrets/oidc.private.pem
```

创建配置：

```bash
sudo nano /opt/authelia/config/configuration.yml
```

内容如下，把域名、secret、OIDC 私钥、Opengist client hash 全部替换掉：

```yaml
theme: auto

server:
  address: tcp://0.0.0.0:9091/

log:
  level: info
  format: text

totp:
  issuer: example.com

identity_validation:
  reset_password:
    jwt_secret: "替换为 /opt/authelia/secrets/jwt_secret 的内容"

authentication_backend:
  file:
    path: /config/users_database.yml

access_control:
  default_policy: deny
  rules:
    - domain: drop.example.com
      policy: one_factor

session:
  secret: "替换为 /opt/authelia/secrets/session_secret 的内容"
  cookies:
    - domain: example.com
      authelia_url: https://auth.example.com
      default_redirection_url: https://drop.example.com

storage:
  encryption_key: "替换为 /opt/authelia/secrets/storage_encryption_key 的内容"
  local:
    path: /config/db.sqlite3

notifier:
  filesystem:
    filename: /config/notification.txt

identity_providers:
  oidc:
    hmac_secret: "替换为 /opt/authelia/secrets/oidc_hmac_secret 的内容"
    jwks:
      - key: |
          -----BEGIN PRIVATE KEY-----
          这里粘贴 /opt/authelia/secrets/oidc.private.pem 内容
          注意每行前面保留 10 个空格
          -----END PRIVATE KEY-----

    clients:
      - client_id: opengist
        client_name: Opengist
        client_secret: "$pbkdf2-sha512$这里换成刚才生成的Opengist secret hash"
        authorization_policy: one_factor
        redirect_uris:
          - https://gist.example.com/oauth/openid-connect/callback
        scopes:
          - openid
          - email
          - profile
          - groups
        grant_types:
          - authorization_code
        token_endpoint_auth_method: client_secret_post
```

注意：`auth.example.com` 必须是 HTTPS，Authelia 官方明确要求 Authelia 通过 HTTPS 提供服务。([Authelia][5])

---

### Authelia Quadlet

```bash
sudo nano /etc/containers/systemd/authelia.container
```

内容：

```ini
[Unit]
Description=Authelia container
After=network-online.target
Wants=network-online.target

[Container]
ContainerName=authelia
Image=ghcr.io/authelia/authelia:4.39.19
AutoUpdate=registry
PublishPort=127.0.0.1:9091:9091
Volume=/opt/authelia/config:/config

[Service]
Restart=always

[Install]
WantedBy=multi-user.target
```

启动：

```bash
sudo systemctl daemon-reload
sudo systemctl start authelia.service
sudo journalctl -u authelia.service -f
```

---

### Caddy 配置：Authelia + PairDrop

你的 Caddyfile 加：

```caddyfile
auth.example.com {
    reverse_proxy 127.0.0.1:9091
}

drop.example.com {
    forward_auth 127.0.0.1:9091 {
        uri /api/authz/forward-auth
        copy_headers Remote-User Remote-Groups Remote-Email Remote-Name
    }

    reverse_proxy 127.0.0.1:3000
}
```

这里 `127.0.0.1:3000` 改成你的 PairDrop 实际监听端口。

然后：

```bash
sudo caddy fmt --overwrite /etc/caddy/Caddyfile
sudo systemctl reload caddy
```

访问 `https://drop.example.com`，应该会先跳到 Authelia 登录。

---

### 给 Opengist 配 OIDC（可选）

创建 Opengist 环境文件：

```bash
sudo install -o root -g root -m 600 /dev/null /etc/opengist.env
sudo nano /etc/opengist.env
```

内容：

```env
OG_EXTERNAL_URL=https://gist.example.com

OG_OIDC_PROVIDER_NAME=Authelia
OG_OIDC_CLIENT_KEY=opengist
OG_OIDC_SECRET=这里填 /opt/authelia/secrets/opengist_oidc_secret_plain 的明文内容
OG_OIDC_DISCOVERY_URL=https://auth.example.com/.well-known/openid-configuration
OG_OIDC_GROUP_CLAIM_NAME=groups
OG_OIDC_ADMIN_GROUP=admins
```

Opengist 官方说明可以用环境变量配置 OIDC，其中包括 provider name、client key、secret、discovery URL、group claim 和 admin group。([Authelia][4])

修改你的 Opengist Quadlet：

```bash
sudo nano /etc/containers/systemd/opengist.container
```

推荐变成：

```ini
[Unit]
Description=Opengist container
After=network-online.target
Wants=network-online.target

[Container]
ContainerName=opengist
Image=ghcr.io/thomiceli/opengist:1
AutoUpdate=registry
EnvironmentFile=/etc/opengist.env
PublishPort=127.0.0.1:6157:6157
PublishPort=2222:2222
Volume=/opt/opengist:/opengist

[Service]
Restart=always

[Install]
WantedBy=multi-user.target
```

重启：

```bash
sudo systemctl daemon-reload
sudo systemctl restart opengist.service
sudo journalctl -u opengist.service -f
```

Caddy：

```caddyfile
gist.example.com {
    reverse_proxy 127.0.0.1:6157
}
```

这里**不要再套 Authelia forward_auth**，因为 Opengist 自己会走 OIDC 登录。否则 OAuth 回调、Git HTTP 操作可能会变复杂。

Opengist 后台里建议打开：

```text
Disable signup
Disable login form
```

Opengist 管理面板支持禁用注册、要求登录、禁用登录表单；禁用登录表单后用户只会看到 OAuth providers。([Opengist][6])

## Headscale

Headscale 官方容器文档说明配置目录挂载到 `/etc/headscale`，数据目录挂载到 `/var/lib/headscale`，容器命令是 `serve`；它也说明容器镜像可用 `docker.io/headscale/headscale:<VERSION>` 或 `ghcr.io/juanfont/headscale:<VERSION>`。([Headscale][7])

创建目录：

```bash
sudo mkdir -p /opt/headscale/config /opt/headscale/lib
```

配置：

```bash
sudo nano /opt/headscale/config/config.yaml
```

最小可用配置：

```yaml
server_url: https://hs.example.com
listen_addr: 127.0.0.1:8070
metrics_listen_addr: 127.0.0.1:9080
grpc_listen_addr: 127.0.0.1:50443
grpc_allow_insecure: false

noise:
  private_key_path: /var/lib/headscale/noise_private.key

prefixes:
  v4: 100.64.0.0/10
  v6: fd7a:115c:a1e0::/48
  allocation: sequential

derp:
  server:
    enabled: true
    region_id: 999
    region_code: "aliyun"
    region_name: "Aliyun DERP"
    ipv4: 你的公网IPv4
    ipv6: 你的公网IPv6
    stun_listen_addr: "[::]:3477"
    private_key_path: /var/lib/headscale/derp_server_private.key

  urls: []

  paths: []
  auto_update_enabled: true
  update_frequency: 3h

database:
  type: sqlite
  sqlite:
    path: /var/lib/headscale/db.sqlite
    write_ahead_log: true

tls_cert_path: ""
tls_key_path: ""

log:
  level: info
  format: text

policy:
  mode: file
  path: ""

dns:
  magic_dns: true
  base_domain: tail.example.com
  override_local_dns: false
  nameservers:
    global: []
  search_domains: []
  extra_records: []

unix_socket: /var/run/headscale/headscale.sock
unix_socket_permission: "0770"

logtail:
  enabled: false

randomize_client_port: false

taildrop:
  enabled: true
```

Headscale 的反代模式里，`server_url` 应该是公网 HTTPS 域名，`listen_addr` 可以监听容器内 `0.0.0.0:8080`，TLS 留给 Caddy 处理时 `tls_cert_path` 和 `tls_key_path` 为空。([Headscale][8])

创建 Quadlet：

```bash
sudo nano /etc/containers/systemd/headscale.container
```

DERP 服务器需要知道真实源 IP, 所以需要设置 `Network=host`，内容：

```ini
[Unit]
Description=Headscale container
After=network-online.target
Wants=network-online.target

[Container]
ContainerName=headscale
Image=docker.io/headscale/headscale:0.28.0
AutoUpdate=registry
Exec=serve
Network=host
Volume=/opt/headscale/config:/etc/headscale:ro
Volume=/opt/headscale/lib:/var/lib/headscale
Tmpfs=/var/run/headscale
HealthCmd=["headscale","health"]
HealthStartPeriod=60s

[Service]
Restart=always

[Install]
WantedBy=multi-user.target
```

启动：

```bash
sudo systemctl daemon-reload
sudo systemctl start headscale.service
sudo journalctl -u headscale.service -f
```

Caddy：

```caddyfile
hs.example.com {
    reverse_proxy 127.0.0.1:8081
}
```

Headscale 反向代理必须支持 WebSocket；不要把 Cloudflare 橙云代理套在 Headscale 前面，Headscale 文档明确说 Cloudflare proxy/tunnel 不支持它需要的 WebSocket POST。([Headscale][8])

---

### Headscale 加设备

先建用户：

```bash
sudo podman exec -it headscale headscale users create richard
sudo podman exec -it headscale headscale users list
```

创建预授权 key，注意新版本通常用 user ID：

```bash
sudo podman exec -it headscale headscale preauthkeys create --user 1 --reusable --expiration 24h
```

客户端加入：

```bash
sudo tailscale up \
  --login-server https://hs.example.com \
  --authkey hskey-auth-xxxx
```

Headscale 的预授权 key 流程就是先创建 user，再创建 preauthkey，然后客户端用 `tailscale up --login-server ... --authkey ...` 注册。([Juan Font][9])

---

### 可选：Headscale 也接 Authelia OIDC

这个可以后面再做。做法是给 Authelia 再加一个 OIDC client：

```yaml
      - client_id: headscale
        client_name: Headscale
        client_secret: "$pbkdf2-sha512$这里换成Headscale client secret hash"
        authorization_policy: one_factor
        redirect_uris:
          - https://hs.example.com/oidc/callback
        scopes:
          - openid
          - email
          - profile
        grant_types:
          - authorization_code
```

然后在 Headscale `config.yaml` 里加：

```yaml
oidc:
  issuer: https://auth.example.com
  client_id: headscale
  client_secret: "这里填 Headscale OIDC 明文 secret"
  expiry: 180d
  scope: ["openid", "profile", "email"]
  email_verified_required: false
```

Headscale OIDC 基本配置需要 issuer URL、client ID、client secret 和 redirect URI，redirect URI 通常是 `https://headscale.example.com/oidc/callback`。 ([Headscale][10])

---

最后检查顺序

```bash
sudo systemctl restart authelia
sudo systemctl restart opengist
sudo systemctl restart headscale
sudo systemctl reload caddy

sudo podman ps
sudo journalctl -u authelia -n 100
sudo journalctl -u opengist -n 100
sudo journalctl -u headscale -n 100
```

建议你先按这个顺序跑通：

```text
Authelia 登录页
→ PairDrop forward_auth
→ Opengist OIDC
→ Headscale 基础 preauthkey
→ Headscale OIDC
```

这样排错最清楚。

[1]: https://www.authelia.com/integration/proxies/caddy/ "Caddy | Integration | Authelia"
[2]: https://raw.githubusercontent.com/authelia/authelia/v4.39.19/config.template.yml "raw.githubusercontent.com"
[3]: https://www.authelia.com/reference/guides/passwords/?utm_source=chatgpt.com "Passwords | Reference"
[4]: https://www.authelia.com/integration/openid-connect/clients/opengist/ "Opengist | OpenID Connect 1.0 | Integration"
[5]: https://www.authelia.com/integration/prologue/get-started/?utm_source=chatgpt.com "Get started | Integration"
[6]: https://opengist.io/docs/configuration/admin-panel.html?utm_source=chatgpt.com "Admin panel"
[7]: https://headscale.net/stable/setup/install/container/ "Container - Headscale"
[8]: https://headscale.net/stable/ref/integration/reverse-proxy/ "Reverse proxy - Headscale"
[9]: https://juanfont.github.io/headscale/development/ref/registration/?utm_source=chatgpt.com "Registration methods"
[10]: https://headscale.net/stable/ref/oidc/ "OpenID Connect - Headscale"

