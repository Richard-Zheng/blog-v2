# Tailscale on OpenWrt

Get tailscale [static binaries](https://tailscale.com/docs/install/linux#static-binaries) and place them under `/usr/sbin`

```
curl -LO https://pkgs.tailscale.com/stable/tailscale_1.96.4_arm64.tgz
tar xvf tailscale_1.96.4_arm64.tgz
cp tailscale_1.96.4_arm64/tailscale /usr/sbin/
cp tailscale_1.96.4_arm64/tailscaled /usr/sbin/
```

## 1. Set Permissions and Create the State Directory

First, ensure both binaries are executable and create the directory where Tailscale will save its authentication state so it survives reboots. Run this in your SSH terminal:

```
chmod +x /usr/sbin/tailscale /usr/sbin/tailscaled
mkdir -p /etc/tailscale
```

## 2. Create the OpenWrt Init Script

OpenWrt uses `procd` to manage services. You need to create an init script for `tailscaled`. Run this entire block of code to create and populate the `/etc/init.d/tailscale` file:

```
cat << 'EOF' > /etc/init.d/tailscale
#!/bin/sh /etc/rc.common

USE_PROCD=1
START=80

start_service() {
    procd_open_instance
    procd_set_param env TS_DEBUG_FIREWALL_MODE=nftables
    procd_set_param command /usr/sbin/tailscaled
    procd_append_param command --state=/etc/tailscale/tailscaled.state
    procd_append_param command --accept-dns=false
    procd_append_param command --port=41641
    procd_set_param respawn
    procd_set_param stdout 1
    procd_set_param stderr 1
    procd_close_instance
}

stop_service() {
    /usr/sbin/tailscaled --cleanup
}
EOF
```

Now, make the init script executable:

```
chmod +x /etc/init.d/tailscale
```

## 3. Enable and Start the Service

Enable the service so it starts automatically on boot, and then start it right now:

```
/etc/init.d/tailscale enable
/etc/init.d/tailscale start
```

Note: You can verify it is running by typing `ps | grep tailscaled`.

## 4. Configure OpenWrt Network and Firewall

For Tailscale to route traffic properly and show up in your LuCI web interface, you need to define the `tailscale0` interface and assign it to a firewall zone.  Run the following `uci` commands to configure the network and firewall automatically:

**Create the network interface:**

```
uci set network.tailscale=interface
uci set network.tailscale.proto='unmanaged'
uci set network.tailscale.device='tailscale0'
uci commit network
/etc/init.d/network restart
```

**Create the firewall zone and forwarding rules:**

```
# Create the zone
uci add firewall zone
uci set firewall.@zone[-1].name='tailscale'
uci set firewall.@zone[-1].input='ACCEPT'
uci set firewall.@zone[-1].forward='ACCEPT'
uci set firewall.@zone[-1].output='ACCEPT'
uci set firewall.@zone[-1].masq='1'
uci set firewall.@zone[-1].device='tailscale0'

# Allow traffic from Tailscale to LAN (Useful if advertising routes)
uci add firewall forwarding
uci set firewall.@forwarding[-1].src='tailscale'
uci set firewall.@forwarding[-1].dest='lan'

# Allow traffic from LAN to Tailscale (Useful if accessing other Tailscale nodes from LAN)
uci add firewall forwarding
uci set firewall.@forwarding[-1].src='lan'
uci set firewall.@forwarding[-1].dest='tailscale'

# Allow WAN inbound to tailscaled UDP 41641
uci add firewall rule
uci set firewall.@rule[-1].name='Allow-Tailscale-WAN'
uci set firewall.@rule[-1].src='wan'
uci set firewall.@rule[-1].proto='udp'
uci set firewall.@rule[-1].dest_port='41641'
uci set firewall.@rule[-1].target='ACCEPT'

uci commit firewall
/etc/init.d/firewall restart
```

## 5. Authenticate and Connect

Now you are ready to bring Tailscale up and authenticate your router.

```
tailscale up
```

This will output a URL. Copy and paste that URL into your web browser to authenticate the router to your Tailnet.

### Headscale

On headscale server side:

create user

```
sudo podman exec -it headscale headscale users create richard
sudo podman exec -it headscale headscale users list
```

create pre-auth key

```
sudo podman exec -it headscale headscale preauthkeys create --user 1 --reusable --expiration 24h
```

On client side:

```
tailscale up \
  --login-server https://hs.example.com \
  --authkey hskey-auth-xxxx
```

### Subnet router

On client side, advertise routes:

```
sudo tailscale up \
  --login-server https://hs.example.com \
  --advertise-routes=192.168.1.0/24
```

or when it's been up:

```
sudo tailscale set --advertise-routes=192.168.1.0/24
```

and allow on server side:

```
headscale nodes list-routes
headscale nodes approve-routes --identifier <Node-ID> --routes 192.168.1.0/24
```

For other clients to accept routes:

```
sudo tailscale set --accept-routes
```
