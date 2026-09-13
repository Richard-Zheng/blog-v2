# immortalwrt-mt798x 编译

首先创建一个 systemd-nspawn 容器，系统选用 Ubuntu 22.04 LTS

```sh
sudo debootstrap --arch=amd64 jammy /var/lib/machines/imm22 https://mirrors.sustech.edu.cn/ubuntu/
```

进入容器

```sh
sudo systemd-nspawn -D /var/lib/machines/imm22 --resolv-conf=bind-host
```

配置软件源

```sh
printf '%s' '
deb https://mirrors.sustech.edu.cn/ubuntu/ jammy main restricted universe multiverse
deb https://mirrors.sustech.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse
deb https://mirrors.sustech.edu.cn/ubuntu/ jammy-backports main restricted universe multiverse

deb http://security.ubuntu.com/ubuntu/ jammy-security main restricted universe multiverse
' | sudo tee /etc/apt/sources.list
```

安装依赖，参考[官方文档](https://openwrt.org/docs/guide-developer/toolchain/install-buildsystem#set_for_ubuntu_2204_that_has_older_python_3xx)

```sh
sudo apt update
sudo apt install build-essential clang flex bison g++ gawk \
  gcc-multilib g++-multilib gettext git libncurses-dev libssl-dev \
  python3-distutils python3-setuptools rsync swig unzip zlib1g-dev file wget
```

创建一个用户

```sh
useradd -m -G sudo -s /bin/bash frain
passwd frain
```

切换用户

```sh
su frain
```

拉取仓库

```sh
cd ~
git clone https://github.com/hanwckf/immortalwrt-mt798x.git --depth 1
cd immortalwrt-mt798x/
```

准备 .config 编译选项，根据机型选择

```sh
cp defconfig/mt7986-ax6000.config .config
```

拉取 feeds

```sh
./scripts/feeds update -a
./scripts/feeds install -a
```

编译 menuconfig 并配置最终编译选项

```sh
make menuconfig
```

下载依赖的源码包，`V=s` 开启详细日志输出。

```sh
make download V=s
```

最后编译

```sh
make -j$(nproc)
```
