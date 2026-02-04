---
slug: colab-huggingface-download
title: Colab 下载 huggingface 模型
tags: ['AI']
---

利用Google Colab的高速网络和百度网盘，实现huggingface模型的下载和保存。

<!-- truncate -->

大模型一个比一个大，动辄几十上百GB的体积让下模型变成了一个很痛苦的事。huggingface自带的cli下载器也并没有帮上什么忙，不支持文件分块级别的断点续传不说，还会越下越慢，下着下着直接没速度，疑似为服务端的rate-limiting. 所以挂代理下载不但费流量，而且还会被huggingface限流，用公开的镜像站除了没有流量费用以外照样有限流问题。

就在这时我发现Google Colab的下载速度飞快，A100的机子连接huggingface下载速度甚至能随便跑到5Gbps! 可是直接用的话有点费compute unit, 要调试还得用野路子开VSCode server, 而且每次打开都得重新下载一次模型。于是想到可以把模型下到网盘里，然后再从网盘下到我本地。用Google Drive的话就很简单，挂载一下drive然后直接复制进去就好了。但是我不想费这么多流量。于是目光转向百度网盘。

注意：Colab选机器配置尽量不要选纯CPU的机器，性能弱鸡网络速度也不行。

百度网盘有官方Linux版，但是需要X11 server才能运行，而且我也需要有个图形界面操作。参考网上的docker镜像脚本[[1]](https://github.com/KevinLADLee/baidunetdisk-docker) [[2]](https://github.com/gshang2017/docker/tree/master/baidunetdisk)安装X11环境以及noVNC. 你要问为什么不直接用docker？请看Colab的[TOS](https://research.google.com/colaboratory/faq.html#disallowed-activities)最后一行

> The following are disallowed from all managed Colab runtimes:
>
> - employing techniques such as containerization to circumvent anti-abuse policies.

```
!export VERSION=4.14.5
!export URI=https://issuepcdn.baidupcs.com/issue/netdisk/LinuxGuanjia/$VERSION/baidunetdisk_${VERSION}_amd64.deb
!apt-get update \
    && apt-get install -y --no-install-recommends wget curl  \
                          ca-certificates \
                          desktop-file-utils    \
                          libasound2-dev        \
                          locales               \
                          fonts-wqy-zenhei      \
                          libgtk-3-0            \
                          libnotify4            \
                          libnss3               \
                          libxss1               \
                          libxtst6              \
                          xdg-utils             \
                          libatspi2.0-0         \
                          libuuid1              \
                          libappindicator3-1    \
                          libsecret-1-0
!curl -L https://issuepcdn.baidupcs.com/issue/netdisk/LinuxGuanjia/4.14.5/baidunetdisk_4.14.5_amd64.deb -o /content/baidunetdisk.deb     \
    && apt-get install -y /content/baidunetdisk.deb \
    && rm /content/baidunetdisk.deb
!apt-get install -y xvfb x11-utils x11-apps x11vnc novnc websockify
```

然后开启一个虚拟的X framebuffer，分辨率不能太低要不然file picker会超出屏幕范围点不到按钮。建议用Terminal里的tmux运行

```
Xvfb :99 -screen 0 1920x1080x24
```

然后运行百度网盘，也是开个tmux窗口运行

```
export DISPLAY=:99
/opt/baidunetdisk/baidunetdisk --no-sandbox
```

再开个vnc server，也是开个tmux窗口运行

```
x11vnc -display :99 -nopw -listen localhost -xkb -forever
```

开noVNC web服务，也是开个tmux窗口运行

```
websockify -D --web=/usr/share/novnc/ 6080 localhost:5900
```

Google Colab自带一个网页反代服务，很方便

```python
from google.colab.output import eval_js
url = eval_js("google.colab.kernel.proxyPort(6080)")
print('Open the link to use! -> ', url)
```

打开链接点开vnc.html，然后连接即可看到百度网盘的界面。剩下的事情就都很简单了。

比如说我们想下载FLUX.2-dev，可以这样下载。

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="black-forest-labs/FLUX.2-dev",
    local_dir="/content/FLUX2dev",
    token="你的token, 要去模型主页点同意条款",
    # flux2-dev.safetensors似乎就是transformers的几个模型文件合并在一起了，可以不用下载。
    ignore_patterns=["flux2-dev.safetensors"]
)
```