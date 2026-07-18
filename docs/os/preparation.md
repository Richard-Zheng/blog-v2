# NJU OS Demo Downloader

南京大学 [jyy](https://jyywiki.cn/) 老师的操作系统课程内容非常丰富和新颖，能看出花了大量心血准备，在这里向 jyy 老师致敬。

jyy 课程的核心其实在他的 demos 里，上课的时候要手边开一个 demos, 对照着看代码和运行、调试，否则只是走马观花。

demos 的 URL:  https://jyywiki.cn/OS/demos/

在浏览器里看 demos 很难受，所以我(用 AI)写了个 Python 脚本一键批量把 demos 拉下来：

```python
import os
import sys
from urllib.parse import urljoin, unquote
import requests
from bs4 import BeautifulSoup

# 配置目标 URL 和本地保存路径
TARGET_URL = "https://jyywiki.cn/OS/demos/"
SAVE_DIR = "./os_demos"

def download_file(url, save_path):
    """下载单个文件，支持大文件流式下载"""
    # 避免重复下载已存在的文件
    if os.path.exists(save_path):
        print(f"[已存在] 跳过: {save_path}")
        return

    print(f"[下载中] {url} -> {save_path}")
    try:
        with requests.get(url, stream=True, timeout=15) as r:
            r.raise_for_status()
            # 自动创建本地父级目录
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except Exception as e:
        print(f"[错误] 下载失败 {url}: {e}")

def crawl_and_download(current_url, current_dir):
    """递归爬取 Nginx autoindex 目录并下载"""
    try:
        response = requests.get(current_url, timeout=10)
        response.raise_for_status()
        response.encoding = 'utf-8'
    except Exception as e:
        print(f"[错误] 无法访问目录 {current_url}: {e}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    
    # 遍历页面上所有的超链接
    for link in soup.find_all('a'):
        href = link.get('href')
        
        # 排除无效链接、上级目录跳转
        if not href or href in ['../', './', '/']:
            continue
        # 排除绝对路径或外部链接
        if href.startswith(('http://', 'https://', '/')):
            continue

        # 解码 URL 中的中文字符或特殊符号（例如 %20 变空格）
        decoded_name = unquote(href)
        full_url = urljoin(current_url, href)
        local_path = os.path.join(current_dir, decoded_name)

        # Nginx 目录的特点是 href 以 '/' 结尾
        if href.endswith('/'):
            # 如果是目录，递归进入下载
            print(f"[发现目录] 进入: {decoded_name}")
            crawl_and_download(full_url, local_path)
        else:
            # 如果是文件，直接下载
            download_file(full_url, local_path)

if __name__ == "__main__":
    print(f"开始从 {TARGET_URL} 批量下载文件...")
    crawl_and_download(TARGET_URL, SAVE_DIR)
    print("下载任务完成！")
```