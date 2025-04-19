import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin

# 设置URL
def get_user_input_url():
    # 获取用户输入的网址，默认为百度首页
    url = input("请输入要爬取的网址(默认:https://www.baidu.com):").strip()
    if not url:
        url = 'https://www.baidu.com'
    elif not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    return url

def fetch_html(url):
    # 设置请求头
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
    }

    try:
        # 发送HTTP请求
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # 检查请求是否成功
        response.encoding = response.apparent_encoding  # 自动检测编码
    
        # 获取HTML内容
        html = response.text
    
        # 使用BeautifulSoup解析
        soup = BeautifulSoup(html, 'lxml')
        title = soup.title.string.strip() if soup.title and soup.title.string else 'None'
        print(f"网页标题: {title}")

        # 创建用于保存图片的目录
        images_dir = 'images'
        os.makedirs(images_dir, exist_ok=True)

        # 处理所有的img标签
        for img in soup.find_all('img'):
            img_url = img.get('src') or img.get('data-src')
            if not img_url:
                continue

            # 构建完整的图片URL
            full_img_url = urljoin(url, img_url)

            try:
                img_response = requests.get(full_img_url, headers=headers, timeout=10)
                img_response.raise_for_status()
                img_data = img_response.content

                # 从URL中提取图片文件名
                parsed_url = urlparse(full_img_url)
                img_filename = os.path.basename(parsed_url.path)
                if not img_filename:
                    continue  # 如果无法提取文件名，跳过该图片

                # 保存图片到本地
                img_path = os.path.join(images_dir, img_filename)
                with open(img_path, 'wb') as f:
                    f.write(img_data)

                # 更新HTML中的图片链接为本地路径
                img['src'] = os.path.join(images_dir, img_filename)
                

            except requests.exceptions.RequestException:
                continue  # 如果下载图片失败，跳过该图片
    
        # 保存HTML到文件
        try:
            with open(f"{url.split('/')[-1]}.html", 'w', encoding='utf-8') as f:
                f.write(str(soup))
            print(f"HTML已保存到 {url.split('/')[-1]}.html")
        except IOError as e:
            print(f"文件保存失败: {e.strerror}")


    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")

if __name__ == "__main__":
    url = get_user_input_url()
    fetch_html(url)