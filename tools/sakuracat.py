import requests

# 1. 配置SOCKS5代理（端口核对你的SakuraCat设置）
proxies = {
    'http': 'socks5://127.0.0.1:7897',
    'https': 'socks5://127.0.0.1:7897'
}

# 2. 设置合法的User-Agent（关键！避免被反爬）
# 可以用浏览器的UA，或注明你的用途（符合维基数据的机器人规则）
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
}

try:
    # 3. 发送请求（添加headers和proxies）
    # 维基数据API请求示例（带合法UA和代理）
    response = requests.get(
        'https://www.wikidata.org/w/api.php?action=query&format=json&titles=Q42',  # 查询Q42（Douglas Adams）
        proxies=proxies,
        headers=headers,
        timeout=10
    )
    if response.status_code == 200:
        print("API返回数据：", response.json())

    print(f"请求成功，状态码：{response.status_code}")
    print(f"验证信息：{response.text[:200]}")  # 打印前200个字符确认
except requests.exceptions.ProxyError as e:
    print(f"代理连接失败：{e}")
    print("请检查：1.SakuraCat是否已连接有效节点 2.端口号是否正确（7897）")
except Exception as e:
    print(f"其他错误：{e}")