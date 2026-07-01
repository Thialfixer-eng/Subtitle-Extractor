import requests, json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Test Baidu sug API (public, no key needed)
r = requests.post('https://fanyi.baidu.com/sug', data={'kw': '你好'}, timeout=10)
data = r.json()
print(json.dumps(data, ensure_ascii=False, indent=2))

print("\n---")

# Test with a longer phrase
r2 = requests.post('https://fanyi.baidu.com/sug', data={'kw': '世界'}, timeout=10)
data2 = r2.json()
print(json.dumps(data2, ensure_ascii=False, indent=2))
