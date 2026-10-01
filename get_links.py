import urllib.request
import re
req = urllib.request.Request('https://html.duckduckgo.com/html/?q=site:indianexpress.com+Mumbai+measles+outbreak+Govandi', headers={'User-Agent': 'Mozilla/5.0'})
html = urllib.request.urlopen(req).read().decode('utf-8')
links = re.findall(r'href="(https://indianexpress.com/article/[^"]+)"', html)
print(list(set(links)))
