import httplib


def HashURL(url):
    SEED = "Mining PageRank is AGAINST GOOGLE'S TERMS OF SERVICE. Yes, I'm talking to you, scammer."
    Result = 0x01020345
    for i in range(len(url)):
        Result ^= ord(SEED[i % len(SEED)]) ^ ord(url[i])
        Result = Result >> 23 | Result << 9
        Result &= 0xffffffff
    return '8%x' % Result


def get(url):
    url = url.strip("\a\b\f\n\r\t\v")
    conn = httplib.HTTPConnection('www.google.com')
    googleurl = '/search?client=navclient-auto&features=Rank:&q=info:' \
                + url + '&ch=' + HashURL(url)
    conn.request("GET", googleurl)
    response = conn.getresponse()
    data = response.read()
    status = response.status
    conn.close()
    pr = data.split(":")[-1].strip('\n')
    if len(pr) == 0 or status != 200:
        pr = '-1'
    return pr


PageRank1 = get('www.facebook.com')
print(PageRank1)