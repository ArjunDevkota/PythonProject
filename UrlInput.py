import regex as re

def usingIpAddress(url):
    symbol = re.findall(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', url)
    if (len(symbol) != 0):
        having_ip = 1  # phishing
    else:
        having_ip = -1  # legitimate

    return (having_ip)

def longURL(url):
    if(len(url)>54):
        return 1    #phishing
    elif(54<len(url) and len(url)<75):
        return 0    #suspicious
    else:
        return -1   #legimate

def tinyURL(url):
    symbol = regex.findall(r'((bit.ly)/([\w]+))', url)
    if (len(symbol) != 0):
        return 1  # phishing
    elif:
        return -1  # legitimate
    else:
        return 0   #suspicious


def atTheRateSymbol(url):
    symbol = regex.findall(r'@', url)
    if (len(symbol) == 0):
        return -1
    else:
        return 1


def doubleSlash(url):


