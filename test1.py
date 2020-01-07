from tldextract import extract
import ssl
import socket
import regex as re
import whois
from bs4 import BeautifulSoup
import urllib.request
import datetime
import requests
import sys
import numpy as np
import pandas as pd
import time


def SSLfinal_State(u):
    subdomain1,domain1,suffix2=extract(u)
    websitedomain=domain1
    opener = urllib.request.urlopen(u).read()
    soup = BeautifulSoup(opener, 'lxml')
    imgs = soup.findAll('img', src=True)
    link_to_same=0
    for image in imgs:
        url=image['src']
        subdomain2,domain2,suffix2=extract(url)
        if "https" in url:
            whoisurl=whois.whois("https://"+domain2+"."+suffix2)
            time.sleep(3)
            listtostr=''.join(whoisurl['name_servers'])
            # print(listtostr)
            # print(websitedomain.upper())
            if websitedomain.upper() in listtostr:
                print("enter")
                link_to_same=link_to_same+1

    return(link_to_same)

url1=input("ENter the url:")
p=SSLfinal_State(url1)
print(p)
