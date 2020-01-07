import regex as re
from tldextract import extract
import whois
import ssl
import socket
import urllib.request
from bs4 import BeautifulSoup
import requests
import datetime
def having_IP(u):
        symbol = re.findall(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', u)
        if (len(symbol) != 0):
            return ("ip--Phishing::'\n'IP Address on URL")
        else:
            return ("ip--Not-Phishing::'\n'No IP Address on URL")

def URL_Length(u):
        if (len(u) < 54):
            return ("longurl--Not-Phishing::'\n'URL length is "+str(len(u))+" which is less than 54")
        elif (len(u) >= 54 and len(u) <= 75):
            return ("longurl--Suspecious::'\n'URL length is "+str(len(u))+" which is between 54 and 75")
        else:
            return ("longrul--Phishing::'\n'URL lenght is "+str(len(u))+" which is greater than 75")

def Shortining_Service(u):
        match = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                          'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                          'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                          'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                          'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                          'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                          'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|tr\.im|link\.zip\.net',
                          u)
        if match:
            return ("Shortining_Service--Phishing::'\n'Shortining")
        else:
            return ("Shortining_Service--Not-Phishing::'\n'No Shortining")

def having_At_Symbol(u):
        symbol = re.findall(r"@", u)
        if (len(symbol) != 0):
            return ("having_At_Symbol--Phishing::'\n'@ symbol in URL")
        else:
            return ("having_At_Symbol--Not-Phishing::'\n'No @ symbol in URL")

def double_slash_redirecting(u):
        symbol = re.findall(r"//", u)
        if (len(symbol) >= 2):
            return ("double_slash_redirecting--Phishing::'\n'// i.e Redirecting symbol in URL")
        else:
            return ("double_slash_redirecting--Not-Phishing::'\n'No // i.e Redirecting symbol in URL")


def Prefix_Suffix(u):
        subdomain, domain, suffix = extract(u)
        if (domain.count("-")):
            return ("Prefix_Suffix--Phishing::'\n'Domain part is "+domain+" which contain - symbol")
        else:
            return ("Prefix_Suffix--Not-Phishing::'\n'Domain part is "+domain+" which not contain - symbol")

def having_Sub_Domain(u):
        subdomain, domain, suffix = extract(u)
        if (subdomain.count(".") == 0):
            return ("having_Sub_Domain--Not-Phishing::'\n'Subdomain is "+subdomain+" which contain zero . symbol")
        elif (subdomain.count(".") == 1):
            return ("having_Sub_Domain--Suspecious::'\n'Subdomain is "+subdomain+" which contain one . symbol")
        else:
            return ("having_Sub_Domain--Phishing::'\n'Subdomain is "+subdomain+" which contain more . symbol")

def SSLfinal_State(url):
    try:
        # check wheather contains https
        try:
            subDomain, domain, suffix = extract(url)
            host_name = domain + "." + suffix
            context = ssl.create_default_context()
            sct = context.wrap_socket(socket.socket(), server_hostname=host_name)
            sct.connect((host_name, 443))
            usehttps1=1
        except:
            usehttps2=2
        # getting the certificate issuer to later compare with trusted issuer
        # getting host name
        subDomain, domain, suffix = extract(url)
        host_name = domain + "." + suffix
        context = ssl.create_default_context()
        sct = context.wrap_socket(socket.socket(), server_hostname=host_name)
        sct.connect((host_name, 443))
        certificate = sct.getpeercert()
        issuer = dict(x[0] for x in certificate['issuer'])
        certificate_Auth = str(issuer['commonName'])
        certificate_Auth = certificate_Auth.split()
        if (certificate_Auth[0] == "Network" or certificate_Auth == "Deutsche"):
            certificate_Auth = certificate_Auth[0] + " " + certificate_Auth[1]
        else:
            certificate_Auth = certificate_Auth[0]

        trusted_Auth = ['Comodo', 'Symantec', 'GoDaddy', 'GlobalSign', 'DigiCert', 'StartCom', 'Entrust', 'Verizon',
                        'Trustwave', 'Unizeto', 'Buypass', 'QuoVadis', 'Deutsche Telekom', 'Network Solutions',
                        'SwissSign', 'IdenTrust', 'Secom', 'TWCA', 'GeoTrust', 'Thawte', 'Doster', 'VeriSign', 'Google']
        if (certificate_Auth=="Google" or certificate_Auth=="DigiCert"):
            Age_of_certificate=2
        else:
            # getting age of certificate
            startingDate = str(certificate['notBefore'])
            endingDate = str(certificate['notAfter'])
            startingYear = int(startingDate.split()[3])
            endingYear = int(endingDate.split()[3])
            Age_of_certificate = (endingYear - startingYear)
        # checking final conditions
        if ((usehttps1 == 1)  and (certificate_Auth in trusted_Auth) and (Age_of_certificate >= 1)):
            return ("SSLfinal_State--Not-Phishing::'\n'Is https://,certificate authorization by "+certificate_Auth+" and Age of certificate is more than one year")
        elif ((usehttps1 == 1) and (certificate_Auth in trusted_Auth)):
            return ("SSLfinal_State--Suspecious::'\n'Is https:// and certificate authorization by "+certificate_Auth)
        else:
            return ("SSLfinal_State--Phishing::'\n'either http:// or certificate authorization by "+certificate_Auth+" is not trusted")
    except:
        return("SSLfinal_State--Phishing Except Part:'\n':Invalid URL")

def domain_registration(u):
  try:
        try:
            w = whois.whois(u)
            updated = w["updated_date"]
            exp = w["expiration_date"]
            length = (exp - updated).days
            if (length >= 365):
                return ("domain_registration--Not-Phishing::'\n'Updated Date "+str(updated)+" and expiration Date "+str(exp)+" length is more than one year")
            else:
                return ("domain_registration--Phishing::'\n'Updated Date "+str(updated)+" and expiration Date "+str(exp)+" length is less than one year")

        except:
            w1=whois.whois(u)
            updated = w1.updated_date
            exp = w1.expiration_date
            length = (exp - updated).days
            if (length >= 365):
                return ("domain_registration--Not-Phishing::'\n'Updated Date "+str(updated)+" and expiration Date "+str(exp)+" length is more than one year")
            else:
                return ("domain_registration--Phishing::'\n'Updated Date "+str(updated)+" and expiration Date "+str(exp)+" length is less than one year")
  except:
      return("domain_registration--Phishing Except Part::'\n'Invalid URL")

def Favicon(u):
    try:
        subDomain, domain, suffix = extract(u)
        websitedomain = domain
        v1 = 0
        try:
            host_name = domain + "." + suffix
            context = ssl.create_default_context()
            sct = context.wrap_socket(socket.socket(), server_hostname=host_name)
            sct.connect((host_name, 443))
            url1 = "https://" + host_name
        except:
            url1 = "http://" + domain + "." + suffix

        opener = urllib.request.urlopen(url1).read()
        soup = BeautifulSoup(opener, 'lxml')
        favicon = soup.find_all('link', href=True)
        for item in soup.find_all('link'):
            p = item.get('href')
            if ".png" in p:
                subDomain, domain, suffix = extract(p)
                if (websitedomain != domain):
                    v1 = v1 + 1

        if v1 == 0:
            return ("Favicon--Not-Phishing'\n'Favicon doesnot load from other URL")
        if favicon == []:
            return ("Favicon--Not-Phishing'\n'Favicon doesnot load from other URL")
        else:
            return ("Favicon--Phishing'\n'Favicon load from other URL")
    except:
        return ("Favicon--Phishing'\n'Invalid URL")

def port(u):
    try:
        subDomain, domain, suffix = extract(u)
        url1=''
        try:
            host_name = domain + "." + suffix
            context = ssl.create_default_context()
            sct = context.wrap_socket(socket.socket(), server_hostname=host_name)
            sct.connect((host_name, 443))
            url1 ='a'
        except:
            url1='a'
        if url1=="a":
            return("Port--Not-Phishing'\n'Open port 80 or 443")
        else:
            return("Port--Phishing'\n'port neither 80 nor 443")
    except:
        return("Port--Phishing'\n'Invalid URL")



def https_token(u):
    try:
        subdomain, domain, suffix = extract(u)
        if (subdomain.count('http') or subdomain.count("https")):
            return ("https_token--Phishing::'\n'http or https token in subdomain part")
        elif (domain.count('http') or domain.count("https")):
            return ("https_token--Phishing::'\n'http or https token in domain part ")
        else:
            return ("https_token--Not-Phishing::'\n'No http or https token in domain or subdomain part")

    except:
        return ("https_token--Phishing Except::'\n'Invalid URL")


def request_url(u):
    try:
        subDomain, domain, suffix = extract(u)
        try:

            host_name = domain + "." + suffix
            context = ssl.create_default_context()
            sct = context.wrap_socket(socket.socket(), server_hostname=host_name)
            sct.connect((host_name, 443))
            url1 = "https://" + host_name
        except:
            url1="http://"+domain+"."+suffix

        websiteDomain = domain
        opener = urllib.request.urlopen(url1).read()
        soup = BeautifulSoup(opener, 'lxml')
        imgs = soup.findAll('img', src=True)
        total = len(imgs)
        linked_to_same = 0
        avg = 0
        for image in imgs:
            subDomain, domain, suffix = extract(image['src'])
            imageDomain = domain
            if (websiteDomain == imageDomain or imageDomain == '' or "cdn" in imageDomain or "img" in imageDomain):
                linked_to_same = linked_to_same + 1
        vids = soup.findAll('video', src=True)
        total = total + len(vids)

        for video in vids:
            subDomain, domain, suffix = extract(video['src'])
            vidDomain = domain
            if (websiteDomain == vidDomain or vidDomain == ''):
                linked_to_same = linked_to_same + 1
        linked_outside = total - linked_to_same
        if (total != 0):
            avg = linked_outside / total

        if (avg < 0.22):
            return ("Request URL--Not-Phishing::'\n'Total Link "+str(total)+" link to outside "+str(linked_outside)+" link to same "+str(linked_to_same)
                    +" avg<0.22")
        elif (0.22 <= avg <= 0.61):
            return ("Request URL--Suspecious::'\n'Total Link "+str(total)+" link to outside "+str(linked_outside)+" link to same "+str(linked_to_same)
                    +" 0.22<avg<0.61")
        else:
            return ("Request URL--Phishing try::'\n'Total Link "+str(total)+" link to outside "+str(linked_outside)+" link to same "+str(linked_to_same)
                    +" avg>0.61")
    except:
        return("Request URL--Phishing excpet::'\n'Invalid URL")

def anchor_of_url(u):
    try:
        subDomain, domain, suffix = extract(u)
        try:

            host_name = domain + "." + suffix
            context = ssl.create_default_context()
            sct = context.wrap_socket(socket.socket(), server_hostname=host_name)
            sct.connect((host_name, 443))
            url1 = "https://" + host_name
        except:
            url1 = "http://" + domain + "." + suffix

        websiteDomain = domain
        opener = urllib.request.urlopen(url1).read()
        soup = BeautifulSoup(opener, 'lxml')
        anchors = soup.findAll('a', href=True)
        total = len(anchors)
        linked_to_same = 0
        avg = 0
        for anchor in anchors:
            subDomain, domain, suffix = extract(anchor['href'])
            anchorDomain = domain
            if (websiteDomain == anchorDomain or anchorDomain == ''):
                linked_to_same = linked_to_same + 1
        linked_outside = total - linked_to_same
        if (total != 0):
            avg = linked_outside / total

        if (avg < 0.31):
            return ("Anchor of URL--Not-phishing'\n'")
        elif (0.31 <= avg <= 0.67):
            return ("Anchor of URL---suspecious'\n'")
        else:
            return ("Anchor of URL--Phishing try'\n'")
    except:
        return ("Anchor of URL--Phishing except'\n'")

def link_in_tag(u):
    try:
        subDomain, domain, suffix = extract(u)
        try:

            host_name = domain + "." + suffix
            context = ssl.create_default_context()
            sct = context.wrap_socket(socket.socket(), server_hostname=host_name)
            sct.connect((host_name, 443))
            url1 = "https://" + host_name
        except:
            url1 = "http://" + domain + "." + suffix
        opener = urllib.request.urlopen(url1).read()
        soup = BeautifulSoup(opener, 'lxml')
        no_of_meta = 0
        no_of_link = 0
        no_of_script = 0
        anchors = 0
        avg = 0
        meta = soup.find_all('meta', content=True)
        for meta1 in meta:
            no_of_meta = no_of_meta + 1

        link = soup.find_all('link', href=True)
        for link1 in link:
            no_of_link = no_of_link + 1

        script = soup.find_all('script', src=True)
        for script1 in script:
            no_of_script = no_of_script + 1

        anchors1 = soup.find_all('a', href=True)
        for anchor in anchors1:
            anchors = anchors + 1

        total = no_of_meta + no_of_link + no_of_script + anchors
        tags = no_of_meta + no_of_link + no_of_script
        if (total != 0):
            avg = tags / total

        if (avg < 0.25):
            return ("link in tag--Not-Phishing'\n'")
        elif (0.25 <= avg <= 0.81):
            return ("link in tag--Suspecious'\n'")
        else:
            return ("Link in tag--Phishing except'\n'")
    except:
        return ("Link in tag--Phishing except'\n'")


def server_form_handler(u):
    try:
        subDomain, domain, suffix = extract(u)
        websitedomain=domain
        try:

            host_name = domain + "." + suffix
            context = ssl.create_default_context()
            sct = context.wrap_socket(socket.socket(), server_hostname=host_name)
            sct.connect((host_name, 443))
            url1 = "https://" + host_name
        except:
            url1 = "http://" + domain + "." + suffix
        opener = urllib.request.urlopen(url1).read()
        soup = BeautifulSoup(opener, 'lxml')
        sfh=soup.find('form').get('action')
        subDomain, domain, suffix = extract(sfh)
        sfhdomain = domain
        if (websitedomain == sfhdomain or sfh=="/search"):
            return ("server form handler--Not-Phishing::'\n'sfhdomain is "+sfhdomain+" = "+websitedomain+" or "+sfh+" which is search engine")
        elif (sfhdomain == ''):
            return ("server form handler--Suspecious try::'\n'sfhdomain "+sfhdomain+" <- is empty")
        else:
            return ("server form handler--Phishing::'\n'sfhdomain "+sfhdomain+" != "+websitedomain)
    except:
        return ("server form handler--Suspecious except::'\n'URL is invalid or doesnot contain SFH")


def email_submit(u):
    try:
        subDomain, domain, suffix = extract(u)
        try:

            host_name = domain + "." + suffix
            context = ssl.create_default_context()
            sct = context.wrap_socket(socket.socket(), server_hostname=host_name)
            sct.connect((host_name, 443))
            url1 = "https://" + host_name
        except:
            url1 = "http://" + domain + "." + suffix
        opener = urllib.request.urlopen(url1).read()
        soup = BeautifulSoup(opener, 'lxml')
        if (soup.find('mailto:') or soup.find('mail')):
            return ("Email Submit--Phishing try::'\n'mail information")
        else:
            return ("Email Submit--Not-Phishing::'\n'doesnot mail information")
    except:
        return ("Email Submiit--Phishing except::'\n'Invalid URL")

def abnormal_url(u):
    try:
        subDomain,domain,suffix=extract(u)
        u=whois.whois(u)
        urlhost=u["domain_name"]
        enteredurl=domain+'.'+suffix
        if enteredurl.upper() in urlhost:
            return("Abnormal URL--Not-Phishing::'\n'entered URL matches Whois database domain name")
        else:
            return("Abnormal URL--Phishing try::'\n'entered URL doesnot matches Whois database domain name")
    except:
        return("Abnormal URL--Phishing except::'\n'Invalid URL")

def website_forwarding(u):
        subDomain, domain, suffix = extract(u)
        try:

            host_name = domain + "." + suffix
            context = ssl.create_default_context()
            sct = context.wrap_socket(socket.socket(), server_hostname=host_name)
            sct.connect((host_name, 443))
            url1 = "https://" + host_name
        except:
            url1 = "http://" + domain + "." + suffix
        r=requests.get(url1)
        list=r.history
        i=len(list)
        if i<=2:
            return("Website Forwading--Not-Phishing::'\n'URL redirected less than 1 times")
        elif (2<i<=4):
            return("Website Forwading--Suspecious::'\n'URL redirected more than 1 less than 4  times")
        else:
            return("Website Forwading--Phishing try::'\n'URL redirected more than 4 times")

def status_bar_customization(u):
    try:
        subDomain, domain, suffix = extract(u)
        try:

            host_name = domain + "." + suffix
            context = ssl.create_default_context()
            sct = context.wrap_socket(socket.socket(), server_hostname=host_name)
            sct.connect((host_name, 443))
            url1 = "https://" + host_name
        except:
            url1 = "http://" + domain + "." + suffix

        opener = urllib.request.urlopen(url1).read()
        soup = BeautifulSoup(opener, 'lxml')
        r=soup.find_all('onmouseover')
        if len(r)==0:
            return("status bar customization--Not-Phishing::'\n'NO onmouseover")
        else:
            return ("status bar customization--Phishing try::'\n'Onmouseover")
    except:
        return ("status bar customization--Phishing except::'\n'invalid url")
def disable_right_click(u):
    try:
        subDomain, domain, suffix = extract(u)
        try:

            host_name = domain + "." + suffix
            context = ssl.create_default_context()
            sct = context.wrap_socket(socket.socket(), server_hostname=host_name)
            sct.connect((host_name, 443))
            url1 = "https://" + host_name
        except:
            url1 = "http://" + domain + "." + suffix

        opener = urllib.request.urlopen(url1).read()
        soup = BeautifulSoup(opener, 'lxml')
        r = soup.findAll('event.button==2')
        if len(r)==0:
            return("disabling_right_click--Not-Phishing::'\n'no event.button==2")
        else:
            return ("disabling_right_click--Phishing try::'\n'event.button==2")
    except:
        return ("disabling_right_click--Phishing except::'\n'invalid URL")


def age_of_domain(u):
    try:
        try:
            w = whois.whois(u)
            creationDate = w.creation_date
            currentDate = datetime.datetime.now()
            length = (currentDate - creationDate[0]).days
            if length >= 180:
                return ("Age of Domain--Not-Phishing::'\n'creation Date is "+str(creationDate)+" current date is "+str(currentDate)+" age is more than 6 month")
            else:
                return ("Age of Domain--Phishing try::'\n'creation Date is "+str(creationDate)+" current date is "+str(currentDate)+" age is less than 6 month")

        except:
            w = whois.whois(u)
            creationDate = w["creation_date"]
            currentDate = datetime.datetime.now()
            length = (currentDate - creationDate).days
            if length >= 180:
                return ("Age of Domain--Not-Phishing::'\n'creation Date is "+str(creationDate)+" current date is "+str(currentDate)+" age is more than 6 month")
            else:
                return ("Age of Domain--Phishing try::'\n'creation Date is "+str(creationDate)+" current date is "+str(currentDate)+" age is less than 6 month")
    except:
        return("Age of Domain--Phishing except::Invalid URL")

def dns_record(u):
        try:
            w=whois.whois(u)
            if(w["domain_name"]==None):
                return("DNS Record--Phishing::'\n'Empty or None Domain name in Whois database")
            else:
                return("DNS Record--Not-Phishing::'\n'Domain Name in Whois database")
        except:
            return("DNS Record--Phishing except::'\n'Invalid URL")

def website_rank(u):
    try:
        p =BeautifulSoup(urllib.request.urlopen("http://data.alexa.com/data?cli=10&dat=s&url=" + u).read(), "xml").find(
            "REACH")['RANK']
        if int(p) < 100000:
            return ("Website Traffic--Not-Phishing::'\n'Website Rank is "+p+" which is <100000")
        elif int(p) > 100000:
            return ("Website Traffic--Suspecious::'\n'Website Rank is "+p+" which is > 100000")
        else:
            return ("Website Traffic--Phishing try::'\n'Website Rank is "+p)
    except:
        return ("Website Traffic--Phishing except::'\n'Invalid URL or Website Rank is None")

def statistical_report(u):
    var1 = ''
    if "https://" in u:
        var1 = u[8:]
    elif "http://" in u:
        var1 = u[7:]
    else:
        var1 = u
    try:
        ip_address = socket.gethostbyname(var1)
    except:
        return ("Statistical Report-- phishing")
    url_match = re.search(
        r'at\.ua|usa\.cc|baltazarpresentes\.com\.br|pe\.hu|esy\.es|hol\.es|sweddy\.com|myjino\.ru|96\.lt|ow\.ly', u)
    ip_match = re.search(
        '146\.112\.61\.108|213\.174\.157\.151|121\.50\.168\.88|192\.185\.217\.116|78\.46\.211\.158|181\.174\.165\.13|46\.242\.145\.103|121\.50\.168\.40|83\.125\.22\.219|46\.242\.145\.98|'
        '107\.151\.148\.44|107\.151\.148\.107|64\.70\.19\.203|199\.184\.144\.27|107\.151\.148\.108|107\.151\.148\.109|119\.28\.52\.61|54\.83\.43\.69|52\.69\.166\.231|216\.58\.192\.225|'
        '118\.184\.25\.86|67\.208\.74\.71|23\.253\.126\.58|104\.239\.157\.210|175\.126\.123\.219|141\.8\.224\.221|10\.10\.10\.10|43\.229\.108\.32|103\.232\.215\.140|69\.172\.201\.153|'
        '216\.218\.185\.162|54\.225\.104\.146|103\.243\.24\.98|199\.59\.243\.120|31\.170\.160\.61|213\.19\.128\.77|62\.113\.226\.131|208\.100\.26\.234|195\.16\.127\.102|195\.16\.127\.157|'
        '34\.196\.13\.28|103\.224\.212\.222|172\.217\.4\.225|54\.72\.9\.51|192\.64\.147\.141|198\.200\.56\.183|23\.253\.164\.103|52\.48\.191\.26|52\.214\.197\.72|87\.98\.255\.18|209\.99\.17\.27|'
        '216\.38\.62\.18|104\.130\.124\.96|47\.89\.58\.141|78\.46\.211\.158|54\.86\.225\.156|54\.82\.156\.19|37\.157\.192\.102|204\.11\.56\.48|110\.34\.231\.42',
        ip_address)
    if url_match:
        return ("Statistical Report-- Phishing::'\n'Hostname belong to top 10 phishing domain")
    elif ip_match:
        return ("Statistical Report-- Phishing::'\n'Hostname belong to top 10 phishing ip")
    else:
        return ("Statistical Report-- Not-Phishing::'\n'Hostname doesnot belong to 10 phishing domain or ip")


def uiFile(u):
    check=[having_IP(u),URL_Length(u),Shortining_Service(u),having_At_Symbol(u),double_slash_redirecting(u),
            Prefix_Suffix(u),having_Sub_Domain(u),SSLfinal_State(u),domain_registration(u),Favicon(u),port(u),https_token(u),request_url(u),
            anchor_of_url(u),link_in_tag(u),server_form_handler(u),email_submit(u),abnormal_url(u),website_forwarding(u),
            status_bar_customization(u),disable_right_click(u),age_of_domain(u),dns_record(u),website_rank(u),statistical_report(u)]
    return check


