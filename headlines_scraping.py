import feedparser
import time
import pickle

#Function to fetch the rss feed and return the parsed RSS
def parseRSS(rss_url):
	return feedparser.parse(rss_url)

#Function that grabs the RSS feed headlines (titles) and returns them as a list

def getHeadlines(rss_url, headlines):
	feed=parseRSS(rss_url)
	for newsitem in feed['items']:
		headlines.append(newsitem['title'])
	return headlines

ep = 'http://ep00.epimg.net/rss/elpais/portada.xml'
em = 'http://estaticos.elmundo.es/elmundo/rss/portada.xml'
lv = 'http://www.lavanguardia.com/mvc/feed/rss/home.xml'
lr = 'http://www.larazon.es/rss/portada.xml'
abc = 'http://www.abc.es/rss/feeds/abcPortada.xml'

lista_periodicos = [ep, em, lv, lr, abc]

hl_ep = []
hl_em = []
hl_lv = []
hl_lr = []
hl_abc = []

lista_titulares = [hl_ep, hl_em, hl_lv, hl_lr, hl_abc]

dic_hl = dict(zip(lista_periodicos, lista_titulares))

#first request (out of the loop)
proc_headlines = {}
f=open('scraping_15.pckl', 'wb')
for k,v in dic_hl.items():
	feed=parseRSS(k)
	#Getting headlines of the selected journal
	v=getHeadlines(k,v)
	proc_headlines[k]=v
pickle.dump(proc_headlines, f)
f.close()

#MASTER LOOP: get executed every 2 hours; parses feed; saves it in a list
starttime = time.time()
while True:
	f=open('scraping_15.pckl','rb')
	object = pickle.load(f)
	f.close()
	w=open('scraping_15.pckl', 'wb')
	for k,v in object.items():
	#request
		feed = parseRSS(k)
		headlines = getHeadlines(k,v)
		print("Updated headlines")
		proc_info=set(headlines)
		proc_info=list(proc_info)
		object[k] = proc_info
	pickle.dump(object, w)
	w.close()
	time.sleep(3600.0-((time.time() -starttime) % 3600.0))
