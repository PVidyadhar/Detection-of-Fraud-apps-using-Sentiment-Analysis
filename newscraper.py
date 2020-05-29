from google_play_scraper import Sort, reviews


def scrape(appid):
	result = reviews(
	    appid,
	    lang='en', # defaults to 'en'
	    country='us', # defaults to 'us'
	    sort=Sort.MOST_RELEVANT, # defaults to Sort.MOST_RELEVANT
	    count=100, # defaults to 100
	  #  filter_score_with=5 # defaults to None(means all score)
	)
	file = open('reviews_scraped.txt','a+')
	#print(len(result))
	#print((result),end="\n\n")
	for i in result:
		print(i['content'],end="\n\n")


	for i in result:
		file.writelines(str((i['content']).encode('utf-8')))
		file.write('\n') 

	return result

#scrape('camera1.themaestrochef.com.cameraappfordogs')

#rey kedar mute nundi teey
#em vinpiyatle