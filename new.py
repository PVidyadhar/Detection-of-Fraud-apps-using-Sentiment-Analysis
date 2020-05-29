
import pickle
def sentiment(result):

	with open(f'model.pkl', 'rb') as f:
	    ensemble_clf = pickle.load(f)
	prediction=[]
	for i in result:
		temp=ensemble_clf.predict([i['content']])
		print(temp.astype(int))
		

		prediction.append(temp)

	p=0
	n=0
	for i in prediction:
		if i==1:
			p+=1
		else:
			n+=1
	overall=max(p,n)/(p+n)
	ratio=(p/(p+n))
	print(p,n,overall)
	fraud=0
	total=0
	li=['fraud','fake','duplicate','cheat','tricked','trap','deceive','scam']
	if(ratio<0.5):
		for r in result:
			rev=r['content']
			rev=rev.split(' ')
			print(rev)
			for i in rev:
				if i in li:
					print(i)
					fraud=1
					break
			if fraud==1:
				total+=1
				fraud=0
	newfraud=0
	print(total)
	#if total>=((p+n)//2):
	if total >= 1:
		newfraud=1
	print(newfraud)
	return prediction,ratio,overall,newfraud
