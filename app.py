import flask
import pickle
import io
import random
from newscraper import scrape
from flask import Response
# Use pickle to load in the pre-trained model.
from live_classifier1 import get_sentiment
app = flask.Flask(__name__, template_folder='templates')
@app.route('/')
def main():
	return(flask.render_template('hello43.html'))

@app.route('/first', methods=[ 'POST'])
def hello():
        appid = flask.request.form['appid']

        result=scrape(appid)
        #x,conf=get_sentiment(result)
        prediction,ratio,overall,newfraud=get_sentiment(result)
        #input_variables = pd.DataFrame([[review]],columns=['review'],dtype=string) original_input={'review':review} input={'ratio':ratio}
        #print(x,conf,prediction)
        '''p=0
        n=0
        for i in prediction:
            if i==1:
                p+=1
            else:
                n+=1
        overall=max(p,n)/(p+n)
        ratio=(p/(p+n))
        print(p,n,overall)'''
        mr=round((ratio*10),2)
        return flask.render_template('ansnew.html',ratio=ratio,mr=mr,overall=overall,newfraud=newfraud)
        #return str('rEturned')
#adarsh.contribution()




if __name__ == '__main__':
    app.run(debug=True)