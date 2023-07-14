#importing necessary libraries/modules
from flask import Flask,render_template,request
import pickle
import numpy as np
import importlib
import itertools

#loading data from pickle files into variables
clean_ratings = pickle.load(open('clean_ratings.pkl','rb'))
pvTable = pickle.load(open('pvTable.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

#basic Flask app boilerplate code
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

'''definition of function to take input from text field
using http POST method for the kNN model and
print out a list of similarly rated movies
according to first movie containing the input genre tag'''
@app.route('/recommend_movies',methods=['post'])
def recommend():

    try:
        name = request.form.get('query')

        check = clean_ratings['title'].str.contains(name,case=False,regex=True)
        idx = check[check].index[0]
        nom = clean_ratings.at[idx,'title']
        mov_id = np.where(pvTable.index == nom)[0][0]
        distances, recommendations = model.kneighbors(pvTable.iloc[mov_id].dropna().values.reshape(1, -1), n_neighbors=10)
    
        data = []
        links = []
        dist = []

        for i in range(len(recommendations)):
            for n in range(len(recommendations[i])):
                data.append(pvTable.index[recommendations[i][n]])
                dist.append(str(distances[i][n]))
                l = str(clean_ratings.at[clean_ratings[clean_ratings['title'] == (pvTable.index[recommendations[i][n]])].index[0],'imdbId'])
                if len(l) == 5:
                    l = ('{}'+l).format('tt00')
                elif len(l) == 6:
                    l = ('{}'+l).format('tt0')
                links.append(l)

        return render_template('index.html',name=name,d_l_dt=zip(data,links,dist))

    except IndexError:
        error_msg = "Sorry, this movie isn't in the database."
        return render_template('index.html',error_msg = error_msg)

if __name__ == "__main__":
    app.run()