import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect, flash, send_file
from sklearn.preprocessing import MinMaxScaler
from werkzeug.utils import secure_filename
import pickle

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect, flash, send_file
from sklearn.preprocessing import MinMaxScaler
from werkzeug.utils import secure_filename
import pickle
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

app = Flask(__name__) #Initialize the flask App

stress = pickle.load(open('stress.pkl','rb'))
@app.route('/')

@app.route('/index')
def index():
	return render_template('index.html')
 

#@app.route('/future')
#def future():
#	return render_template('future.html')    

@app.route('/login')
def login():
	return render_template('login.html')
@app.route('/upload')
def upload():
    return render_template('upload.html')  
@app.route('/preview',methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset,encoding = 'unicode_escape')
        df.set_index('Id', inplace=True)
        return render_template("preview.html",df_view = df)	


#@app.route('/home')
#def home():
 #   return render_template('home.html')

@app.route('/prediction', methods = ['GET', 'POST'])
def prediction():
    return render_template('prediction.html')


#@app.route('/upload')
#def upload_file():
#   return render_template('BatchPredict.html')




@app.route('/predict', methods=['POST'])
def predict():
    int_feature = [x for x in request.form.values()]
    final_features = [np.array(int_feature)]
    result = stress.predict(final_features)

    # Map prediction to stress levels
    stress_levels = ["Low/Normal", "Medium Low", "Medium", "Medium High", "High"]
    predicted_level = stress_levels[result[0]]

    # Define recommendations
    recommendations = {
        "Low/Normal": ["Maintain a healthy diet", "Engage in physical activities", "Get enough sleep"],
        "Medium Low": ["Practice deep breathing exercises", "Take regular breaks from work", "Stay hydrated"],
        "Medium": ["Try meditation or yoga", "Reduce screen time", "Listen to calming music"],
        "Medium High": ["Talk to friends or family", "Seek professional counseling", "Consider relaxation therapy"],
        "High": ["Consult a doctor or therapist", "Follow a structured stress management plan", "Avoid excessive caffeine or alcohol"]
    }

    # Get recommendations for the predicted stress level
    suggested_actions = recommendations.get(predicted_level, ["No recommendation available"])

    return render_template('prediction.html', prediction_text=predicted_level, recommendations=suggested_actions)

	
    
@app.route('/performance')
def performance():
	return render_template('performance.html')
    
@app.route('/chart')
def chart():
	return render_template('chart.html')    
    
    
    
if __name__ == "__main__":
    app.run(debug=True)

