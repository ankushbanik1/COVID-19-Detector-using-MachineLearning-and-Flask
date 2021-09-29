from flask import Flask,render_template,request
import pickle

# from keras import models
file=open('coronadetector\my_model.pkl','rb')
clf=pickle.load(file)

file.close()

app=Flask(__name__)


@app.route('/',methods=['GET','POST'])
def hello_world():
    if request.method == 'POST':
        mydict=request.form
        fever=int(mydict['Fever'])
        bodypain=int(mydict['Bodypain'])
        age=int(mydict['age'])
        runnynose=int(mydict['Runnynose'])
        breathing=int(mydict['breathhing'])
        Cough=int(mydict['Cough'])
        
        input_feature=[fever,bodypain,age,runnynose,breathing,Cough]
        #input_feature=[100,1,45,1,1,0]
        infprob=clf.predict_proba([input_feature])[0][1]
        print(infprob)
        return render_template('result.html',inf=infprob)
   
    return render_template('index.html')
   
if __name__ == '__main__'  :
    app.run(debug=False) 
