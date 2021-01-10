from flask import Flask, render_template, request, redirect, session
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, FloatField, IntegerField, SelectField, SubmitField
from wtforms.validators import DataRequired, InputRequired, NumberRange
import pandas as pd
import numpy as np
from numpy import random
import joblib


df= pd.read_csv("YearPredictionMSD.txt",sep=",",header=None)
app= Flask(__name__)
Bootstrap(app)
app.config["SECRET_KEY"]="hard to guess"
predict_value=[]

### Forms ###

class EnterYourInfos(FlaskForm):

	number=random.randint(0,df.shape[0]-2)
	row=df.loc[number]
	predict_value.append(row[0])

	t1 = FloatField("Timbre average 1 :", validators= [DataRequired(), NumberRange(min =df.min()[1], max=df.max()[1])],default=row[1])
	t2 = FloatField("Timbre average 2 :", validators= [DataRequired(), NumberRange(min =df.min()[2], max=df.max()[2])],default=row[2])
	t3 = FloatField("Timbre average 3 :", validators= [DataRequired(), NumberRange(min =df.min()[3], max=df.max()[3])],default=row[3])
	t4 = FloatField("Timbre average 4 :", validators= [DataRequired(), NumberRange(min =df.min()[4], max=df.max()[4])],default=row[4])
	t5 = FloatField("Timbre average 5 :", validators= [DataRequired(), NumberRange(min =df.min()[5], max=df.max()[5])],default=row[5])
	t6 = FloatField("Timbre average 6 :", validators= [DataRequired(), NumberRange(min =df.min()[6], max=df.max()[6])],default=row[6])
	t7 = FloatField("Timbre average 7 :", validators= [DataRequired(), NumberRange(min =df.min()[7], max=df.max()[7])],default=row[7])
	t8 = FloatField("Timbre average 8 :", validators= [DataRequired(), NumberRange(min =df.min()[8], max=df.max()[8])],default=row[8])
	t9 = FloatField("Timbre average 9 :", validators= [DataRequired(), NumberRange(min =df.min()[9], max=df.max()[9])],default=row[9])
	t10 = FloatField("Timbre average 10 :", validators= [DataRequired(), NumberRange(min =df.min()[10], max=df.max()[10])],default=row[10])
	t11 = FloatField("Timbre average 11 :", validators= [DataRequired(), NumberRange(min =df.min()[11], max=df.max()[11])],default=row[11])
	t12 = FloatField("Timbre average 12 :", validators= [DataRequired(), NumberRange(min =df.min()[12], max=df.max()[12])],default=row[12])


	submit = SubmitField("submit")


@app.route("/")
def index():
	return render_template('base.html')


@app.route("/predict",methods=["GET","POST"])
def prediction():
	form=EnterYourInfos(request.form)
	if request.method=="POST" and form.validate_on_submit():

		rf= joblib.load("result.joblib")
		predic=[form.t1.data,form.t2.data,form.t3.data,form.t4.data,form.t5.data,form.t6.data,
		form.t7.data,form.t8.data,form.t9.data,form.t10.data,form.t11.data,form.t12.data]
		predic=np.array(predic)
		predic=np.expand_dims(predic,0)
	
		
		res= rf.predict(predic)
		predict_value.append(res)
		return redirect("/results")

	return render_template('prediction_form.html',form=form)



@app.route("/results")
def show_result():
	decade=predict_value[0]-predict_value[0]%10
	return render_template('results.html',pred=predict_value[1][0],true_value=predict_value[0],true_decade=decade)


if __name__ == '__main__':
	print(app.url_map)
	app.run(host='127.0.0.1', port=5000, debug=True)




