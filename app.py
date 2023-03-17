import sys
import os

from flask import Flask,render_template,request,jsonify

from src.pipeline.predict_pipeline import CustomeData,PredictPipeline
from src.exception import CustomeException
from src.logger import logging




application=Flask(__name__)
app=application


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/home',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        try:
            data=CustomeData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('writing_score')),
                writing_score=float(request.form.get('reading_score'))
            )

            pred_input_df=data.get_data_as_data_frame()
            logging.info(f"Data for Prediction is {pred_input_df.values}")
            predict_pipeline=PredictPipeline()
            results=predict_datapoint.predict(pred_input_df)
            return render_template('home.html',results=results[0])



        except Exception as e:
            logging.info(f'Error Occured {CustomeException(e,sys)}')


if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)
