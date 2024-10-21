import sys
import os

# Add the path to the src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))

from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from flask import Flask, request, render_template


application = Flask(__name__)

app = application

# Route for our homepage
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home1.html')
    else:
        data = CustomData(
            kpi_cause_code= request.form.get('kpi_cause_code'),
            light_code= request.form.get('light_code'),
            weather_code= request.form.get('weather_code'),
            traffic_code= request.form.get('traffic_code'),
            road_type_code= request.form.get('road_type_code'),
            road_code= request.form.get('road_code'),
            collision_code= request.form.get('collision_code'),
            damage_code= request.form.get('damage_code'),
            location_code= request.form.get('location_code')
            )
        pred_df = data.get_data_as_df()
        print(pred_df)
        
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
               
        return render_template('home1.html', results=results[0])


if __name__=="__main__":
    app.run(host="127.0.0.1", port=8080, debug=True) # remove debug and port, when deploying as container to the cloud.