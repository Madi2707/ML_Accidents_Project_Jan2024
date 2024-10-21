import sys
import pandas as pd
from src.components.exception import CustomException
from src.components.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)
            data_scale= preprocessor.transform(features)
            # Convert sparse matrix to dense array
            data_scale_dense = data_scale.toarray()
            preds = model.predict(data_scale_dense)
                   
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, kpi_cause_code: str, light_code:str, weather_code:str, traffic_code:str, 
                 road_type_code:str, road_code:str, collision_code:str, damage_code:str, location_code:str):
        
        self.kpi_cause_code = kpi_cause_code
        self.light_code = light_code
        self.weather_code = weather_code
        self.traffic_code = traffic_code
        self.road_type_code = road_type_code
        self.road_code = road_code
        self.collision_code = collision_code
        self.damage_code = damage_code
        self.location_code = location_code

    def get_data_as_df(self):
        try:
            custom_data_dict = {'kpi_cause_code': [self.kpi_cause_code],
                                'light_code': [self.light_code],
                                'weather_code': [self.weather_code],
                                'traffic_code': [self.traffic_code],
                                'road_type_code':[self.road_type_code],
                                'road_code': [self.road_code],
                                'collision_code': [self.collision_code],
                                'damage_code': [self.damage_code],
                                'location_code': [self.location_code]}
            return pd.DataFrame(custom_data_dict)
        
        except Exception as e:
            raise CustomException(e, sys)
