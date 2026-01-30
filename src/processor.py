import pandas as pd 
import numpy as np 

def calculate_distance(lat1,lon1, lat2, lon2):
    """ Calculamos la distancia Haversine entre el cliente y el comercio. """
    r = 6371 # Radio de la Tierra en km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * r * np.arctan2(np.sqrt(a), np.sqrt(1-a))


def preprocess_data(df):
    """Pipeline de limpieza e ingeriería de características."""

    # 1. Ingeniería de distancias

    df['distancia_km'] = calculate_distance(df['lat'],df['long'],df['merch_lat'],df['merch_long'])

    # 2. Ingeniería de tiempo
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['hora'] = df['trans_date_trans_time'].dt.hour
    df['dia_semana'] = df['trans_date_trans_time'].dt.dayofweek

    # 3. Ingenieía demografica
    df['dob'] = pd.to_datetime(df['dob'])
    df['edad'] = (df['trans_date_trans_time'] - df['dob']).dt.days // 365 

    # 4. Selección de variables para el modelo

    features = ['amt','distancia_km','hora','dia_semana','edad']
   
   
    # Si la etiqueta is_fraud existe (train/test),  la mantenemos
    if 'is_fraud' in df.columns:
        features.append('is_fraud')

    return df[features].copy()
