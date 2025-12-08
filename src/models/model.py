import pandas as pd
import numpy as np
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns
import os
import zipfile
from sklearn.linear_model import LinearRegression

class XPModel:
    def __init__(self, parameters):
        self.parameters = parameters
        self.trained = False

    @staticmethod
    def load_and_preprocess_data():
        # Path to the archive.zip file
        zip_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'data', 'archive.zip')
        zip_path = os.path.abspath(zip_path)
        print("Path to archive.zip:", zip_path)

        # List files in the zip to find the CSV name
        with zipfile.ZipFile(zip_path, 'r') as archive:
            print("Files in archive:", archive.namelist())
            # Try to find the first CSV file in the archive
            csv_files = [f for f in archive.namelist() if f.endswith('.csv')]
            if not csv_files:
                raise FileNotFoundError("No CSV file found in archive.zip")
            csv_name = csv_files[0]
            print("Using CSV file:", csv_name)
            with archive.open(csv_name) as csvfile:
                df = pd.read_csv(csvfile)

        # Convert GAME_DATE to datetime
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

        # Extract year from GAME_DATE
        df['YEAR'] = df['GAME_DATE'].dt.year

        # Aggregate data by year
        yearly_data = df.groupby('YEAR').agg({
            'LOC_X': 'mean',
            'LOC_Y': 'mean',
            'SHOT_DISTANCE': 'mean',
            'SHOT_TYPE': lambda x: (x == '3PT').mean(),  # Proportion of 3-pointers
            'SHOT_MADE': 'mean',  # FG%
            'GAME_DATE': 'count'  # Total shots
        }).rename(columns={'SHOT_MADE': 'FG_PCT', 'GAME_DATE': 'TOTAL_SHOTS'}).reset_index()

        print(yearly_data[['YEAR', 'SHOT_DISTANCE', 'SHOT_TYPE', 'FG_PCT', 'TOTAL_SHOTS']])
        return df, yearly_data

    @staticmethod
    def plot_trends(yearly_data):
        # Plot shot distance trends
        plt.figure(figsize=(10, 6))
        plt.plot(yearly_data['YEAR'], yearly_data['SHOT_DISTANCE'], marker='o', label='Shot Distance')
        plt.title('Average Shot Distance Over Time')
        plt.xlabel('Year')
        plt.ylabel('Shot Distance (ft)')
        plt.grid(True)
        plt.legend()
        plt.show()

        # Plot 3PT shot trends
        plt.figure(figsize=(10, 6))
        plt.plot(yearly_data['YEAR'], yearly_data['SHOT_TYPE'], marker='o', color='orange', label='3PT Shot Proportion')
        plt.title('Proportion of 3PT Shots Over Time')
        plt.xlabel('Year')
        plt.ylabel('Proportion of 3PT Shots')
        plt.grid(True)
        plt.legend()
        plt.show()

        # Plot FG% over time
        plt.figure(figsize=(10, 6))
        plt.plot(yearly_data['YEAR'], yearly_data['FG_PCT'], marker='o', color='green', label='FG%')
        plt.title('Field Goal Percentage Over Time')
        plt.xlabel('Year')
        plt.ylabel('FG%')
        plt.grid(True)
        plt.legend()
        plt.show()

        # Plot total shots per year
        plt.figure(figsize=(10, 6))
        plt.bar(yearly_data['YEAR'], yearly_data['TOTAL_SHOTS'], color='purple', label='Total Shots')
        plt.title('Total Shots Per Year')
        plt.xlabel('Year')
        plt.ylabel('Total Shots')
        plt.grid(True)
        plt.legend()
        plt.show()

    @staticmethod
    def plot_shot_location_heatmap(df):
        # Filter for made shots
        made_shots = df[df['SHOT_MADE'] == True]

        # Create a heatmap of shot locations
        plt.figure(figsize=(12, 8))
        sns.kdeplot(
            x=made_shots['LOC_X'], 
            y=made_shots['LOC_Y'], 
            cmap='Reds', 
            fill=True, 
            thresh=0, 
            levels=100
        )
        plt.title('Heatmap of Shot Locations (Made Shots)')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.show()

    def train_trend_model(self, yearly_data):
        # Train a linear regression model for each feature
        self.models = {}
        for feature in ['LOC_X', 'LOC_Y', 'SHOT_DISTANCE', 'SHOT_TYPE']:
            X = yearly_data[['YEAR']]
            y = yearly_data[feature]
            model = LinearRegression()
            model.fit(X, y)
            self.models[feature] = model
        self.trained = True

    def predict_trends(self, future_years):
        if not self.trained:
            raise Exception("Trend model must be trained before making predictions.")
        
        # Predict future trends
        predictions = {}
        for feature, model in self.models.items():
            predictions[feature] = model.predict(future_years)
        return predictions