# main.py

from models.model import XPModel
import numpy as np
import pandas as pd

def main():
    print("Initializing NBA Shot Trend Prediction Application")
    
    # Load and preprocess data
    df, yearly_data = XPModel.load_and_preprocess_data()

    # Plot historical trends
    print("Plotting historical trends...")
    XPModel.plot_trends(yearly_data)

    # Plot shot location heatmap
    print("Plotting shot location heatmap...")
    XPModel.plot_shot_location_heatmap(df)

if __name__ == "__main__":
    main()