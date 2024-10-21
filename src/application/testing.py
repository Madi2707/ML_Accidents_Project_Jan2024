import sys
import os

# Add the path to the src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))

# Now import your modules
try:
    from src.pipeline.predict_pipeline import CustomData, PredictPipeline  # Adjust this import
    print("Imports successful!")
except Exception as e:
    print("Import failed:", e)

