@echo off
echo ========================================
echo AgroClimate - Crop Climate Impact Prediction Model
echo ========================================

echo.
echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Running AgroClimate pipeline...
python src/main.py --mode full

echo.
echo ========================================
echo AgroClimate execution completed!
echo ========================================

echo.
echo Check the following directories for results:
echo - Models: models/
echo - Predictions: outputs/predictions/
echo - Visualizations: outputs/visualizations/

pause