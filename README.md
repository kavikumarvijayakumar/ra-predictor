# AI-Based Surface Roughness (Ra) Prediction – End Milling

## Setup
    pip install -r requirements.txt
    python generate_dataset.py     # synthetic demo data -> data/milling_data.csv
    python train.py                # compares 6-7 models, saves best to artifacts/
    streamlit run app.py           # launch the application

## Use your own data
Replace `data/milling_data.csv` (or upload in the app's "Data & retrain" tab) with columns:
speed (rpm), feed_per_tooth (mm/tooth), doc (mm), woc (mm), tool_wear (mm), tool_dia (mm),
flutes, material (Al6061 / AISI1045 / Ti6Al4V, or your own labels), Ra (um).

## Files
- common.py – constants, derived features (cutting speed, MRR)
- generate_dataset.py – synthetic data generator
- train.py – training, CV, comparison, permutation importance, model export
- optimizer.py – Ra/MRR optimisation (differential evolution on the surrogate model)
- app.py – Streamlit UI (Predict, Optimize, Batch, Model report, Retrain)

Note: the bundled dataset is synthetic; accuracy numbers only become meaningful on real measurements.
