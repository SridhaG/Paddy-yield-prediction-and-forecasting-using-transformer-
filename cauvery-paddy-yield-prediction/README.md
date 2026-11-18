# Cauvery Delta Paddy Yield Prediction

District-season paddy yield (kg/ha) for Thanjavur, Thiruvarur, and Nagapattinam using a 15-box area-weighted temporal transformer, with LSTM, Random Forest, and XGBoost baselines.

## Run the pipeline

```bash
pip install -r requirements-train.txt
python src/collect_gee_data.py --synthetic
python src/train.py --model all
python src/evaluate.py
python src/visualize.py
```

`--synthetic` builds satellite and weather tables in the same schema as the Google Earth Engine collector. Yield labels are in `data/yield.csv`.

## Dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Models

- AW-MIL temporal transformer (main model)
- LSTM baseline
- Random Forest
- XGBoost
