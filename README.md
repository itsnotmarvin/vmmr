# CarIQ — Vehicle Intelligence

Upload any car photo. Get instant identification, reliability data, repair costs, and what to inspect before buying.

## Stack
ResNet-50 (PyTorch) · Gradio · Hugging Face Spaces · M1 MPS

## Run
```bash
cd src && /opt/anaconda3/envs/vmmr/bin/python train.py   # train
cd .. && /opt/anaconda3/envs/vmmr/bin/python app/app.py  # run app
```

## Structure
- `src/` — model, training, inference, Grad-CAM
- `app/` — Gradio UI
- `data/` — knowledge base
- `models/` — saved weights (after training)

## Dataset
Training needs the Stanford Cars dataset (~1.9 GB), not tracked in this repo.
Download it from [Kaggle](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset)
and extract so the repo root contains:
```
archive/
  car_devkit/
  cars_test/
  cars_train/
```

