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

Built by Marvin Ayala — Kean University AI
