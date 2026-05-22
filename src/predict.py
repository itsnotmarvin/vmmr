import torch
from torchvision import transforms
from model import get_model

val_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_model(path):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    ckpt   = torch.load(path, map_location=device)
    model  = get_model(num_classes=196).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt['class_names'], device

def predict(pil_image, model, class_names, device, top_k=5):
    tensor = val_tfm(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        out   = model(tensor)
        probs = torch.softmax(out, dim=1)
    values, indices = probs.topk(top_k)
    return [(class_names[i], float(c) * 100) for i, c in zip(indices[0], values[0])]

def get_gradcam_tensor(pil_image, device):
    return val_tfm(pil_image).unsqueeze(0).to(device)
