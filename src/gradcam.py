import torch
import cv2
import numpy as np

class GradCAM:
    def __init__(self, model):
        self.model      = model
        self.gradient   = None
        self.activation = None
        model.layer4[-1].register_forward_hook(self._save_activation)
        model.layer4[-1].register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activation = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradient = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        self.model.eval()
        output = self.model(input_tensor)
        self.model.zero_grad()
        output[0, class_idx].backward()
        weights  = self.gradient.mean(dim=[2, 3], keepdim=True)
        heatmap  = (weights * self.activation).sum(dim=1).squeeze()
        heatmap  = torch.relu(heatmap).cpu().numpy()
        heatmap  = (heatmap - heatmap.min()) / (heatmap.max() + 1e-8)
        heatmap  = cv2.resize(heatmap, (224, 224))
        heatmap  = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        return heatmap

def overlay_heatmap(original_rgb_np, heatmap_bgr, alpha=0.4):
    heatmap_rgb      = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    original_resized = cv2.resize(original_rgb_np, (224, 224))
    return cv2.addWeighted(original_resized, 1 - alpha, heatmap_rgb, alpha, 0)
