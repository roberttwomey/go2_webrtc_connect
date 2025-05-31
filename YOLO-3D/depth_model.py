import sys
import os
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

# Add depth_anything folder to sys.path
sys.path.append('/Users/heather/Laika/go2_webrtc_connect/YOLO-3D/depth_anything/depth_anything')
from dpt import DepthAnything

class DepthEstimator:
    """
    Depth estimation using Depth Anything V2 with local model.
    """
    def __init__(self, model_size='small', device=None):
        """
        Initialize the depth estimator.
        
        Args:
            model_size (str): Model size ('small', 'base', 'large')
            device (str): Device to run inference on ('cuda', 'cpu', 'mps')
        """
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
                os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
                print("Using MPS with fallback for unsupported ops")
            else:
                device = 'cpu'
        
        self.device = device
        print(f"Using device: {self.device} for depth estimation")

        # Map model_size to local checkpoint paths and encoders
        model_map = {
            'small': {'path': '/Users/heather/Laika/go2_webrtc_connect/YOLO-3D/checkpoints/depth_anything_vits14.pth', 'encoder': 'vits'},
            'base': {'path': '/Users/heather/Laika/go2_webrtc_connect/YOLO-3D/checkpoints/depth_anything_vitb14.pth', 'encoder': 'vitb'},
            'large': {'path': '/Users/heather/Laika/go2_webrtc_connect/YOLO-3D/checkpoints/depth_anything_vitl14.pth', 'encoder': 'vitl'}
        }
        
        if model_size not in model_map:
            print(f"Invalid model size '{model_size}', defaulting to 'small'")
            model_size = 'small'
        
        model_path = model_map[model_size]['path']
        encoder = model_map[model_size]['encoder']
        
        print(f"Loading Depth Anything V2 model from {model_path} with encoder {encoder}")
        self.model = DepthAnything({'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]})
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint, strict=False)


        self.model.eval()

        # Define the preprocessing pipeline
        self.preprocess = transforms.Compose([
            transforms.Resize((518, 518)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def estimate_depth(self, image):
        """
        Estimate depth from an image.
        
        Args:
            image (numpy.ndarray): Input image (BGR format).
            
        Returns:
            numpy.ndarray: Depth map (normalized to 0-1).
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        input_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            depth_tensor = self.model(input_tensor)
        
        depth_map = depth_tensor.squeeze().cpu().numpy()
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        if depth_max > depth_min:
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        
        return depth_map

    def colorize_depth(self, depth_map, cmap=cv2.COLORMAP_INFERNO):
        depth_map_uint8 = (depth_map * 255).astype(np.uint8)
        return cv2.applyColorMap(depth_map_uint8, cmap)
    
    def get_depth_at_point(self, depth_map, x, y):
        if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
            return depth_map[y, x]
        return 0.0

    def get_depth_in_region(self, depth_map, bbox, method='median'):
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(depth_map.shape[1] - 1, x2), min(depth_map.shape[0] - 1, y2)
        region = depth_map[y1:y2, x1:x2]
        if region.size == 0:
            return 0.0
        if method == 'median':
            return float(np.median(region))
        elif method == 'mean':
            return float(np.mean(region))
        elif method == 'min':
            return float(np.min(region))
        else:
            return float(np.median(region))
