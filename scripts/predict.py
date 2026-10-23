"""Save a segmentation mask, class probabilities and relative-area estimate."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from breast_benchmark.models import MODEL_REGISTRY, build_model, configure_classification_classes
from breast_benchmark.size_regression import decode_area_percent

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--model', choices=list(MODEL_REGISTRY), default='TransUNetMAESizeDeepHeads')
    p.add_argument('--classes', nargs='+', default=['normal','benign','malignant'])
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    model, _ = build_model(args.model, pretrained=False)
    configure_classification_classes(model, tuple(args.classes))
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(payload['model'] if 'model' in payload else payload)
    model.to(args.device).eval()
    with Image.open(args.image) as image:
        original_size = (image.height, image.width)
        image = TF.pil_to_tensor(image.convert('RGB')).float()/255
    image = TF.resize(image,[224,224],interpolation=InterpolationMode.BILINEAR,antialias=True)
    with torch.no_grad():
        segmentation, logits, log_area = model.forward_multitask(image[None].to(args.device))
        probability = torch.sigmoid(segmentation[0].float())
        probability = F.interpolate(probability,original_size,mode='bilinear',align_corners=False)
        mask = (probability[0,0].cpu().numpy() >= .5).astype(np.uint8)*255
        classes = torch.softmax(logits.float(),dim=1)[0].cpu().tolist()
        area = float(decode_area_percent(log_area)[0])
    args.output_root.mkdir(parents=True,exist_ok=True)
    Image.fromarray(mask).save(args.output_root/'segmentation.png')
    (args.output_root/'prediction.json').write_text(json.dumps({
        'class_probabilities':dict(zip(args.classes,classes)),
        'predicted_class':args.classes[int(np.argmax(classes))],
        'estimated_tumor_area_percent':area,
        'segmentation_mask':'segmentation.png'},indent=2)+'\n')

if __name__ == '__main__':
    main()
