from pathlib import Path
import argparse
import json
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mss_transnet.model import MSS_TransNet
from mss_transnet.losses import decode_area


def main():
    p=argparse.ArgumentParser(description="Predict the lesion mask, diagnosis and relative lesion area for one image")
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--image",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--binary",action="store_true")
    a=p.parse_args();payload=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
    model=MSS_TransNet(**payload["model_config"]);model.load_state_dict(payload["model"])
    model.to(a.device).eval()
    with Image.open(a.image) as image:
        size=(image.height,image.width)
        array=np.asarray(image.convert("RGB").resize((224,224),Image.Resampling.BILINEAR)).copy()
    tensor=torch.from_numpy(array).permute(2,0,1).float()[None].to(a.device)/255
    with torch.no_grad():
        result=model(tensor);classes=payload["classes"];logits=result["classification"]
        if a.binary and len(classes)==3:
            logits=logits[:,[classes.index("benign"),classes.index("malignant")]];classes=["benign","malignant"]
        probability=logits.softmax(1)[0].cpu().tolist()
        # Metrics use the 224 x 224 grid; the original-size mask is provided for viewing.
        mask=(result["segmentation"].sigmoid()[0,0].cpu().numpy()>=.5).astype(np.uint8)*255
        original=F.interpolate(result["segmentation"].sigmoid(),size,mode="bilinear",align_corners=False)
        original=(original[0,0].cpu().numpy()>=.5).astype(np.uint8)*255
        area=float(decode_area(result["area"])[0])
    a.output.mkdir(parents=True,exist_ok=True)
    Image.fromarray(mask).save(a.output/"mask_224.png")
    Image.fromarray(original).save(a.output/"mask_original_size.png")
    (a.output/"prediction.json").write_text(json.dumps(dict(class_probabilities=dict(zip(classes,probability)),
        predicted_class=classes[int(np.argmax(probability))],auxiliary_relative_area_percent=area,
        area_is_physical_measurement=False),indent=2)+"\n")

if __name__=="__main__":main()
