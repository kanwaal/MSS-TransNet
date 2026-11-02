from pathlib import Path
import csv
import json

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn.functional as F

from mss_transnet.data import UltrasoundDataset,prepare_busi,prepare_busbra,read_manifest
from mss_transnet.losses import class_weights,reconstruction_loss,joint_losses
from mss_transnet.metrics import segmentation_summary,classification_summary
from mss_transnet.model import MSS_TransNet,deep_head
from mss_transnet._vendor.transunet2024.decoder2d import MultiScaleMaskedTransformerDecoder2d


@pytest.fixture(scope="module")
def model():
    torch.set_num_threads(2);torch.manual_seed(42)
    return MSS_TransNet()


def test_architecture_and_full_masked_grid(model):
    linears=[m for m in model.classifier if isinstance(m,torch.nn.Linear)]
    assert [(x.in_features,x.out_features) for x in linears]==[(768,256),(256,128),(128,64),(64,3)]
    recorded=[]
    handle=model.encoder.encoder.register_forward_pre_hook(lambda module,args:recorded.append(args[0].shape))
    image=torch.rand(1,3,224,224);model.eval()
    with torch.no_grad():
        prediction,pixels=model.reconstruct(image,.30,torch.Generator().manual_seed(42))
        result=model(image)
    handle.remove()
    assert prediction.shape==(1,3,224,224)
    assert pixels.sum()==59*16*16
    assert all(shape[1:]==(196,768) for shape in recorded)
    assert result["segmentation"].shape==(1,1,224,224)
    assert result["classification"].shape==(1,3) and result["area"].shape==(1,)
    assert all(torch.isfinite(x).all() for x in result.values())


def test_only_hidden_pixels_affect_reconstruction():
    pred=torch.zeros(1,3,32,32);target=torch.ones_like(pred)
    hidden=torch.zeros(1,1,32,32,dtype=torch.bool);hidden[:,:,:16,:16]=True
    expected=reconstruction_loss(pred,target,hidden)
    pred[:,:,16:,:]=999
    assert reconstruction_loss(pred,target,hidden)==expected
    assert expected==1


def test_phase_trainability(model):
    model.set_trainable("ssl",{"reconstruction":1})
    assert all(p.requires_grad for p in model.encoder.parameters())
    assert all(p.requires_grad for p in model.reconstruction.parameters())
    assert all(not p.requires_grad for p in model.classifier.parameters())
    assert all(not p.requires_grad for p in model.segmentation.parameters())
    model.set_trainable("sml",{k:1 for k in ["segmentation","classification","area","reconstruction"]},True)
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert all(p.requires_grad for p in model.classifier.parameters())
    model.set_trainable("sml",{k:1 for k in ["segmentation","classification","area","reconstruction"]})


def test_attention_mask_uses_prediction_and_handles_empty_roi():
    decoder=MultiScaleMaskedTransformerDecoder2d(in_channels=24,mask_classification=False,
        num_classes=1,hidden_dim=24,num_queries=1,nheads=4,dim_feedforward=96,
        dec_layers=1,pre_norm=False,mask_dim=24,enforce_input_project=False,
        non_object=False,num_feature_levels=3,is_masking=True,is_masking_argmax=False,
        is_mhsa_float32=False,no_max_hw_pe=False,use_layer_scale=False).eval()
    with torch.no_grad():
        for parameter in decoder.mask_embed.parameters():parameter.zero_()
        decoder.mask_embed.layers[-1].bias.fill_(1.)
    query=torch.zeros(1,2,24)
    prior=torch.tensor([[[[-100.,100.],[-100.,100.]]]]).expand(2,24,-1,-1)/24
    _,_,mask=decoder.forward_prediction_heads(query,prior,(2,2),False)
    assert mask[0,0].tolist()==[True,False,True,False]
    seen=[];attention=decoder.transformer_cross_attention_layers[0].multihead_attn
    original=attention.forward
    def capture(*args,**kwargs):
        seen.append(kwargs['attn_mask'].clone())
        return original(*args,**kwargs)
    attention.forward=capture
    features=[torch.rand(2,24,k,k) for k in [2,4,8]]
    output=decoder(features,-torch.ones(2,24,16,16))
    attention.forward=original
    assert torch.isfinite(output['pred_masks']).all()
    assert not seen[0].any()  # fully forbidden masks are made safe in upstream forward


def test_ssl_does_not_open_annotation_files(tmp_path):
    image=tmp_path/"image.png";Image.fromarray(np.zeros((32,32,3),dtype=np.uint8)).save(image)
    rows=[dict(sample_id=s,group_id=s,class_name="benign",image_path=str(image),
               mask_paths=str(tmp_path/"DOES_NOT_EXIST.png"),split=s) for s in ["train","val","test"]]
    path=tmp_path/"manifest.csv"
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    sample=UltrasoundDataset(path,"train",image_only=True)[0]
    assert set(sample)=={"image","sample_id"}


def test_class_weight_normalization_and_reporting():
    weights=class_weights([0,0,0,1,2],3)
    logits=torch.tensor([[1.,2.,3.],[3.,2.,1.]])
    targets=torch.tensor([0,2])
    actual=F.cross_entropy(logits,targets,weight=weights)
    manual=-(logits.log_softmax(1)[range(2),targets]*weights[targets]).sum()/weights[targets].sum()
    assert torch.allclose(actual,manual)
    rows=[dict(class_name="normal",**{k:1 for k in ["dice","iou","precision","sensitivity","specificity","pixel_accuracy"]}),
          dict(class_name="benign",**{k:.5 for k in ["dice","iou","precision","sensitivity","specificity","pixel_accuracy"]})]
    summary=segmentation_summary(rows)
    assert summary["images"]==1 and summary["mean"]["dice"]==.5
    classified=[dict(true_index=i,**{"probability_"+c:float(i==j) for j,c in enumerate(["normal","benign","malignant"])}) for i in range(3)]
    c=classification_summary(classified,("normal","benign","malignant"))
    assert c["macro_specificity"]==c["macro_f1"]==c["accuracy"]==1


def test_busi_folds_and_target_patients(tmp_path):
    root=tmp_path/"BUSI"
    for name in ["normal","benign","malignant"]:
        directory=root/name;directory.mkdir(parents=True)
        for i in range(20):
            Image.fromarray(np.zeros((8,8),dtype=np.uint8)).save(directory/f"{name} ({i}).png")
            Image.fromarray(np.zeros((8,8),dtype=np.uint8)).save(directory/f"{name} ({i})_mask.png")
    prepare_busi(root,tmp_path/"busi_folds")
    heldout=[]
    for path in sorted((tmp_path/"busi_folds").glob("*.csv")):
        rows=read_manifest(path);heldout += [r["sample_id"] for r in rows if r["split"]=="test"]
    assert len(heldout)==len(set(heldout))==60
    target=tmp_path/"BUSBRA";(target/"Images").mkdir(parents=True);(target/"Masks").mkdir()
    metadata=[]
    for i in range(80):
        identifier=f"bus_{i:04d}"
        metadata.append(dict(ID=identifier,Case=str(i//2),Pathology="benign" if i//2<20 else "malignant"))
        Image.fromarray(np.zeros((8,8),dtype=np.uint8)).save(target/"Images"/(identifier+".png"))
        Image.fromarray(np.zeros((8,8),dtype=np.uint8)).save(target/"Masks"/(identifier.replace("bus_","mask_")+".png"))
    with (target/"bus_data.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(metadata[0]));writer.writeheader();writer.writerows(metadata)
    prepare_busbra(target,tmp_path/"target")
    full,limited=[read_manifest(tmp_path/"target"/(n+".csv")) for n in ["full","limited"]]
    assert {r["sample_id"] for r in full if r["split"]=="test"}=={r["sample_id"] for r in limited if r["split"]=="test"}
