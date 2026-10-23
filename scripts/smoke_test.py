"""Check all head variants, 30% masking and gradients without data or weights."""
from pathlib import Path
import gc
import json
import sys
import torch
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from breast_benchmark.models import MODEL_REGISTRY, build_model, configure_classification_classes
from breast_benchmark.train import parser
from breast_benchmark.pretrain import parser as ssl_parser

def main():
    torch.set_num_threads(2)
    args = parser().parse_args([])
    assert args.mae_mask_ratio == .30 and not args.pretrained
    assert ssl_parser().parse_args(['--image-manifest','unused.csv','--output-root','unused']).mae_mask_ratio == .30
    assert json.loads((ROOT/'configs/joint_mask30.json').read_text())['mae_mask_ratio'] == .30
    torch.manual_seed(42)
    baseline, _ = build_model('TransUNetMAESize',pretrained=False)
    reference = {n:p.detach().clone() for n,p in baseline.model.state_dict().items()}
    del baseline
    results = []
    for key,spec in MODEL_REGISTRY.items():
        torch.manual_seed(42)
        model,_ = build_model(key,pretrained=False)
        assert sum(p.numel() for p in model.parameters()) == spec.expected_parameters
        assert all(torch.equal(p,reference[n]) for n,p in model.model.state_dict().items())
        mask, visible = model.random_patch_mask(3,.30,torch.device('cpu'))
        assert mask.sum(1).eq(59).all() and visible.shape == (3,137)
        pixels = model._pixel_mask(mask)
        assert pixels.shape == (3,1,224,224)
        assert pixels.sum(1).sum((1,2)).eq(59*16*16).all()
        if key.endswith('DeepHeads'):
            image = torch.rand(1,3,224,224)
            model.eval()
            with torch.no_grad():
                outputs,classes,area = model.forward_multitask(image)
                rec,hidden = model.forward_reconstruction(image,.30)
                loss = model.compute_reconstruction_loss(rec,image,hidden)
            assert outputs[0].shape == (1,1,224,224) and classes.shape == (1,3)
            assert area.eq(0).all() and rec.shape == (1,196,768) and torch.isfinite(loss)
            optimizer = torch.optim.SGD(list(model.classifier.parameters())+list(model.tumor_size_regressor.parameters()),lr=.01)
            pooled = torch.randn(3,768,1,1)
            for step in range(2):
                optimizer.zero_grad()
                loss = F.cross_entropy(model.classifier(pooled),torch.arange(3))
                loss += F.smooth_l1_loss(model.tumor_size_regressor(pooled),torch.ones(3,1))
                loss.backward()
                if step == 1:
                    for head in [model.classifier,model.tumor_size_regressor]:
                        for layer in head:
                            if isinstance(layer,torch.nn.Linear):
                                assert torch.isfinite(layer.weight.grad).all() and layer.weight.grad.abs().sum() > 0
                optimizer.step()
            configure_classification_classes(model,('benign','malignant'))
            assert model.classifier[-1].in_features == 128 and model.classifier[-1].out_features == 2
            del optimizer, outputs,classes,area,rec,hidden,loss
        results.append({'model':key,'parameters':spec.expected_parameters,'masked_patches':59})
        del model
        gc.collect()
    print(json.dumps({'passed':True,'variants':results,'weights_loaded':False},indent=2))

if __name__ == '__main__':
    main()
