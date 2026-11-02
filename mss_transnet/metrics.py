"""Lesion-only segmentation and macro classification reporting."""
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

METRICS = ("dice","iou","precision","sensitivity","specificity","pixel_accuracy")


def segmentation_metrics(logits, target):
    pred = logits.sigmoid()>=.5;truth = target>=.5
    axes = (1,2,3)
    tp = (pred&truth).sum(axes).float();tn = (~pred&~truth).sum(axes).float()
    fp = (pred&~truth).sum(axes).float();fn = (~pred&truth).sum(axes).float()
    def ratio(n,d,empty=0.):return torch.where(d>0,n/d.clamp_min(1),empty)
    return dict(dice=ratio(2*tp,2*tp+fp+fn,1.),iou=ratio(tp,tp+fp+fn,1.),
        precision=ratio(tp,tp+fp,0.),sensitivity=ratio(tp,tp+fn,0.),
        specificity=ratio(tn,tn+fp,0.),pixel_accuracy=ratio(tp+tn,tp+tn+fp+fn))


def segmentation_summary(rows):
    lesion = [r for r in rows if r["class_name"]!="normal"]
    if not lesion:raise ValueError("No lesion images available for reported segmentation")
    return dict(images=len(lesion),normal_images_excluded=len(rows)-len(lesion),
        mean={k:float(np.mean([float(r[k]) for r in lesion])) for k in METRICS},
        image_sd={k:float(np.std([float(r[k]) for r in lesion],ddof=1)) if len(lesion)>1 else 0.
                  for k in METRICS})


def classification_summary(rows, classes):
    truth = np.array([int(r["true_index"]) for r in rows])
    probability = np.array([[float(r["probability_"+c]) for c in classes] for r in rows])
    pred = probability.argmax(1)
    cm = confusion_matrix(truth,pred,labels=np.arange(len(classes)))
    tp = np.diag(cm).astype(float);fn = cm.sum(1)-tp;fp = cm.sum(0)-tp;tn = cm.sum()-tp-fn-fp
    def div(n,d):return np.divide(n,d,out=np.zeros_like(n,dtype=float),where=d!=0)
    precision,recall = div(tp,tp+fp),div(tp,tp+fn)
    result = dict(images=len(rows),accuracy=float((pred==truth).mean()),
        macro_precision=float(precision.mean()),macro_recall=float(recall.mean()),
        macro_specificity=float(div(tn,tn+fp).mean()),macro_f1=float(div(2*tp,2*tp+fp+fn).mean()),
        classes=list(classes),confusion_counts=cm.tolist(),
        confusion_row_percent=(div(cm.astype(float),cm.sum(1,keepdims=True))*100).tolist())
    malignant = classes.index("malignant")
    binary = truth==malignant
    if len(np.unique(binary))==2:
        fpr,tpr,_ = roc_curve(binary,probability[:,malignant])
        result.update(malignant_auc=float(roc_auc_score(binary,probability[:,malignant])),
                      roc_fpr=fpr.tolist(),roc_tpr=tpr.tolist())
    return result
