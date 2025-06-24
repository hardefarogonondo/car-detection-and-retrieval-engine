from torchvision.ops import batched_nms
import math
import torch
import torch.nn.functional as F

def generate_default_boxes():
    fmap_dims = {"conv4_3": 38, "conv7": 19, "conv8_2": 10, "conv9_2": 5, "conv10_2": 3, "conv11_2": 1}
    obj_scales = {"conv4_3": 0.1, "conv7": 0.2, "conv8_2": 0.375, "conv9_2": 0.55, "conv10_2": 0.725, "conv11_2": 0.9}
    aspect_ratios = {
        "conv4_3": [1., 2., 0.5], "conv7": [1., 2., 3., 0.5, 0.333], "conv8_2": [1., 2., 3., 0.5, 0.333],
        "conv9_2": [1., 2., 3., 0.5, 0.333], "conv10_2": [1., 2., 0.5], "conv11_2": [1., 2., 0.5]
    }
    default_boxes = []
    fmaps = list(fmap_dims.keys())
    for k, fmap in enumerate(fmaps):
        for i in range(fmap_dims[fmap]):
            for j in range(fmap_dims[fmap]):
                cx, cy = (j + 0.5) / fmap_dims[fmap], (i + 0.5) / fmap_dims[fmap]
                for ratio in aspect_ratios[fmap]:
                    default_boxes.append([cx, cy, obj_scales[fmap] * math.sqrt(ratio), obj_scales[fmap] / math.sqrt(ratio)])
                    if ratio == 1.:
                        try:
                            additional_scale = math.sqrt(obj_scales[fmap] * obj_scales[fmaps[k+1]])
                        except IndexError:
                            additional_scale = 1.
                        default_boxes.append([cx, cy, additional_scale, additional_scale])
    default_boxes = torch.tensor(default_boxes, dtype=torch.float32)
    default_boxes.clamp_(0, 1)
    return default_boxes

def post_process_predictions(predicted_locs, predicted_scores, default_boxes, num_classes, img_size, conf_thresh=0.5, nms_thresh=0.45):
    batch_size = predicted_locs.size(0)
    predicted_scores = F.softmax(predicted_scores, dim=2)
    db_cx, db_cy, db_w, db_h = default_boxes[:, 0], default_boxes[:, 1], default_boxes[:, 2], default_boxes[:, 3]
    pred_cx = db_cx.unsqueeze(0) + predicted_locs[:, :, 0] * 0.1 * db_w.unsqueeze(0)
    pred_cy = db_cy.unsqueeze(0) + predicted_locs[:, :, 1] * 0.1 * db_h.unsqueeze(0)
    pred_w = db_w.unsqueeze(0) * torch.exp(predicted_locs[:, :, 2] * 0.2)
    pred_h = db_h.unsqueeze(0) * torch.exp(predicted_locs[:, :, 3] * 0.2)
    pred_boxes_xmin = (pred_cx - pred_w / 2) * img_size
    pred_boxes_ymin = (pred_cy - pred_h / 2) * img_size
    pred_boxes_xmax = (pred_cx + pred_w / 2) * img_size
    pred_boxes_ymax = (pred_cy + pred_h / 2) * img_size
    pred_boxes = torch.stack([pred_boxes_xmin, pred_boxes_ymin, pred_boxes_xmax, pred_boxes_ymax], dim=2)
    all_batch_preds = []
    for i in range(batch_size):
        scores = predicted_scores[i][:, 1:]
        boxes = pred_boxes[i]
        conf_scores, conf_labels = torch.max(scores, dim=1)
        mask = conf_scores > conf_thresh
        if mask.sum() == 0:
            all_batch_preds.append({"boxes": torch.empty(0, 4), "scores": torch.empty(0), "labels": torch.empty(0, dtype=torch.int64)})
            continue
        final_boxes, final_scores, final_labels = boxes[mask], conf_scores[mask], conf_labels[mask] + 1
        indices_to_keep = batched_nms(final_boxes, final_scores, final_labels.float(), nms_thresh)
        all_batch_preds.append({
            "boxes": final_boxes[indices_to_keep],
            "scores": final_scores[indices_to_keep],
            "labels": final_labels[indices_to_keep],
        })
    return all_batch_preds