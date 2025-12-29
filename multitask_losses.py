"""
Custom Loss Functions for Multi-Task Pneumothorax Detection
Combines classification loss and bounding box localization loss
"""

import tensorflow as tf
import keras.backend as K


def focal_loss_classification(alpha=0.25, gamma=2.0):
    """
    Focal loss for binary classification.
    Addresses class imbalance by down-weighting easy examples.

    Args:
        alpha: Weighting factor for positive class (0.25 works well for imbalanced data)
        gamma: Focusing parameter (2.0 is standard)

    Returns:
        Loss function
    """
    def compute_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        # Compute focal loss
        cross_entropy = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = tf.pow(1 - pt, gamma)
        alpha_weight = y_true * alpha + (1 - y_true) * (1 - alpha)

        loss = alpha_weight * focal_weight * cross_entropy
        return tf.reduce_mean(loss)

    return compute_loss


def smooth_l1_loss_bbox(y_true, y_pred):
    """
    Smooth L1 loss for bounding box regression.
    Also known as Huber loss - less sensitive to outliers than L2 loss.

    Only computes loss for positive samples (where y_true bbox is not all zeros).

    Args:
        y_true: Ground truth bounding boxes [batch, 4]
        y_pred: Predicted bounding boxes [batch, 4]

    Returns:
        Smooth L1 loss
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Mask for positive samples (bbox is not all zeros)
    # Sum across bbox coordinates - if sum > 0, it's a positive sample
    positive_mask = tf.reduce_sum(y_true, axis=-1) > 0  # Shape: [batch]
    positive_mask = tf.cast(positive_mask, tf.float32)

    # Compute absolute difference
    diff = tf.abs(y_true - y_pred)

    # Smooth L1: L1 for large errors, L2 for small errors
    smooth_l1 = tf.where(diff < 1.0, 0.5 * tf.square(diff), diff - 0.5)

    # Sum across bbox coordinates
    smooth_l1 = tf.reduce_sum(smooth_l1, axis=-1)  # Shape: [batch]

    # Only apply loss to positive samples
    smooth_l1 = smooth_l1 * positive_mask

    # Average over positive samples only
    n_positive = tf.maximum(tf.reduce_sum(positive_mask), 1.0)  # Avoid division by zero
    loss = tf.reduce_sum(smooth_l1) / n_positive

    return loss


def iou_loss_bbox(y_true, y_pred):
    """
    IoU (Intersection over Union) loss for bounding box regression.
    Directly optimizes for IoU metric, which is better than L1/L2 for bbox.

    Only computes loss for positive samples.

    Args:
        y_true: Ground truth bounding boxes [batch, 4] in format [x_min, y_min, x_max, y_max]
        y_pred: Predicted bounding boxes [batch, 4]

    Returns:
        1 - IoU loss (so higher IoU = lower loss)
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Mask for positive samples
    positive_mask = tf.reduce_sum(y_true, axis=-1) > 0
    positive_mask = tf.cast(positive_mask, tf.float32)

    # Extract coordinates
    x1_true, y1_true, x2_true, y2_true = tf.split(y_true, 4, axis=-1)
    x1_pred, y1_pred, x2_pred, y2_pred = tf.split(y_pred, 4, axis=-1)

    # Compute intersection
    x1_inter = tf.maximum(x1_true, x1_pred)
    y1_inter = tf.maximum(y1_true, y1_pred)
    x2_inter = tf.minimum(x2_true, x2_pred)
    y2_inter = tf.minimum(y2_true, y2_pred)

    inter_width = tf.maximum(0.0, x2_inter - x1_inter)
    inter_height = tf.maximum(0.0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height

    # Compute union
    true_area = (x2_true - x1_true) * (y2_true - y1_true)
    pred_area = (x2_pred - x1_pred) * (y2_pred - y1_pred)
    union_area = true_area + pred_area - inter_area

    # Compute IoU
    iou = inter_area / (union_area + 1e-7)
    iou = tf.squeeze(iou, axis=-1)  # Shape: [batch]

    # Only apply to positive samples
    iou = iou * positive_mask

    # Average over positive samples
    n_positive = tf.maximum(tf.reduce_sum(positive_mask), 1.0)
    avg_iou = tf.reduce_sum(iou) / n_positive

    # Return 1 - IoU so lower IoU = higher loss
    return 1.0 - avg_iou


def combined_multitask_loss(classification_weight=1.0, bbox_weight=1.0,
                            use_focal=True, use_iou=True):
    """
    Combined loss for multi-task learning: classification + bounding box localization.

    Args:
        classification_weight: Weight for classification loss (default: 1.0)
        bbox_weight: Weight for bbox loss (default: 1.0)
        use_focal: Use focal loss for classification (recommended for imbalanced data)
        use_iou: Use IoU loss for bbox (recommended) vs Smooth L1

    Returns:
        Combined loss function
    """
    if use_focal:
        class_loss_fn = focal_loss_classification(alpha=0.25, gamma=2.0)
    else:
        class_loss_fn = tf.keras.losses.BinaryCrossentropy()

    if use_iou:
        bbox_loss_fn = iou_loss_bbox
    else:
        bbox_loss_fn = smooth_l1_loss_bbox

    def compute_loss(y_true, y_pred):
        """
        Args:
            y_true: List of [class_true, bbox_true]
            y_pred: List of [class_pred, bbox_pred]

        Returns:
            Combined weighted loss
        """
        class_true, bbox_true = y_true
        class_pred, bbox_pred = y_pred

        # Compute individual losses
        class_loss = class_loss_fn(class_true, class_pred)
        bbox_loss = bbox_loss_fn(bbox_true, bbox_pred)

        # Combine with weights
        total_loss = classification_weight * class_loss + bbox_weight * bbox_loss

        return total_loss

    return compute_loss


# Metric functions for tracking during training
def classification_accuracy(y_true, y_pred):
    """
    Binary classification accuracy metric.
    """
    # y_true and y_pred are the classification outputs only
    y_true = tf.cast(y_true > 0.5, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    return tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))


def bbox_iou_metric(y_true, y_pred):
    """
    Average IoU for bounding boxes (only on positive samples).
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Mask for positive samples
    positive_mask = tf.reduce_sum(y_true, axis=-1) > 0
    positive_mask = tf.cast(positive_mask, tf.float32)

    # Extract coordinates
    x1_true, y1_true, x2_true, y2_true = tf.split(y_true, 4, axis=-1)
    x1_pred, y1_pred, x2_pred, y2_pred = tf.split(y_pred, 4, axis=-1)

    # Compute intersection
    x1_inter = tf.maximum(x1_true, x1_pred)
    y1_inter = tf.maximum(y1_true, y1_pred)
    x2_inter = tf.minimum(x2_true, x2_pred)
    y2_inter = tf.minimum(y2_true, y2_pred)

    inter_width = tf.maximum(0.0, x2_inter - x1_inter)
    inter_height = tf.maximum(0.0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height

    # Compute union
    true_area = (x2_true - x1_true) * (y2_true - y1_true)
    pred_area = (x2_pred - x1_pred) * (y2_pred - y1_pred)
    union_area = true_area + pred_area - inter_area

    # Compute IoU
    iou = inter_area / (union_area + 1e-7)
    iou = tf.squeeze(iou, axis=-1)

    # Only compute for positive samples
    iou = iou * positive_mask
    n_positive = tf.maximum(tf.reduce_sum(positive_mask), 1.0)

    return tf.reduce_sum(iou) / n_positive
