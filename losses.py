import tensorflow.keras.backend as K


def clip(z):
    """ Clip all values in a tensor to prevent divide by 0 errors. """
    z = K.clip(z, 1e-7, 1)
    return z


def weighted_pixel_bce_loss(beta, batch_size):
    """
    Weighted pixel-wise binary cross entropy loss function, averaged over the batch.
    Parameters:
    beta: weighting factor. Set this to be the average proportion of class = 1 pixels in a training image
    batch_size: number of training examples in the batch
    """

    def compute_loss(y_true, y_pred):
        # Assign a greater loss to false negative predictions to prevent model always predicting y = 0 for all pixels
        px_wt = 1. / beta
        # Find the number of total pixels in an image
        num_pix = K.int_shape(y_pred)[1] * K.int_shape(y_pred)[2]
        # Calculate the loss
        bce = - ((px_wt * y_true * K.log(clip(y_pred))) + (1 - y_true) * K.log(clip(1 - y_pred)))

        # Sum and average the loss by the number of pixels and by the batch size
        loss = (K.sum(bce) / num_pix) / batch_size

        return loss

    return compute_loss


def dice_loss():
    """
    Computes the dice loss for a predicted segmentation.
    """

    def compute_loss(y_true, y_pred):
        # Compute dice coefficient and return loss
        numerator = 2 * K.sum(y_true * y_pred)
        denominator = K.sum(y_true ** 2) + K.sum(y_pred ** 2)
        dice_coefficient = numerator / denominator
        loss = 1 - dice_coefficient

        return loss

    return compute_loss


def combined_dice_wpce_loss(beta, batch_size):
    """
    A combination of 2x dice loss and 1x weighted pixel-wise binary cross entropy loss for a prediction.
    Tends to result in better performance than wpce or dice alone.
    Parameters:
    beta: weighting factor. Set this to be the average proportion of class = 1 pixels in a training image
    batch_size: number of training examples in the batch
    """

    def compute_loss(y_true, y_pred):
        # Get weighted pixel cross entropy loss
        # Assign a greater loss to false negative predictions to prevent model always predicting y = 0 for all pixels
        px_wt = 1. / beta
        # Find the number of total pixels in an image
        num_pix = K.int_shape(y_pred)[1] * K.int_shape(y_pred)[2]
        bce = - ((px_wt * y_true * K.log(clip(y_pred))) + (1 - y_true) * K.log(clip(1 - y_pred)))

        # Sum and average the loss by the number of pixels and by the batch size
        wpce_loss = (K.sum(bce) / num_pix) / batch_size

        # Compute dice coefficient and return loss
        numerator = 2 * K.sum(y_true * y_pred)
        denominator = K.sum(y_true ** 2) + K.sum(y_pred ** 2)
        dice_coefficient = numerator / denominator
        _dice_loss = 1 - dice_coefficient

        # The final loss value is a mix of both loss functions
        loss = 2 * _dice_loss + wpce_loss

        return loss

    return compute_loss


def focal_loss(alpha=0.25, gamma=2.0):
    """
    Focal Loss for binary classification.
    Focuses training on hard examples and down-weights easy examples.
    Excellent for imbalanced datasets where positive samples are rare.

    Parameters:
    alpha: weighting factor for positive class (default 0.25 for imbalanced data)
    gamma: focusing parameter (default 2.0). Higher gamma = more focus on hard examples

    Reference: Lin et al., "Focal Loss for Dense Object Detection"
    https://arxiv.org/abs/1708.02002
    """

    def compute_loss(y_true, y_pred):
        # Clip predictions to prevent log(0)
        y_pred = clip(y_pred)

        # Calculate focal loss components
        # For positive samples: -alpha * (1 - p)^gamma * log(p)
        # For negative samples: -(1 - alpha) * p^gamma * log(1 - p)
        cross_entropy = - (y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))

        # Compute the focal weight: (1 - pt)^gamma
        # pt is the predicted probability for the true class
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = K.pow(1 - pt, gamma)

        # Apply alpha weighting
        alpha_weight = y_true * alpha + (1 - y_true) * (1 - alpha)

        # Combine all components
        loss = K.mean(alpha_weight * focal_weight * cross_entropy)

        return loss

    return compute_loss


def focal_tversky_loss(alpha=0.7, gamma=0.75):
    """
    Focal Tversky Loss - excellent for segmentation with severe class imbalance.
    Combines Tversky index (generalization of Dice) with focal weighting.
    Better at detecting small pneumothorax regions.

    Parameters:
    alpha: weight for false positives vs false negatives (0.7 = prioritize recall)
    gamma: focal parameter (0.75 works well for medical segmentation)

    Reference: Abraham & Khan, "A Novel Focal Tversky Loss Function"
    """

    def compute_loss(y_true, y_pred):
        # Flatten the tensors
        y_true_flat = K.flatten(y_true)
        y_pred_flat = K.flatten(y_pred)

        # Calculate Tversky components
        true_pos = K.sum(y_true_flat * y_pred_flat)
        false_neg = K.sum(y_true_flat * (1 - y_pred_flat))
        false_pos = K.sum((1 - y_true_flat) * y_pred_flat)

        # Tversky Index
        tversky_index = (true_pos + 1e-7) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + 1e-7)

        # Apply focal weighting
        focal_tversky = K.pow(1 - tversky_index, gamma)

        return focal_tversky

    return compute_loss


def combined_dice_focal_loss(alpha=0.25, gamma=2.0, dice_weight=2.0, focal_weight=1.0):
    """
    Combined Dice + Focal Loss for segmentation.
    Recommended for pneumothorax detection with high recall priority.

    Parameters:
    alpha: focal loss alpha parameter (default 0.25)
    gamma: focal loss gamma parameter (default 2.0)
    dice_weight: weight for dice loss component (default 2.0)
    focal_weight: weight for focal loss component (default 1.0)
    """

    def compute_loss(y_true, y_pred):
        # Compute dice loss
        numerator = 2 * K.sum(y_true * y_pred)
        denominator = K.sum(y_true ** 2) + K.sum(y_pred ** 2)
        dice_coefficient = numerator / (denominator + 1e-7)
        _dice_loss = 1 - dice_coefficient

        # Compute focal loss
        y_pred_clipped = clip(y_pred)
        cross_entropy = - (y_true * K.log(y_pred_clipped) + (1 - y_true) * K.log(1 - y_pred_clipped))
        pt = y_true * y_pred_clipped + (1 - y_true) * (1 - y_pred_clipped)
        focal_weight_term = K.pow(1 - pt, gamma)
        alpha_weight = y_true * alpha + (1 - y_true) * (1 - alpha)
        _focal_loss = K.mean(alpha_weight * focal_weight_term * cross_entropy)

        # Combine losses
        loss = dice_weight * _dice_loss + focal_weight * _focal_loss

        return loss

    return compute_loss
