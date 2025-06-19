from keras import ops

# Double casting is done here in order to work in any setup with any backend as some are
# more strict than others - you may want to remove it if you are sure of your setup

def quadratic_vwap_loss(y_true, y_pred):
    y_true = ops.cast(y_true, y_pred.dtype)
    y_pred = ops.cast(y_pred, y_true.dtype)
    vwap_achieved = ops.sum(y_pred[..., 0] * y_true[..., 1], axis = 1) / ops.sum(y_pred[..., 0], axis = 1)
    vwap_mkt = ops.sum(y_true[..., 0] * y_true[..., 1], axis = 1) / ops.sum(y_true[..., 0], axis = 1)
    vwap_diff = vwap_achieved / vwap_mkt - 1.
    loss = ops.mean(ops.square(vwap_diff))
    return loss

def absolute_vwap_loss(y_true, y_pred):
    y_true = ops.cast(y_true, y_pred.dtype)
    y_pred = ops.cast(y_pred, y_true.dtype)
    vwap_achieved = ops.sum(y_pred[..., 0] * y_true[..., 1], axis = 1) / ops.sum(y_pred[..., 0], axis = 1)
    vwap_mkt = ops.sum(y_true[..., 0] * y_true[..., 1], axis = 1) / ops.sum(y_true[..., 0], axis = 1)
    vwap_diff = vwap_achieved / vwap_mkt - 1.
    loss = ops.mean(ops.abs(vwap_diff))
    return loss

def volume_curve_loss(y_true, y_pred):
    y_true = ops.cast(y_true, y_pred.dtype)
    y_pred = ops.cast(y_pred, y_true.dtype)
    volume_curve_achieved = y_pred[..., 0] / ops.sum(y_pred[..., 0], axis = 1, keepdims=True)
    volume_curve_mkt = y_true[..., 0] / ops.sum(y_true[..., 0], axis = 1, keepdims=True)
    volume_curve_diff = volume_curve_achieved - volume_curve_mkt
    loss = ops.mean(ops.square(volume_curve_diff))
    return loss

def impact_block(x, lam=1.0):
    return 0.5 * lam * ops.square(x)

def impact_exp(x, theta=1.0):
    return theta * ((1 - x) * ops.log(1 - x + 1e-6) + x)

def impact_power(x, alpha=1.0):
    return ((alpha + 1) / (alpha + 2)) * ops.power(x, (alpha + 2) / (alpha + 1))

def vwap_afs_loss(y_true, y_pred):
    y_true = ops.cast(y_true, y_pred.dtype)
    y_pred = ops.cast(y_pred, y_true.dtype)

    # VWAP Loss
    pred_v, true_v, true_p = y_pred[..., 0], y_true[..., 0], y_true[..., 1]
    vwap_achieved = ops.sum(pred_v * true_p, axis=1) / ops.sum(pred_v, axis=1, keepdims=True)
    vwap_mkt = ops.sum(true_v * true_p, axis=1) / ops.sum(true_v, axis=1, keepdims=True)
    vwap_diff = vwap_achieved / vwap_mkt - 1.
    vwap_loss = ops.mean(ops.square(vwap_diff))
    
    impact_loss = ops.mean(impact_power(ops.cumsum(pred_v)))

    return vwap_loss + impact_loss