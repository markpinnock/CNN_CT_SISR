# import skimage.measure as sk
import tensorflow as tf
l

# def imgQualityMetrics(pred, hi_vol):
#     dims = pred.shape
#     pSNR = 0
#     SSIM = 0
#
#     for idx in list(range(dims[0])):
#         pSNR += sk.compare_psnr(hi_vol[idx, :, :, :, 0], pred[idx, :, :, :, 0])
#         SSIM += sk.compare_ssim(hi_vol[idx, :, :, :, 0], pred[idx, :, :, :, 0])
#
#     return pSNR, SSIM


@tf.function
def trainStep(lo_vol, hi_vol, Model, ModelOptimiser, model_loss, model_metric):
    with tf.GradientTape() as tape:
        pred = Model(lo_vol, training=True)
        losses = model_loss(pred, hi_vol)
        gradients = tape.gradient(losses, Model.trainable_variables)
        ModelOptimiser.apply_gradients(zip(gradients, Model.trainable_variables))
        model_metric.update_state(pred, hi_vol)


@tf.function
def valStep(lo_vol, hi_vol, Model, model_metric):
    pred = Model(lo_vol, training=False)  
    model_metric.update_state(pred, hi_vol)
