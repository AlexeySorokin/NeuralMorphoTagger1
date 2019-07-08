from keras import backend as kb


class MulticlassSigmoidLoss:

    def __init__(self, alpha=1.0):
        self.alpha = kb.constant(alpha, dtype=kb.floatx())
        self.name = "multiclass_sigmoid_loss"

    def __call__(self, y_true, y_pred):
        y_pred = kb.clip(y_pred, kb.epsilon(), 1.0 - kb.epsilon())
        pos_loss = -y_true * kb.log(y_pred)
        neg_loss = -(1.0 - y_true) * kb.log(1.0 - y_pred)
        loss = self.alpha * pos_loss + neg_loss
        loss = kb.max(loss, axis=-1)
        return loss


class MulticlassSigmoidAccuracy:

    def __init__(self):
        self.name = "msa"

    def __call__(self, y_true, y_pred):
        y_true_positive = kb.greater(y_true, 0.5)
        y_pred_positive = kb.greater(y_pred, 0.5)
        all_equal = kb.min(kb.cast(kb.equal(y_true_positive, y_pred_positive), kb.floatx()), axis=-1)
        return all_equal


class MulticlassSoftmaxLoss:
    
    def __init__(self):
        pass
    
    def __call__(self, y_true, y_pred):
        y_pred = kb.clip(y_pred, kb.epsilon(), 1.0 - kb.epsilon())
        loss = -kb.log(kb.sum(y_true * y_pred, axis=-1))
        return loss

class MulticlassSoftmaxAccuracy:

    def __init__(self):
        self.name = "msoa"

    def __call__(self, y_true, y_pred):
        y_pred_max = kb.max(y_pred, axis=-1)
        y_pred_is_max = kb.cast(kb.equal(y_pred, kb.expand_dims(y_pred_max, -1)), kb.floatx())
        return kb.max(y_true * y_pred_is_max, axis=-1)