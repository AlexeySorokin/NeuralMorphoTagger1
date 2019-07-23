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


class MaskedAccuracy:

    def __init__(self, pad=0, for_sequence=False, name="acc"):
        if isinstance(pad, float):
            pad = pad
        self.pad = [kb.constant(x, dtype="int64") for x in pad]
        self.for_sequence = for_sequence
        self.name = name
        if self.for_sequence:
            self.name = "sent_" + self.name

    def __call__(self, y_true, y_pred):
        y_true, y_pred = kb.argmax(y_true, axis=-1), kb.argmax(y_pred, axis=-1)
        are_equal = kb.cast(kb.equal(y_true, y_pred), kb.floatx())
        are_not_pad = kb.ones_like(y_true, dtype=kb.floatx())
        for x in self.pad:
            are_not_pad *= kb.cast(kb.not_equal(y_true, x), kb.floatx())
        are_equal *= are_not_pad
        if self.for_sequence:
            are_equal = kb.min(kb.cast(kb.equal(are_equal, are_not_pad), kb.floatx()), axis=-1)
            return are_equal
        else:
            are_equal_count = kb.sum(are_equal)
            are_not_pad_count = kb.sum(are_not_pad)
            return are_equal_count / are_not_pad_count

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