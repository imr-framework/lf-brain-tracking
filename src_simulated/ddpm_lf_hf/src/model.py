from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def ResBlock(x, ch):
    h = Conv2D(ch, 3, padding='same')(x)
    h = BatchNormalization()(h)
    h = Activation('swish')(h)

    h = Conv2D(ch, 3, padding='same')(h)
    h = BatchNormalization()(h)

    if x.shape[-1] != ch:
        x = Conv2D(ch, 1, padding='same')(x)

    return Activation('swish')(x + h)

def Down(x, ch):
    x = Conv2D(ch, 3, strides=2, padding='same')(x)
    return Activation('swish')(x)

def Up(x, ch):
    x = UpSampling2D()(x)
    x = Conv2D(ch, 3, padding='same')(x)
    return Activation('swish')(x)

def DiffusionUNet(shape):
    xt = Input(shape)
    lf = Input(shape)

    x = Concatenate()([xt, lf])

    d1 = ResBlock(x, 64)
    d1d = Down(d1, 128)

    d2 = ResBlock(d1d, 128)
    d2d = Down(d2, 256)

    d3 = ResBlock(d2d, 256)
    d3d = Down(d3, 512)

    # add more ResBlocks in the bottleneck if needed
    mid = ResBlock(d3d, 512)

    u2 = Up(mid, 256)
    u2 = ResBlock(Concatenate()([u2, d3]), 256)

    u1 = Up(u2, 128)
    u1 = ResBlock(Concatenate()([u1, d2]), 128)

    u0 = Up(u1, 64)
    u0 = ResBlock(Concatenate()([u0, d1]), 64)

    out = Conv2D(3, 1)(u0)
    return Model([xt, lf], out)

# run a quick test
if __name__ == "__main__":
    model = DiffusionUNet((128, 128, 3))
    model.summary()
