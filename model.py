#from tensorflow.compat.v1.keras.models import Model
from tensorflow.compat.v1.keras.layers import Input, Dense, Conv2D, GlobalAveragePooling2D
from tensorflow.compat.v1.keras.applications import DenseNet121

# generic model design
def model_fn(actions, image_dim):
    """
    # comment out generally
    # unpack the actions from the list
    kernel_1, filters_1, kernel_2, filters_2, kernel_3, filters_3, kernel_4, filters_4 = actions
    x = Conv2D(filters_1, (kernel_1, kernel_1), strides=(2, 2), padding='same', activation='relu')(ip)
    x = Conv2D(filters_2, (kernel_2, kernel_2), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(filters_3, (kernel_3, kernel_3), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(filters_4, (kernel_4, kernel_4), strides=(1, 1), padding='same', activation='relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(2, activation='softmax')(x)
    #x = Dense(10, activation='softmax')(x)

    model = Model(ip, x)
    """
    ip = Input(shape=image_dim)
    model = DenseNet121(include_top=True,
                        weights="imagenet",
                        input_tensor=ip,
                        input_shape=image_dim,
                        pooling=None,
                        classes=2)
    return model
