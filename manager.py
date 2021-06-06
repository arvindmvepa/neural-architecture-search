import numpy as np

from tensorflow.compat.v1.keras.models import Model
from tensorflow.compat.v1.keras import backend as K
from tensorflow.compat.v1.keras.callbacks import ModelCheckpoint
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class NetworkManager:
    '''
    Helper class to manage the generation of subnetwork training given a dataset
    '''
    def __init__(self, dataset, image_dim, epochs=5, acc_beta=0.8, clip_rewards=0.0):
        '''
        Manager which is tasked with creating subnetworks, training them on a dataset, and retrieving
        rewards in the term of accuracy, which is passed to the controller RNN.

        Args:
            dataset: a tuple of 2 generators (train_gen, val_gen)
            epochs: number of epochs to train the subnetworks
            child_batchsize: batchsize of training the subnetworks
            acc_beta: exponential weight for the accuracy
            clip_rewards: float - to clip rewards in [-range, range] to prevent
                large weight updates. Use when training is highly unstable.
        '''
        self.dataset = dataset
        self.image_dim = image_dim
        self.epochs = epochs
        self.clip_rewards = clip_rewards

        self.beta = acc_beta
        self.beta_bias = acc_beta
        self.moving_auc = 0.0

    def get_rewards(self, model_fn, actions):
        '''
        Creates a subnetwork given the actions predicted by the controller RNN,
        trains it on the provided dataset, and then returns a reward.

        Args:
            model_fn: a function which accepts one argument, a list of
                parsed actions, obtained via an inverse mapping from the
                StateSpace.
            actions: a list of parsed actions obtained via an inverse mapping
                from the StateSpace. It is in a specific order as given below:

                Consider 4 states were added to the StateSpace via the `add_state`
                method. Then the `actions` array will be of length 4, with the
                values of those states in the order that they were added.

                If number of layers is greater than one, then the `actions` array
                will be of length `4 * number of layers` (in the above scenario).
                The index from [0:4] will be for layer 0, from [4:8] for layer 1,
                etc for the number of layers.

                These action values are for direct use in the construction of models.

        Returns:
            a reward for training a model with the given actions
        '''
        with tf.Session(graph=tf.Graph()) as network_sess:
            K.set_session(network_sess)

            # generate a submodel given predicted actions
            model = model_fn(actions, self.image_dim)  # type: Model
            model.compile('adam', 'categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])

            # unpack the dataset
            train_gen, val1_gen, val2_gen = self.dataset

            # train the model using Keras methods
            model.fit(train_gen, epochs=self.epochs, verbose=1, validation_data=val1_gen,
                      callbacks=[ModelCheckpoint('weights/temp_network.h5',
                                                 monitor='val_auc', mode='max',
                                                 verbose=1, save_best_only=True,
                                                 save_weights_only=True)])

            # load best performance epoch in this training session
            model.load_weights('weights/temp_network.h5')

            # evaluate the model
            loss, acc, auc = model.evaluate(val2_gen)

            # compute the reward
            reward = (acc - self.moving_auc)

            # if rewards are clipped, clip them in the range -0.05 to 0.05
            if self.clip_rewards:
                reward = np.clip(reward, -0.05, 0.05)

            # update moving accuracy with bias correction for 1st update
            if self.beta > 0.0 and self.beta < 1.0:
                self.moving_auc = self.beta * self.moving_auc + (1 - self.beta) * acc
                self.moving_auc = self.moving_auc / (1 - self.beta_bias)
                self.beta_bias = 0

                reward = np.clip(reward, -0.1, 0.1)

            print()
            print("Manager: EWA Accuracy = ", self.moving_auc)

        # clean up resources and GPU memory
        network_sess.close()

        return reward, acc
