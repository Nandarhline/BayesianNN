import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers

class Pbnn:

    def __init__(self, config = {"output_dist": "Normal", "learn_all_params": True, "fixed_param": None}):
        self.n_infeatures = config["n_infeatures"] 
        self.n_outfeatures = config["n_outfeatures"] 
        self.n_samples = config["n_samples"]
        self.output_dist = config["output_dist"] if config.get("output_dist") is not None\
            else "Normal"
        self.learn_all_params = config["learn_all_params"] if config.get("learn_all_params") is not None\
            else True
        self.fixed_param = config["fixed_param"] if config.get("learn_all_params") is False\
            else None
        

    def out_dist(self, params):
        if self.output_dist == "Weibull":
            if self.learn_all_params is True:
                dist = tfp.distributions.Weibull(concentration=1e-3+ tf.math.softplus(0.05 *
                                                 params[:,self.n_outfeatures:2*self.n_outfeatures]), 
                                                 scale=1e-3+ tf.math.softplus(0.05 *
                                                 params[:,0:self.n_outfeatures]))
            else:
                dist = tfp.distributions.Weibull(concentration=self.fixed_param, 
                                                 scale=1e-3+ tf.math.softplus(0.05 *
                                                 params[:,0:self.n_outfeatures]))
        elif self.output_dist == "Normal":    
            if self.learn_all_params is True:
                dist = tfp.distributions.Normal(loc=params[:,0:self.n_outfeatures], 
                                                scale=1e-3+ tf.math.softplus(0.05 *
                                                params[:,self.n_outfeatures:2*self.n_outfeatures]))
            else:
                dist = tfp.distributions.Normal(loc=params[:,0:self.n_outfeatures], 
                                                scale=self.fixed_param)
        return dist
    
        
    def build_bnn(self, n_hidden_layers=3, width_hidden_layers=[16,32,16]):
        kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) / self.n_samples
        bias_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) / self.n_samples
        
        if self.learn_all_params is True:
            width_output_layer = 2*self.n_outfeatures
        else:
            width_output_layer = self.n_outfeatures
        
        inputs = tf.keras.layers.Input(shape=(self.n_infeatures,))
        features = tfp.layers.DenseFlipout(width_hidden_layers[0],
                      bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
                      bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                      kernel_divergence_fn=kernel_divergence_fn,
                      bias_divergence_fn=bias_divergence_fn,activation="relu")(inputs)
        if n_hidden_layers>1:
            for i in range(n_hidden_layers-1):
                features = tfp.layers.DenseFlipout(width_hidden_layers[i+1],
                      bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
                      bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                      kernel_divergence_fn=kernel_divergence_fn,
                      bias_divergence_fn=bias_divergence_fn,activation="relu")(features)
        params = tfp.layers.DenseFlipout(width_output_layer,
                      bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
                      bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                      kernel_divergence_fn=kernel_divergence_fn,
                      bias_divergence_fn=bias_divergence_fn)(features)
        dist = tfp.layers.DistributionLambda(self.out_dist)(params)
        self.model = Model(inputs=inputs, outputs=dist)
        return self.model.summary()
    
     
    def NLL(self, y, distr): 
        return -distr.log_prob(y) 
    
    
    def train_bnn(self, X, Y, train_env = {"optimizer": optimizers.Adam,
                                           "learning_rate": 0.001,
                                           "batch_size": 64,
                                           "epochs": 1000,
                                           "callback_patience": 30,
                                           "verbose": 0}):
        optimizer = train_env["optimizer"] if train_env.get("optimizer") is not None\
            else optimizers.Adam
        learning_rate = train_env["learning_rate"] if train_env.get("learning_rate") is not None\
            else 0.001
        batch_size = train_env["batch_size"] if train_env.get("batch_size") is not None\
            else 64
        epochs = train_env["epochs"] if train_env.get("epochs") is not None\
            else 1000
        callback_patience = train_env["callback_patience"] if train_env.get("callback_patience") is not None\
            else 30
        verbose = train_env["verbose"] if train_env.get("verbose") is not None\
            else 0
        self.model.compile(optimizer(learning_rate=learning_rate), loss=self.NLL)
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=callback_patience, 
                                                    verbose=0, mode='auto', baseline=None,
                                                    restore_best_weights=True)        
        history = self.model.fit(X, Y, batch_size=batch_size, epochs=epochs, verbose=verbose, 
                                 validation_split = 0, callbacks = callback)
        self.weights = self.model.get_weights()
        print('Training completed')
        print('Minimum loss: ', min(history.history['loss']))
        return
    
    
    def test_bnn(self, Xtest, nsim=100):
        Y = np.zeros([len(Xtest), self.n_outfeatures, nsim])
        for i in range(nsim):
            Y[:,:,i] = self.model.predict(Xtest, verbose=0)
            Mean_Y = np.mean(Y, axis=2)
            Stdv_Y = np.std(Y, axis=2)
        return Mean_Y, Stdv_Y
    
    
    def evaluate_bnn(self, Xtest, Ytest, nsim=100):
        if self.output_dist == "Normal":
            LL_Ytest = np.zeros([len(Xtest), self.n_outfeatures, nsim])
            for i in range(nsim):
                prediction_distribution = self.model(Xtest)
                LL_Ytest[:,:,i] = prediction_distribution.log_prob(Ytest)               
            Mean_LL = np.mean(LL_Ytest, axis=2)
        return Mean_LL


    def modeluq_bnn(self, Xtest, nsim=100):
        if self.output_dist == "Weibull":
            shapeY = np.zeros([len(Xtest), self.n_outfeatures, nsim])
            scaleY = np.zeros([len(Xtest), self.n_outfeatures, nsim])
            for i in range(nsim):
                Xtest = np.array(Xtest)
                prediction_distribution = self.model(Xtest)
                shapeY[:,:,i] = prediction_distribution.concentration.numpy()
                scaleY[:,:,i] = prediction_distribution.scale.numpy()
            Mean_shapeY = np.mean(shapeY, axis=2)
            Stdv_shapeY = np.std(shapeY, axis=2)
            Mean_scaleY = np.mean(scaleY, axis=2)
            Stdv_scaleY = np.std(scaleY, axis=2)
            return Mean_shapeY, Stdv_shapeY, Mean_scaleY, Stdv_scaleY
    
        elif self.output_dist == "Normal":
            muY = np.zeros([len(Xtest), self.n_outfeatures, nsim])
            sigmaY = np.zeros([len(Xtest), self.n_outfeatures, nsim])
            for i in range(nsim):
                Xtest = np.array(Xtest)
                prediction_distribution = self.model(Xtest)
                muY[:,:,i] = prediction_distribution.loc.numpy()
                sigmaY[:,:,i] = prediction_distribution.scale.numpy()
            Mean_muY = np.mean(muY, axis=2)
            Stdv_muY = np.std(muY, axis=2)
            Mean_sigmaY = np.mean(sigmaY, axis=2)
            Stdv_sigmaY = np.std(sigmaY, axis=2)
            return Mean_muY, Stdv_muY, Mean_sigmaY, Stdv_sigmaY
