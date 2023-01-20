// #pragma once
// Use FORCE training to teach a time series to a reservoir computer with spiking neurons.
#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include <random>

typedef double bd_float_t;
typedef size_t bd_size_t;
typedef Eigen::ArrayXd BDArray;
typedef Eigen::VectorXd BDVector;
typedef Eigen::MatrixXd BDMatrix;

class BDSpikingForceLearner
{
protected:
    // Izhikevich neuron model constants
    bd_float_t dt = 0.04;// ms, Euler integration step size
    bd_float_t I_bias = 1000.0;// pA, constant bias current
    bd_float_t C = 250.0;// microF, membrane capacitance
    bd_float_t k = 2.5;// ns/mV, scaling factor of action potential half-width
    bd_float_t v_peak = 30.0;// mV, peak voltage, maximum v attained during a spike
    bd_float_t v_reset = -65.0;// mV, reset potential, v becomes this after a spike
    bd_float_t v_resting = -60.0;// mV, resting v
    bd_float_t v_threshold = -40.0;// mV, threshold v (when b=0 and I_bias=0)
    bd_float_t a = 0.002;// ms^-1, reciprocal of u time constant
    // If we set b to something non-0, go back into the code and swap back in the version of the u and v updates that uses b.
    // bd_float_t b = 0.0;// nS, sensitivity of u to subthreshold oscillations of v
    bd_float_t d = 100.0;// pA, incremental increase in u after each spike
    bd_float_t tau_d = 20.0;// ms, synaptic decay time
    bd_float_t tau_r = 2.0;// ms, synaptic rise time
    // derived constants we can calculate once
    bd_float_t dt_over_C = dt / C;
    bd_float_t dt_a = dt * a;
    bd_float_t exp_neg_dt_over_tau_d = exp(-dt / tau_d);
    bd_float_t exp_neg_dt_over_tau_r = exp(-dt / tau_r);
    bd_float_t one_over_tr_td = 1 / (tau_r * tau_d);
    // Network property constants
    bd_size_t num_neurons = 1000;// number of neurons in reservoir network
    bd_float_t p = 0.1;// connection density of reservoir network
    bd_float_t G = 5000;// global weighting factor of reservoir connections
    BDMatrix reservoir_weights;// static weights of reservoir network
    bd_size_t num_predictions;// dimensionality of time series to learn
    bd_size_t num_context_cues;// dimensionality of context cue input
    BDArray Q_prediction;// one-per-input weighting factors for prediction inputs
    BDArray Q_context;// one-per-input weighting factors for context inputs
    // Fixed network weight matrices
    BDMatrix prediction_input_weights;// static encoding weights of prediction fed back into network
    BDMatrix context_input_weights;// static encoding weights of context cues fed back into network
    // Neuron state vectors
    BDArray I_synapse;// pA, post-synaptic current
    BDArray v;// mV, membrane potential
    BDArray u;// pA, adaptation current
    BDArray h;// pA/ms, synaptic current gating variable?
    BDArray hr;// pA/ms, output current gating variable?
    BDVector r;// pA, network output before transformation by output weights
    // State matrices used in FORCE training.
    BDMatrix P;// "network estimate of the inverse of the correlation matrix" according to the paper
    BDMatrix output_weights;// output weights used to generate prediction from r
    BDVector prediction;// predicted value of the time series.
public:
    BDSpikingForceLearner(BDVector prediction_scaling_factors, BDVector context_scaling_factors);
    void neuronSimStep(BDVector context);
    void recursiveLeastSquaresStep(BDVector correct_output);
    BDVector getPrediction();
};
