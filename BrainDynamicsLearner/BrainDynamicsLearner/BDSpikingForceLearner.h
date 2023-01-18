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
    bd_float_t dt = 0.04;
    bd_float_t BIAS = 1000.0;
    bd_float_t ff = 2.5;
    bd_float_t vr = -60.0;
    bd_float_t vt = -40.0;
    bd_float_t C = 250.0;
    bd_float_t a = 0.002;
    bd_float_t b = 0.0;
    bd_float_t vpeak = 30.0;
    bd_float_t vreset = -65.0;
    bd_float_t d = 100.0;
    bd_float_t tr = 2.0;
    bd_float_t td = 20.0;
    bd_float_t exp_neg_dt = exp(-dt / tr);
    bd_float_t one_over_tr_td = 1 / (tr * td);
    // Network property constants
    bd_size_t num_neurons = 1000;
    bd_float_t p = 0.1;// connection density of reservoir network
    bd_float_t G = 5000;// global weighting factor of reservoir connections
    BDMatrix reservoir_weights;
    bd_size_t num_predictions;
    bd_size_t num_context_cues;
    BDArray Q;// one-per-input weighting factors for prediction inputs
    BDArray WE2;// one-per-input weighting factors for context inputs
    // Fixed network weight matrices
    BDMatrix prediction_input_weights;
    BDMatrix context_input_weights;
    // Our predicted value of the time series.
    BDVector prediction;
    // Neuron state vectors
    BDArray IPSC;
    BDArray v;
    BDArray u;
    BDArray h;
    BDArray hr;
    BDVector r;
    // State matrices used in FORCE training.
    BDMatrix Pinv1;
    BDMatrix output_weights;
public:
    BDSpikingForceLearner(BDVector prediction_scaling_factors, BDVector context_scaling_factors);
    void neuronSimStep(BDVector context);
    void recursiveLeastSquaresStep(BDVector correct_output);
    BDVector getPrediction();
};
