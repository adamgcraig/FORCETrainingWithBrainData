#include "BDSpikingForceLearner.h"

BDSpikingForceLearner::BDSpikingForceLearner(BDVector prediction_scaling_factors, BDVector context_scaling_factors)
{
    num_predictions = prediction_scaling_factors.size();
    num_context_cues = context_scaling_factors.size();
    Q = prediction_scaling_factors.array();// one-per-input weighting factors
    WE2 = context_scaling_factors.array();// one-per-input weighting factors
    // Fixed network weight matrices
    prediction_input_weights = Eigen::MatrixXd::Random(num_neurons, num_predictions).array() * Q.transpose().replicate(num_neurons,1).array();
    // std::cout << "prediction input weights: " << prediction_input_weights << std::endl;
    context_input_weights = Eigen::MatrixXd::Random(num_neurons, num_context_cues).array() * WE2.transpose().replicate(num_neurons, 1).array();
    // std::cout << "context input weights: " << context_input_weights << std::endl;
    reservoir_weights = G * ( Eigen::MatrixXd::Random(num_neurons, num_neurons).array().abs() < p).select(Eigen::MatrixXd::Random(num_neurons, num_neurons), Eigen::MatrixXd::Constant(num_neurons, num_neurons, 0.0));
    std::cout << "reservoir weights: " << reservoir_weights << std::endl;
    // Neuron state vectors
    IPSC = Eigen::VectorXd::Constant(num_neurons, 0.0);
    prediction = Eigen::VectorXd::Constant(num_predictions, 0.0);
    v = vr + (vpeak - vr)*( Eigen::ArrayXd::Random(num_neurons) + 1 )/2;
    u = Eigen::ArrayXd::Constant(num_neurons, 0.0);
    h = Eigen::ArrayXd::Constant(num_neurons, 0.0);
    r = Eigen::ArrayXd::Constant(num_neurons, 0.0);
    hr = Eigen::ArrayXd::Constant(num_neurons, 0.0);
    // State matrices used in FORCE training.
    Pinv1 = Eigen::MatrixXd::Identity(num_neurons, num_neurons);
    output_weights = Eigen::MatrixXd::Constant(num_predictions, num_neurons, 0.0);
}

void BDSpikingForceLearner::neuronSimStep(BDVector context)
{
    BDMatrix weighted_input = prediction_input_weights * prediction + context_input_weights * context;
    std::cout << "weighted input: " << weighted_input.transpose() << std::endl;
    // After this, we are mostly in element-wise operation mode, as we update the state of each neuron independently.
    BDArray I = IPSC + weighted_input.array() + BIAS; // IPSC + E1 * z1 + E2_z2_qq + BIAS;
    // std::cout << "I: " << I.transpose() << std::endl;
    BDArray v_ = v; // Set v(t - 1) = v so that we can still use this value after updating v to v(t).
    v += dt * ( ff * (v_-vr) * (v_-vt) - u + I ) / C; // v(t) = v(t - 1) + dt * v'(t-1)
    u += dt * (  a * ( b * (v_-vr) - u )  ); // same with u, the v_ term makes it so that the integration of u uses v(t - 1), instead of the updated v(t)
    Eigen::Array<bool,Eigen::Dynamic,1> is_spike = v >= vpeak;
    std::cout << "is_spike: " << is_spike.transpose() << std::endl;
    v = is_spike.select(vreset,v); // implements v = c if v > vpeak add 0 if false, add c - v if true, v + c - v = c
    // std::cout << "v: " << v.transpose() << std::endl;
    u = is_spike.select(u+d,u); // implements set u to u + d if v > vpeak, component by component.
    // std::cout << "u: " << u.transpose() << std::endl;
    // We briefly work in matrix operation mode here when transmitting spikes between neurons.
    BDVector is_spike_double = (BDVector) is_spike.cast<double>();
    BDVector JD = reservoir_weights * is_spike_double; // Compute the increase in current due to spiking.
                                                           // Just implement the double-exponential synapse so that we do not need an if-statement every step.
    // std::cout << "JD: " << JD.transpose() << std::endl;
    IPSC = exp_neg_dt * IPSC + dt * h;
    // std::cout << "IPSC: " << IPSC.transpose() << std::endl;
    r = exp_neg_dt * r.array() + dt * hr;
    std::cout << "r: " << r.transpose() << std::endl;
    h = exp_neg_dt * h + one_over_tr_td * JD.array();
    // std::cout << "h: " << h.transpose() << std::endl;
    hr = exp_neg_dt * hr + one_over_tr_td * is_spike_double.array();
    // std::cout << "hr: " << hr.transpose() << std::endl;
    // We use matrices one more time when we compute the output of this time step, which will serve as the input of the next.
    prediction = output_weights * r;
}

void BDSpikingForceLearner::recursiveLeastSquaresStep(BDVector correct_result)
{
    BDVector error_value = prediction - correct_result;
    std::cout << "error_value: " << error_value.transpose() << std::endl;
    BDVector cd1 = Pinv1 * r;
    // std::cout << "cd1: " << cd1.transpose() << std::endl;
    Eigen::RowVectorXd cd1_t = cd1.transpose();
    bd_float_t cd1_dot_r = cd1.dot(r);
    if (cd1_dot_r != 0)
    {
        output_weights -= error_value * cd1_t;
        // std::cout << "output weights: " << output_weights << std::endl;
        Pinv1 -= (cd1 * cd1_t) * 1 / cd1.dot(r);
        // std::cout << "P^-1: " << Pinv1 << std::endl;
    }
}

BDVector BDSpikingForceLearner::getPrediction()
{
    return prediction;
}