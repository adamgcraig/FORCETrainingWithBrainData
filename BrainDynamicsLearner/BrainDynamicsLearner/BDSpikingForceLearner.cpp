#include "BDSpikingForceLearner.h"

BDSpikingForceLearner::BDSpikingForceLearner(bd_float_t reservoir_scaling_factor, BDVector prediction_scaling_factors, BDVector context_scaling_factors)
{
    G = reservoir_scaling_factor;
    num_predictions = prediction_scaling_factors.size();
    num_context_cues = context_scaling_factors.size();
    Q_prediction = prediction_scaling_factors.array();// one-per-input weighting factors
    Q_context = context_scaling_factors.array();// one-per-input weighting factors
    // Fixed network weight matrices
    prediction_input_weights = Eigen::MatrixXd::Random(num_neurons, num_predictions).array() * Q_prediction.transpose().replicate(num_neurons,1).array();
    // std::cout << "prediction input weights: " << prediction_input_weights << std::endl;
    context_input_weights = Eigen::MatrixXd::Random(num_neurons, num_context_cues).array() * Q_context.transpose().replicate(num_neurons, 1).array();
    // std::cout << "context input weights: " << context_input_weights << std::endl;
    std::default_random_engine generator;
    bd_float_t reservoir_std_dev = G / ( p * std::sqrt((bd_float_t) num_neurons) );
    std::normal_distribution<bd_float_t> distribution(0.0, reservoir_std_dev);
    bd_size_t num_reservoir_weights = num_neurons * num_neurons;
    bd_float_t *norm_weights_array = new bd_float_t[num_reservoir_weights];
    for (size_t i = 0; i < num_reservoir_weights; i++)
    {
        norm_weights_array[i] = distribution(generator);
    }
    BDMatrix norm_weights = Eigen::Map<BDMatrix>(norm_weights_array, num_neurons, num_neurons);
    BDMatrix zero_weights = Eigen::MatrixXd::Constant(num_neurons, num_neurons, 0.0);
    reservoir_weights = ( Eigen::MatrixXd::Random(num_neurons, num_neurons).array().abs() < p).select(norm_weights, zero_weights);
    // std::cout << "reservoir weights: " << reservoir_weights << std::endl;
    // bd_float_t norm_weight_mean = norm_weights.mean();
    // BDMatrix nw_diff_from_mean = norm_weights.array() - norm_weight_mean;
    // BDMatrix nw_diff_from_mean_sq = nw_diff_from_mean.array() * nw_diff_from_mean.array();
    // bd_float_t norm_weight_variance = nw_diff_from_mean_sq.sum() / ((bd_float_t) nw_diff_from_mean_sq.size() - 1.0);
    // Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> is_non_0 = reservoir_weights.array() != 0;
    // bd_float_t nw_fraction_non_0 = is_non_0.cast<bd_float_t>().sum() / (bd_float_t) is_non_0.size();
    // std::cout << "reservoir weight mean: " << norm_weight_mean << ", variance: " << norm_weight_variance << ", fraction non-0: " << nw_fraction_non_0 << std::endl;
    // Neuron state vectors
    I_synapse = Eigen::VectorXd::Constant(num_neurons, 0.0);
    prediction = Eigen::VectorXd::Constant(num_predictions, 0.0);
    v = v_resting + (v_peak - v_resting)*( Eigen::ArrayXd::Random(num_neurons) + 1 )/2;
    u = Eigen::ArrayXd::Constant(num_neurons, 0.0);
    h = Eigen::ArrayXd::Constant(num_neurons, 0.0);
    r = Eigen::ArrayXd::Constant(num_neurons, 0.0);
    hr = Eigen::ArrayXd::Constant(num_neurons, 0.0);
    // State matrices used in FORCE training.
    P = 2*Eigen::MatrixXd::Identity(num_neurons, num_neurons);
    output_weights = Eigen::MatrixXd::Constant(num_predictions, num_neurons, 0.0);
}

void BDSpikingForceLearner::neuronSimStep(BDVector context)
{
    BDMatrix I_input = prediction_input_weights * prediction + context_input_weights * context;
    // std::cout << "weighted input: " << weighted_input.transpose() << std::endl;
    // After this, we are mostly in element-wise operation mode, as we update the state of each neuron independently.
    BDArray I = I_bias + I_synapse + I_input.array(); // membrane current
    // std::cout << "I: " << I.transpose() << std::endl;
    // We have b set to 0, so we can save some time with a simplified version of these steps:
    BDArray v_ = v; // Set v(t - 1) = v so that we can still use this value after updating v to v(t).
    BDArray v_minus_v_resting = v_ - v_resting;
    v += dt_over_C * ( k * v_minus_v_resting * (v_-v_threshold) - u + I ); // v(t) = v(t - 1) + dt * v'(t-1)
    u += dt * (  a * ( b * v_minus_v_resting - u )  ); // same with u, the v_ term makes it so that the integration of u uses v(t - 1), instead of the updated v(t)
    // v += dt_over_C * (k * (v - v_resting) * (v - v_threshold) - u + I_bias + I_synapse + I_input.array());
    Eigen::Array<bool,Eigen::Dynamic,1> is_spike = v >= v_peak;
    BDArray is_spike_double = (BDVector)is_spike.cast<bd_float_t>();
    // std::cout << "is_spike: " << is_spike.transpose() << std::endl;
    v = is_spike.select(v_reset,v); // implements v = c if v > vpeak add 0 if false, add c - v if true, v + c - v = c
    u = is_spike.select(u + d, u); // set u to u + d if v > vpeak
    // std::cout << "v: " << v.transpose() << std::endl;
    // u += -dt_a * u + d*is_spike_double;// set u to u + d if v > vpeak
    // std::cout << "u: " << u.transpose() << std::endl;
    // We briefly work in matrix operation mode here when transmitting spikes between neurons.
    // BDVector s = reservoir_weights * (BDVector)is_spike_double; // Compute the increase in current due to spiking.
    // Just implement the double-exponential synapse so that we do not need an if-statement every step.
    // std::cout << "JD: " << JD.transpose() << std::endl;
    I_synapse = exp_neg_dt_over_tau_r * I_synapse + dt * h;
    // std::cout << "r: " << r.transpose() << std::endl;
    h = exp_neg_dt_over_tau_d * h + one_over_tr_td * (reservoir_weights * (BDVector)is_spike_double).array();
    // std::cout << "IPSC: " << IPSC.transpose() << std::endl;
    r = exp_neg_dt_over_tau_r * r.array() + dt * hr;
    // std::cout << "h: " << h.transpose() << std::endl;
    hr = exp_neg_dt_over_tau_d * hr + one_over_tr_td * is_spike_double;
    // std::cout << "hr: " << hr.transpose() << std::endl;
    // We use matrices one more time when we compute the output of this time step, which will serve as the input of the next.
    prediction = output_weights * r;
}

void BDSpikingForceLearner::recursiveLeastSquaresStepForError(BDVector error_value)
{
    // BDVector error_value = prediction - correct_output;
    // std::cout << "error_value: " << error_value.transpose() << std::endl;
    BDVector Pr = P * r;
    // std::cout << "cd1: " << cd1.transpose() << std::endl;
    Eigen::RowVectorXd rTP = Pr.transpose();
    output_weights -= error_value * rTP;
    // std::cout << "output weights: " << output_weights << std::endl;
    P -= (Pr * rTP) / (1 + r.dot(Pr));
    // std::cout << "P^-1: " << Pinv1 << std::endl;
}

void BDSpikingForceLearner::recursiveLeastSquaresStep(BDVector correct_output)
{
    recursiveLeastSquaresStepForError(prediction - correct_output);
}

void BDSpikingForceLearner::doNSimStepsAnd1LeastSquaresStep(bd_size_t N, BDVector context, BDVector correct_average_output)
{
    BDMatrix predictions = Eigen::MatrixXd::Constant(num_predictions, N, 0.0);
    for (size_t step = 0; step < N; step++)
    {
        neuronSimStep(context);
        predictions.col(step) = getPrediction();
    }
    BDVector average_error = predictions.rowwise().mean() - correct_average_output;
    recursiveLeastSquaresStepForError(average_error);
}

BDVector BDSpikingForceLearner::getPrediction()
{
    return prediction;
}