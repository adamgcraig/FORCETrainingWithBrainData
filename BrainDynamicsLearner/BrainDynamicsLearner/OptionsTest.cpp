// BrainDynamicsLearner.cpp : This file contains the 'main' function. Program execution begins and ends there.
// written by Adam Craig at the HKBU Center for Nonlinear Studies
// created 2023-01-16
// last updated 2023-01-18
// This program uses the FORCE training method to teach a reservoir computing model to replicate fMRI time series.
// Adapted from
// https://github.com/ModelDBRepository/190565/blob/master/CODE%20FOR%20FIGURE%205%20(USER%20SUPPLIED%20SUPERVISOR)/IZFORCEMOVIE.m
// Nicola, W., & Clopath, C. (2017).
// Supervised learning in spiking neural networks with FORCE training.
// Nature communications, 8(1), 1 - 15.

#include <iostream>
#include <fstream>
// #include <Eigen/Dense>
#include <boost/program_options.hpp>
// #include "BDSpikingForceLearner.h"

namespace prog_opts = boost::program_options;

// void one_area_test();

// Just do a basic test where the reservoir computer learns all brain areas of a single fMRI time series over multiple iterations. 
// Prints out the root-mean-squared-error of each pass of the full time series.
// void all_areas_test(bd_size_t reps_per_sequence, bd_size_t sim_steps_per_data_step, bd_size_t num_context_dimensions, bd_float_t data_scaling_factor, bd_float_t reservoir_factor, bd_float_t Q_prediction, bd_float_t Q_context);

/*
BDMatrix load_fmri_data(std::string file_name);
BDMatrix generate_sinusoid_hdts(bd_size_t num_dimensions, bd_size_t num_time_points);
BDMatrix generate_decay_hdts(bd_size_t num_dimensions, bd_size_t num_time_points, bd_float_t pulse_amplitude, bd_float_t decay_rate);
*/

int main(int argc, char* argv[])
{
    prog_opts::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("compression", prog_opts::value<int>(), "set compression level")
        ;
    prog_opts::variables_map vm;
    prog_opts::store(prog_opts::parse_command_line(argc, argv, desc), vm);
    prog_opts::notify(vm);
    if (vm.count("help"))
    {
        std::cout << desc << "\n";
        return 1;
    }
    if (vm.count("compression")) {
        std::cout << "Compression level was set to " << vm["compression"].as<int>() << ".\n";
    }
    else
    {
        std::cout << "Compression level was not set.\n";
    }
    // one_area_test();
    // all_areas_test(20, 1000, 75, 0.001, 5000, 4000, 4000);
}

// void one_area_test()
// {
// }
/*
void all_areas_test(bd_size_t reps_per_sequence, bd_size_t sim_steps_per_data_step, bd_size_t num_context_dimensions, bd_float_t data_scaling_factor, bd_float_t reservoir_factor, bd_float_t Q_prediction, bd_float_t Q_context)
{
    std::cout << "This is BrainDynamicsLearner main.\n";
    // const bd_size_t sim_steps_per_data_step = 7000 / 4; // Delta-t of the simulation is 0.04 ms. Delta-t of the actual data is 700 ms.
    // Load the fMRI time series and divide all values by 100.
    BDMatrix fmri_activity = data_scaling_factor * load_fmri_data("C:\\Users\\agcraig\\Documents\\HCP_data\\fMRI_binaries\\ts_100206_1_LR.bin");
    bd_float_t root_mean_squared_activity = std::sqrt(fmri_activity.array().square().mean());
    std::cout << "root mean squared fMRI activity = " << root_mean_squared_activity << std::endl;
    // std::cout << "fmri_activity: \n" << fmri_activity << std::endl;
    bd_size_t num_brain_areas = fmri_activity.rows();
    bd_size_t num_data_times = fmri_activity.cols();
    BDVector current_fmri_activity(num_brain_areas);
    // Generate the HDTS context signal.
    BDMatrix context = generate_sinusoid_hdts(num_context_dimensions, num_data_times);
    // std::cout << "hdts_context: \n" << context << std::endl;
    BDVector current_context(num_context_dimensions);
    // Initialize the reservoir computing model.
    BDVector prediction_factors = Eigen::VectorXd::Constant(num_brain_areas, Q_prediction);
    BDVector context_factors = Eigen::VectorXd::Constant(num_context_dimensions, Q_context);
    BDSpikingForceLearner fmriLearner(reservoir_factor, prediction_factors, context_factors);
    BDVector prediction;
    BDVector errors_for_sim_step;
    BDMatrix errors_for_data_step = Eigen::MatrixXd::Constant(num_brain_areas, sim_steps_per_data_step, NAN);
    BDMatrix errors_for_repetition = Eigen::MatrixXd::Constant(num_brain_areas, num_data_times * sim_steps_per_data_step, NAN);
    bd_size_t index_into_repetition;
    bd_float_t rmse;
    // Train the model.
    for (size_t rep_index = 0; rep_index < reps_per_sequence; rep_index++)
    {
        index_into_repetition = 0;
        for (size_t data_step_index = 0; data_step_index < num_data_times; data_step_index++)
        {
            current_fmri_activity = fmri_activity.col(data_step_index);
            current_context = context.col(data_step_index);
            fmriLearner.doNSimStepsAnd1LeastSquaresStep(sim_steps_per_data_step, current_context, current_fmri_activity);
            for (size_t sim_step_index = 0; sim_step_index < sim_steps_per_data_step; sim_step_index++)
            {
                fmriLearner.neuronSimStep(current_context);
                prediction = fmriLearner.getPrediction();
                errors_for_sim_step = prediction - current_fmri_activity;
                errors_for_data_step.col(sim_step_index) = errors_for_sim_step;
                errors_for_repetition.col(index_into_repetition) = errors_for_sim_step;
                index_into_repetition++;
                // rmse = std::sqrt(errors_for_sim_step.array().square().mean());
                // std::cout << "repetition " << rep_index << ",\t data step " << data_step_index << ",\t sim step " << sim_step_index << ",\t RMSE = " << rmse << std::endl;
            }
            BDVector data_step_mean_error = errors_for_data_step.rowwise().mean();
            fmriLearner.recursiveLeastSquaresStepForError(data_step_mean_error);
            // fmriLearner.recursiveLeastSquaresStep(current_fmri_activity);
            // rmse = std::sqrt(errors_for_data_step.array().square().mean());
            // std::cout << "repetition " << rep_index << ",\t data step " << data_step_index << ",\t RMSE = " << rmse << std::endl;
        }
        rmse = std::sqrt(errors_for_repetition.array().square().mean());
        std::cout << "repetition " << rep_index << ",\t RMSE = " << rmse << std::endl;
    }
}

BDMatrix load_fmri_data(std::string file_name) {
    BDMatrix fmri_data;
    const bd_size_t num_brain_areas = 360;
    std::streampos size;
    bd_size_t num_floats;
    bd_size_t num_time_points;
    char* memblock;
    std::ifstream file(file_name, std::ios::in | std::ios::binary | std::ios::ate);
    if (file.is_open())
    {
        std::cout << "reading from file " << file_name << std::endl;
        size = file.tellg();
        num_floats = size / 8;// 8 bytes per 64-bit float
        num_time_points = num_floats / num_brain_areas;
        std::cout << "detected " << num_floats << " values, which we will try to read into a " << num_brain_areas << "x" << num_time_points << " matrix.\n";
        memblock = new char[size];
        file.seekg(0, std::ios::beg);
        file.read(memblock, size);
        file.close();
        std::cout << "finished reading\n";
        fmri_data = Eigen::MatrixXd::Map((bd_float_t*)memblock, num_brain_areas, num_time_points);
        delete[] memblock;
    }
    else
    {
        std::cout << "Unable to open file " << file_name << std::endl;
        fmri_data = Eigen::MatrixXd::Constant(1, 1, NAN);
    }
    return fmri_data;
}

BDMatrix interpolate_fmri_data(BDMatrix fmri_data, bd_float_t out_points_per_in_point)
{
    BDMatrix result = fmri_data;
    return result;
}

BDMatrix generate_sinusoid_hdts(bd_size_t num_dimensions, bd_size_t num_time_points)
{
    bd_size_t sinusoid_length = num_time_points / num_dimensions;
    // Make the basic half-sinusoid shape.
    Eigen::RowVectorXd sinusoid = Eigen::RowVectorXd::LinSpaced(sinusoid_length, 0.0, EIGEN_PI).array().sin();
    // Make a matrix of 0s.
    BDMatrix hdts = Eigen::MatrixXd::Constant(num_dimensions, num_time_points, 0.0);
    // Copy the half-sinusoid shape into each dimension (column) at the time point just after the one in the previous dimension ends.
    bd_size_t start_col = 0;
    for (size_t dimension_index = 0; dimension_index < num_dimensions; dimension_index++)
    {
        hdts(dimension_index, Eigen::seqN(start_col, sinusoid_length)) = sinusoid;
        start_col += sinusoid_length;
    }
    return hdts;
}

BDMatrix generate_decay_hdts(bd_size_t num_dimensions, bd_size_t num_time_points, bd_float_t pulse_amplitude, bd_float_t decay_rate)
{
    BDMatrix hdts = Eigen::MatrixXd::Constant(num_dimensions, num_time_points, 0.0);
    BDVector current_hdts = Eigen::VectorXd::Constant(num_dimensions, 0.0);
    bd_size_t current_spike_location = 0;
    current_hdts(0) = pulse_amplitude;
    hdts.col(0) = current_hdts;
    bd_float_t multiplier = 1 - decay_rate;
    bd_size_t time_between_spikes = num_time_points / num_dimensions;
    for (size_t t = 1; t < num_time_points; t++)
    {
        current_hdts *= multiplier;
        if (t % time_between_spikes == 0)
        {
            current_spike_location++;
            current_hdts[current_spike_location] = pulse_amplitude;
        }
        hdts.col(t) = current_hdts;
    }
    return hdts;
}
*/
// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
