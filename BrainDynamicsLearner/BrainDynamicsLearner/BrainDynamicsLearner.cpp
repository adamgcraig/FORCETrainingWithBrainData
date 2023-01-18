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
#include <Eigen/Dense>
#include "BDSpikingForceLearner.h"

BDMatrix load_fmri_data(std::string file_name);
BDMatrix generate_hdts_context(bd_size_t num_dimensions, bd_size_t num_time_points, bd_float_t pulse_amplitude, bd_float_t decay_rate);

int main()
{
    std::cout << "This is BrainDynamicsLearner main.\n";
    const bd_size_t reps_per_sequence = 10;
    const bd_size_t sim_steps_per_data_step = 1;// 7000 / 4; // Delta-t of the simulation is 0.04 ms. Delta-t of the actual data is 700 ms.
    // Load the fMRI time series.
    BDMatrix fmri_activity = load_fmri_data("C:\\Users\\agcraig\\Documents\\HCP_data\\fMRI_binaries\\ts_100206_1_LR.bin");
    // std::cout << "fmri_activity: \n" << fmri_activity << std::endl;
    bd_size_t num_brain_areas = fmri_activity.rows();
    bd_size_t num_data_times = fmri_activity.cols();
    BDVector current_fmri_activity(num_brain_areas);
    const bd_size_t num_context_dimensions = 24;
    // Generate the HDTS context signal.
    BDMatrix context = generate_hdts_context(num_context_dimensions, num_data_times, 1.0, 0.1);
    // std::cout << "hdts_context: \n" << context << std::endl;
    BDVector current_context(num_context_dimensions);
    // Initialize the reservoir computing model.
    BDVector prediction_factors = Eigen::VectorXd::Constant(num_brain_areas, 400.0);
    BDVector context_factors = Eigen::VectorXd::Constant(num_context_dimensions, 4000.0);
    BDSpikingForceLearner fmriLearner(prediction_factors, context_factors);
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
            for (size_t sim_step_index = 0; sim_step_index < sim_steps_per_data_step; sim_step_index++)
            {
                fmriLearner.neuronSimStep(current_context);
                fmriLearner.recursiveLeastSquaresStep(current_fmri_activity);
                prediction = fmriLearner.getPrediction();
                errors_for_sim_step = prediction - current_fmri_activity;
                errors_for_data_step.col(sim_step_index) = errors_for_sim_step;
                errors_for_repetition.col(index_into_repetition) = errors_for_sim_step;
                index_into_repetition++;
                // rmse = std::sqrt(errors_for_sim_step.array().square().mean());
                // std::cout << "repetition " << rep_index << ",\t data step " << data_step_index << ",\t sim step " << sim_step_index << ",\t RMSE = " << rmse << std::endl;
            }
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

BDMatrix generate_hdts_context(bd_size_t num_dimensions, bd_size_t num_time_points, bd_float_t pulse_amplitude, bd_float_t decay_rate)
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

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
