// BrainDynamicsLearner.cpp : This file contains the 'main' function. Program execution begins and ends there.
// FORCE-train a spiking NN on fMRI data.
// Adapted from
// https://github.com/ModelDBRepository/190565/blob/master/CODE%20FOR%20FIGURE%205%20(USER%20SUPPLIED%20SUPERVISOR)/IZFORCEMOVIE.m
// Nicola, W., & Clopath, C. (2017).
// Supervised learning in spiking neural networks with FORCE training.
// Nature communications, 8(1), 1 - 15.

#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include "BDSpikingForceLearner.h"

int main()
{
    const size_t num_brain_areas = 360;
    const size_t bytes_per_time_point = num_brain_areas * 64/8;// We store the values in 64-bit floating point numbers.
    char *time_point_bytes = new char [bytes_per_time_point];
    double* time_point = NULL;
    std::ifstream fmri_file;
    BDVector input_factors = Eigen::VectorXd::Constant(num_brain_areas, 400.0);
    const bd_size_t num_context_cues = 5;
    Eigen::Matrix<double,num_context_cues,1> context_factors = { 0.1, 1.0, 10.0, 100.0, 1000.0 };
    BDVector current_fmri_activity(num_brain_areas);
    BDSpikingForceLearner fmriLearner(input_factors, context_factors);
    BDVector prediction;
    Eigen::VectorXd context;
    std::cout << "This is BrainDynamicsLearner main.\n";
    fmri_file.open("C:\\Users\\agcraig\\Documents\\HCP_data\\fMRI_binaries\\ts_100206_1_LR.bin", std::ios::binary);
    if (!fmri_file.is_open())
    {
        std::cout << "failed to open fMRI data file.";
        return 1;
    }
    size_t num_time_points = 0;
    while ( fmri_file.good() && (num_time_points < 10) )
    {
        fmri_file.read(time_point_bytes, bytes_per_time_point);
        time_point = (double*)time_point_bytes;
        current_fmri_activity = Eigen::Map<Eigen::RowVectorXd>(time_point, num_brain_areas);
        context = Eigen::VectorXd::Random(num_context_cues);
        fmriLearner.neuronSimStep(context);
        prediction = fmriLearner.getPrediction();
        fmriLearner.recursiveLeastSquaresStep(current_fmri_activity);
        num_time_points++;
        std::cout << num_time_points << "\n predicted:\t" << prediction.transpose() << "\n real:\t" << current_fmri_activity.transpose() << "\n";
    }
    if ( fmri_file.fail() )
    {
        std::cout << "read failed\n";
    }
    if ( fmri_file.eof() )
    {
        std::cout << "reached end of file\n";
    }
    fmri_file.close();
    // Eigen::MatrixXd m1 = Eigen::MatrixXd::Random(4, 2);
    // Eigen::MatrixXd m2 = Eigen::MatrixXd::Random(2, 3);
    // Eigen::MatrixXd m3 = m1 * m2;
    // std::cout << "m1 * m2:\n" << m3 << std::endl;
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
