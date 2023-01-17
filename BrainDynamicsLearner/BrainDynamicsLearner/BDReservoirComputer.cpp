#include "BDReservoirComputer.h"

BDReservoirComputer::BDReservoirComputer()
{
}

BDReservoirComputer::BDReservoirComputer(uint32_t rng_seed, std::vector<bdrc_weight_t> input_weight_multipliers, bdrc_size_t num_neurons, bdrc_weight_t reservoir_weight_multiplier, bdrc_probability_t reservoir_target_density, bdrc_size_t num_outputs)
{
	rng.seed(rng_seed);
	std::uniform_real_distribution<bdrc_weight_t> input_weight_dist(-1.0, 1.0);
	bdrc_weight_t reservoir_weight_std = reservoir_weight_multiplier / ((bdrc_weight_t)reservoir_target_density * sqrt((bdrc_weight_t)num_neurons));
	std::uniform_real_distribution<bdrc_probability_t> reservoir_connection_dist(0.0, reservoir_weight_std);
	std::normal_distribution<bdrc_size_t> reservoir_weight_dist(0.0, 1.0);
	std::uniform_real_distribution<bdrc_weight_t> output_weight_dist(-1.0, 1.0);
	bdrc_size_t num_inputs = input_weight_multipliers.size();
	neurons.resize(num_neurons);
	for (size_t i = 0; i < neurons.size(); i++)
	{
		// neurons[i].init();
	}
	input_weights.resize(num_neurons);
	for (bdrc_size_t i = 0; i < input_weights.size(); i++)
	{
		input_weights[i].resize(num_inputs);
		for (bdrc_size_t j = 0; j < input_weights[i].size(); j++)
		{
			input_weights[i][j] = input_weight_multipliers[j] * input_weight_dist(rng);
		}
	}
	reservoir_weights.resize(num_neurons);
	for (bdrc_size_t i = 0; i < reservoir_weights.size(); i++)
	{
		reservoir_weights[i].resize(num_neurons);
		for (bdrc_size_t j = 0; j < reservoir_weights[i].size(); j++)
		{
			bdrc_weight_t is_connected = (bdrc_weight_t) ( reservoir_connection_dist(rng) <= reservoir_target_density );
			reservoir_weights[i][j] = is_connected * reservoir_weight_dist(rng);
		}
	}
	output_weights.resize(num_outputs);
	for (bdrc_size_t i = 0; i < output_weights.size(); i++)
	{
		output_weights[i].resize(num_neurons);
		for (bdrc_size_t j = 0; j < output_weights[i].size(); j++)
		{
			output_weights[i][j] = output_weight_dist(rng);
		}
	}
}