#pragma once
#include<vector>
#include<random>
#include "BDIzhikevichNeuron.h"

typedef size_t bdrc_size_t;
typedef double bdrc_weight_t;
typedef double bdrc_probability_t;

class BDReservoirComputer
{
protected:
	std::mt19937 rng;
	std::vector<std::vector<bdrc_weight_t>> input_weights;
	std::vector<std::vector<bdrc_weight_t>> reservoir_weights;
	std::vector<std::vector<bdrc_weight_t>> output_weights;
	std::vector<BDIzhikevichNeuron> neurons;
	
public:
	BDReservoirComputer();
	BDReservoirComputer(uint32_t rng_seed, std::vector<bdrc_weight_t> input_weight_multipliers, bdrc_size_t num_neurons, bdrc_weight_t reservoir_weight_multiplier, bdrc_probability_t reservoir_target_density, bdrc_size_t num_outputs);
};

