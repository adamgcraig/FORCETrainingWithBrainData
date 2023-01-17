#pragma once

#include<vector>
#include<numeric>
#include "BDNeuron.h"

typedef double bdi_real_t;
typedef bool bdi_bool_t;

typedef struct {
	bdi_real_t dt;// Euler integration step size
	bdi_real_t td;// ?
	bdi_real_t a;// ?
	bdi_real_t b;// ?
	bdi_real_t C;// capacitance?
	bdi_real_t d;// ?
	bdi_real_t ff;// ?
	bdi_real_t vt;// threshold voltage
	bdi_real_t vr;// reversal voltage
	bdi_real_t vpeak;// peak of voltage spike
	bdi_real_t vreset;// reset voltage after spike
	bdi_real_t Er;// ?
	bdi_real_t BIAS;// bias current
} BDIzhikevichParameters;

class BDIzhikevichNeuron : BDNeuron
{

protected:

	// The default values are those used in the original code from...
	// constant parameters
	bdi_real_t dt = 0.04;// Euler integration step size
	bdi_real_t td = 20;// ?
	bdi_real_t a = 0.002;// ?
	bdi_real_t b = 0;// ?
	bdi_real_t C = 250;// capacitance?
	bdi_real_t d = 100;// ?
	bdi_real_t ff = 2.5;// ?
	bdi_real_t vt = -40;// threshold voltage
	bdi_real_t vr = -60;// reversal voltage
	bdi_real_t vpeak = 30;// peak of voltage spike
	bdi_real_t vreset = -65;// reset voltage after spike
	bdi_real_t Er = 0;// ?
	bdi_real_t BIAS = 1000;// bias current

	// weights to give received inputs
	std::vector<bdi_real_t> weights;

	// variable state
	bdi_real_t v = 0;// voltage
	bdi_real_t u = 0;// ?
	bdi_real_t IPSC = 0;// ?
	bdi_real_t h = 0;// ?
	bdi_real_t r = 0;// ?
	bdi_real_t hr = 0;// ?
	bdi_real_t JD = 0;// ?
	bdi_bool_t is_spiking = false;// whether or not the neuron is spiking on this time step

public:

	BDIzhikevichNeuron();
	BDIzhikevichNeuron(BDIzhikevichParameters params, std::vector<bdi_real_t> synapse_weights, bdi_real_t initial_voltage);
	void init(BDIzhikevichParameters params, std::vector<bdi_real_t> synapse_weights, bdi_real_t initial_voltage);
	void step(std::vector<bdi_real_t> input);

};

