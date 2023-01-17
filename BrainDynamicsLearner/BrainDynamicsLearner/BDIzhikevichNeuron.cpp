#include "BDIzhikevichNeuron.h"

BDIzhikevichNeuron::BDIzhikevichNeuron() {}

BDIzhikevichNeuron::BDIzhikevichNeuron(BDIzhikevichParameters params, std::vector<bdi_real_t> synapse_weights, bdi_real_t initial_voltage) {
	init(params, synapse_weights, initial_voltage);
}

void BDIzhikevichNeuron::init(BDIzhikevichParameters params, std::vector<bdi_real_t> synapse_weights, bdi_real_t initial_voltage) {

	// Copy over the parameters.
	dt = params.dt;
	td = params.td;
	a = params.a;
	b = params.b;
	C = params.C;
	d = params.d;
	ff = params.ff;
	vt = params.vt;
	vr = params.vr;
	vpeak = params.vpeak;
	vreset = params.vreset;
	Er = params.Er;
	BIAS = params.BIAS;

	// Copy over the weights.
	weights = synapse_weights;

	// Initialize the state.
	v = initial_voltage;// voltage
	u = 0;// ?
	IPSC = 0;// ?
	h = 0;// ?
	r = 0;// ?
	hr = 0;// ?
	JD = 0;// ?
	is_spiking = false;// whether or not the neuron is spiking on this time step

}

void BDIzhikevichNeuron::step(std::vector<bdi_real_t> input) {
	// Find current total value of the membrane current.
	bdi_real_t I = IPSC + std::inner_product( weights.begin(), weights.end(), input.begin(), 0 ) + BIAS;
	// Update membrane voltage.
	bdi_real_t v_previous = v;
	bdi_real_t v_minus_vr = v_previous - vr;
	v = v_previous + dt * ( ff * v_minus_vr * (v_previous - vt) - u + I ) / C;
	u = u + dt * a * (b * v_minus_vr - u);
	is_spiking = v >= vpeak;
	if (is_spiking)
	{
		v = vpeak;
		u = u + d;
	}
	// Implement the synapse, either single or double exponential.
}