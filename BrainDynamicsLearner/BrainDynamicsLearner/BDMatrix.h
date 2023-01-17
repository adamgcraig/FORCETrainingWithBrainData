#pragma once
// A wrapper for whatever linear algebra library we end up using.
// For now, Eigen should be good enough.

#include <Eigen/src/Core/Matrix.h>

typedef size_t bdm_size_t;
typedef double bdm_real_t;
typedef MatrixXd BDMInnerMatRep;

class BDMatrix
{
private:
	BDMInnerMatRep m;
	void setInnerRepresentation(BDMInnerMatRep m_new);
	BDMInnerMatRep getInnerRepresentation();
public:
	BDMatrix();
	BDMatrix multiplyBy(BDMatrix other);
	void doPInverseUpdateInPlace(BDMatrix output);
};

