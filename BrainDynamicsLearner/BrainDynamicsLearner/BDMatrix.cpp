#include "BDMatrix.h"

BDMatrix::BDMatrix() {}

void BDMatrix::setInnerRepresentation(BDMInnerMatRep m_new) {
	m = m_new;
}

BDMInnerMatRep BDMatrix::getInnerRepresentation() {
	return m;
}

BDMatrix BDMatrix::multiplyBy(BDMatrix other) {
	BDMatrix result;
	result.setInnerRepresentation( m * other.getInnerRepresentation() );
	return result;
}

void BDMatrix::doPInverseUpdateInPlace(BDMatrix output) {

}