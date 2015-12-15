#pragma once

#include "Picture.h"
#include "Rng.h"
#include <armadillo>
#include <vector>

class PdbPicture : public Picture
{
private:
	std::string filename;
	arma::mat points;
	std::vector<char> atomTypes;

	float cellSize;    // in Ångström
	float jiggleAlpha; // in Ångström

public:
	PdbPicture(std::string filename, std::vector<char> &chains, float cellSize, float jiggleAlpha, int label);
	~PdbPicture();
	std::string identify();
	void normalize();
	void random_rotation(RNG &rng);
	void jiggle(RNG &rng);
	Picture *distort(RNG &rng, batchType type = TRAINBATCH);
	void codifyInputData(SparseGrid &grid, std::vector<float> &features, int &nSpatialSites, int spatialSize);
};
