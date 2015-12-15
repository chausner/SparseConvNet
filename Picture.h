#pragma once
#include <string>
#include <vector>
#include "SparseGrid.h"
#include "Rng.h"
#include "types.h"

class Picture {
public:
  virtual void codifyInputData(SparseGrid &grid, std::vector<float> &features,
                               int &nSpatialSites, int spatialSize) = 0;
  virtual Picture *distort(RNG &rng, batchType type) { return this; }
  virtual std::string identify();
  std::vector<int> labels;
  Picture(std::vector<int> labels);
  virtual ~Picture();
};
