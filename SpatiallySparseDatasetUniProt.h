#pragma once

#include <climits>
#include <string>
#include "SpatiallySparseDataset.h"

SpatiallySparseDataset LoadUniProtDataset(const char *filename, batchType batchType, float cellSize, float jiggleAlpha, int maxEntries = INT_MAX);