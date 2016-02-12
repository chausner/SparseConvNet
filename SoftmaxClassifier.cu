#include "SoftmaxClassifier.h"
#include <algorithm>
#include <iostream>
#include <vector>
#include <cassert>
#include "utilities.h"

__global__ void dDerivativeOfCostWRTpreSoftmaxTopLevelWeights(
    int batchSize, float *topDelta, float *topGrid, int *labels, int N) {
  for (int k = 0; k < batchSize; k++) {
    for (int i = threadIdx.x; i < N; i += NTHREADS) {
	  topDelta[k * N + i] = topGrid[k * N + i] - (labels[k * N + i] == 1);
    }
  }
}

void SoftmaxClassifier(SpatiallySparseBatchInterface &input,
                       SpatiallySparseBatch &batch, int nTop,
                       cudaMemStream &memStream) {
  // Assume no dropout in the output layer! nClasses:=input.nFeatures.
  assert(batch.batchSize == input.nSpatialSites);
  assert(input.nFeatures == input.featuresPresent.size());

  if (batch.type ==
      TRAINBATCH) { // Begin backprop. Top layer: d Cost / d SoftmaxInput
    input.sub->dfeatures.resize(input.nSpatialSites *
                                input.featuresPresent.size());
    dDerivativeOfCostWRTpreSoftmaxTopLevelWeights
            << <1, NTHREADS, 0, memStream.stream>>>
        (batch.batchSize, input.sub->dfeatures.dPtr(),
         input.sub->features.dPtr(), batch.labels.dPtr(), input.nFeatures);
  }

  input.sub->features.copyToCPUAsync(memStream);
  batch.labels.copyToCPUAsync(memStream);

  float *probs = &input.sub->features.hVector()[0];
  for (int i = 0; i < batch.batchSize; ++i)
    batch.probabilities.push_back(std::vector<float>(
        probs + i * input.nFeatures, probs + (i + 1) * input.nFeatures));
  for (int i = 0; i < batch.batchSize; i++)
    //batch.predictions.push_back(vectorTopIndices(batch.probabilities[i], nTop));
    batch.predictions.push_back(multiLabelClassification(batch.probabilities[i], 0.5));

  if (batch.type != UNLABELEDBATCH) {
    batch.mistakes += batch.batchSize;
    for (int i = 0; i < batch.batchSize; i++) {
      //batch.negativeLogLikelihood -=
      //    log(max(batch.probabilities[i][batch.labels.hVector()[i]], 1.0e-15));
	  for (int j = 0; j < input.nFeatures; j++) {
		float error = batch.probabilities[i][j] - batch.labels.hVector()[i * input.nFeatures + j];
		batch.sumSquaredError += error * error;
	  }
      for (int j = 0; j < batch.predictions[i].size(); j++) {
		if (batch.labels.hVector()[i * input.nFeatures + batch.predictions[i][j]] == 1) {
          batch.mistakes -= 1.0 / input.nFeatures;
        }
      }
	  for (int j = 0; j < input.nFeatures; j++) {
		if (std::find(batch.predictions[i].begin(), batch.predictions[i].end(), j) == batch.predictions[i].end()) {
	      if (batch.labels.hVector()[i * input.nFeatures + j] == 0) {
		    batch.mistakes -= 1.0 / input.nFeatures;
		  }
		}
	  }
    }
  }
  // std::cout << (int)batch.negativeLogLikelihood << " " << std::flush;
  input.sub->features.copyToGPUAsync(memStream);
  cudaCheckError();
}
