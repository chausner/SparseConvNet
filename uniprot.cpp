#include <iostream>
#include "SparseConvNet.h"
#include "NetworkArchitectures.h"
#include "SpatiallySparseDatasetUniProt.h"

int epoch = 0;
int cudaDevice = -1; //PCI bus ID, -1 for default GPU
int batchSize = 50;

int main()
{
	std::string baseName = "weights/pfp";

	float cellSize = 1;
	float jiggleAlpha = 0;
	int maxEntries = 999999;
	int epochs = 3000;

	SpatiallySparseDataset trainSet = LoadUniProtDataset("Data/UniProt/train.txt", TRAINBATCH, cellSize, jiggleAlpha, maxEntries);
	SpatiallySparseDataset testSet = LoadUniProtDataset("Data/UniProt/test.txt", TESTBATCH, cellSize, jiggleAlpha, maxEntries);

	trainSet.summary();
	testSet.summary();

	DeepC2 cnn(3, 4, 32, VLEAKYRELU, trainSet.nFeatures, trainSet.nClasses, 0.0f, cudaDevice);

	if (epoch > 0)
		cnn.loadWeights(baseName, epoch);

	for (epoch++; epoch <= epochs; epoch++)
	{
		std::cout << "epoch:" << epoch << ": " << std::flush;

		cnn.processDataset(trainSet, batchSize, 0.003 * exp(-5.0 / epochs * epoch));

		if (epoch % 50 == 0)
		{
			cnn.saveWeights(baseName, epoch);
			cnn.processDatasetRepeatTest(testSet, batchSize, 3);
		}
	}
}
