#include <iostream>
#include "SparseConvNet.h"
#include "NetworkArchitectures.h"
#include "SpatiallySparseDatasetUniProt.h"

int epoch = 0;
int cudaDevice = -1; //PCI bus ID, -1 for default GPU
int batchSize = 50;

class DeepPOFMP : public SparseConvNet {
public:
  DeepPOFMP(int dimension, int nInputFeatures, int nClasses, int cudaDevice,
            int nTop = 1)
      : SparseConvNet(dimension, nInputFeatures, nClasses, cudaDevice) {
    const float fmpShrink = powf(2, 0.6666); // Fractional Max-Pooling ratio.
    int l = 7; // Number of levels of FMP
    for (int i = 0; i < l; i++) {
      addLeNetLayerPOFMP(32 * (i + 1), 2, 1, 2, fmpShrink, VLEAKYRELU);
    }
    addLeNetLayerMP(32 * (l + 1), 2, 1, 1, 1, VLEAKYRELU);
    addLeNetLayerMP(32 * (l + 2), 1, 1, 1, 1, VLEAKYRELU);
    addSoftmaxLayer();
  }
};

int main(int argc, char *argv[])
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

	DeepC2 cnn(3, 5, 32, VLEAKYRELU, trainSet.nFeatures, trainSet.nClasses, 0.0f, cudaDevice);
    //DeepPOFMP cnn(3, trainSet.nFeatures, trainSet.nClasses, cudaDevice);

	if (epoch > 0)
		cnn.loadWeights(baseName, epoch);
    
    //cnn.processDatasetRepeatTest(testSet, batchSize, 25, "", "confusion.txt");    
    //return 0;

	for (epoch++; epoch <= epochs; epoch++)
	{
		std::cout << "epoch:" << epoch << ": " << std::flush;

		cnn.processDataset(trainSet, batchSize, 0.003 * exp(-5.0 / epochs * epoch));

		if (epoch % 100 == 0)
		{
			cnn.saveWeights(baseName, epoch);
			cnn.processDatasetRepeatTest(testSet, batchSize, 25, "", "confusion.txt");
		}
	}
}
