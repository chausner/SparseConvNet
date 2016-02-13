#include <iostream>
#include <iomanip>
#include <vector>
#include <tuple>
#include "SparseConvNet.h"
#include "NetworkArchitectures.h"
#include "SpatiallySparseDataset.h"
#include "PdbPicture.h"

int epoch = 0;
int cudaDevice = -1; //PCI bus ID, -1 for default GPU
int batchSize = 1;

class DeepPOFMP : public SparseConvNet {
public:
  DeepPOFMP(int dimension, int nInputFeatures, int nClasses, int cudaDevice,
            int nTop = 1)
      : SparseConvNet(dimension, nInputFeatures, nClasses, cudaDevice) {
    const float fmpShrink = powf(2, 0.6666); // Fractional Max-Pooling ratio.
    int l = 7; // Number of levels of FMP
    for (int i = 0; i < l; i++) {
      addLeNetLayerPOFMP(64 * (i + 1), 2, 1, 2, fmpShrink, VLEAKYRELU);
    }
    addLeNetLayerMP(64 * (l + 1), 2, 1, 1, 1, VLEAKYRELU);
    addLeNetLayerMP(64 * (l + 2), 1, 1, 1, 1, VLEAKYRELU);
    addSoftmaxLayer();
  }
};

SpatiallySparseDataset CreateSinglePdbPictureDataset(std::string filename, std::string chainsStr, float cellSize, float jiggleAlpha, int nFeatures, int nClasses)
{
	std::vector<char> chains(chainsStr.begin(), chainsStr.end());

	PdbPicture *pdbPicture = new PdbPicture(filename, chains, cellSize, jiggleAlpha, std::vector<int>());

	SpatiallySparseDataset dataset;

	dataset.name = filename;
	dataset.type = TESTBATCH;
	dataset.nFeatures = nFeatures;
	dataset.nClasses = nClasses;
	dataset.pictures.push_back(pdbPicture);
	
	return dataset;
}

std::vector<std::tuple<std::string, float>> ReadClassThresholds(std::string filename)
{
	std::ifstream file(filename);

	if (!file)
	{
		std::cout << "Cannot find " << filename << std::endl;
		exit(EXIT_FAILURE);
	}

	int nClasses;

	file >> nClasses;

	std::string line;
	std::getline(file, line);

	std::vector<std::tuple<std::string, float>> classThresholds;

	for (int i = 0; i < nClasses; i++)
	{
		std::string className;
		std::getline(file, className);
		std::getline(file, line);
		float threshold = atof(line.c_str());
		classThresholds.push_back(std::make_tuple(className, threshold));
	}

	return classThresholds;
}

int main(int argc, char *argv[])
{
	if (argc != 5)
	{
		std::cout << "Usage: uniprot_query weightsfile thresholdsfile pdbfile chains" << std::endl;
		std::cout << std::endl;
		std::cout << "    weightsfile:    path to .cnn file with learned network weights" << std::endl;
		std::cout << "    thresholdsfile: path to .txt file with labels and thresholds" << std::endl;
		std::cout << "    pdbfile:        path to PDB file to be analyzed" << std::endl;
		std::cout << "    chains:         chains in the PDB file to be analyzed (e.g. AB)" << std::endl;
		return 0;
	}

	float cellSize = 1;
	float jiggleAlpha = 0;
	int nFeatures = 11;	
	int nReps = 25;

	auto classThresholds = ReadClassThresholds(argv[2]);

	int nClasses = classThresholds.size();

	SpatiallySparseDataset dataset = CreateSinglePdbPictureDataset(argv[3], argv[4], cellSize, jiggleAlpha, nFeatures, nClasses);

    DeepPOFMP cnn(3, nFeatures, nClasses, cudaDevice);

	cnn.loadWeights(argv[1]);

	std::vector<std::vector<float>> probabilities;

	cnn.processDatasetRepeatTestGetProbabilities(dataset, batchSize, nReps, probabilities);

    for (int i = 0; i < dataset.nClasses; i++)
    	std::cout << std::left << std::setw(60) << std::setprecision(5) << std::fixed << 
		std::get<0>(classThresholds[i]) << std::setw(10) << probabilities[0][i] <<
		"(" << std::get<1>(classThresholds[i]) << ")" << std::endl;

	std::cout << std::endl << "Predicted classes:" << std::endl;

	for (int i = 0; i < dataset.nClasses; i++)
		if (probabilities[0][i] > std::get<1>(classThresholds[i]))
			std::cout << std::get<0>(classThresholds[i]) << std::endl;
}
