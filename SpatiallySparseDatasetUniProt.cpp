#include <vector>
#include <iostream>
#include <fstream>
#include "SpatiallySparseDatasetUniProt.h"
#include "PdbPicture.h"

SpatiallySparseDataset LoadUniProtDataset(const char *filename, batchType batchType, float cellSize, float jiggleAlpha, int numEntries)
{
	SpatiallySparseDataset dataset;

	dataset.name = batchType == TRAINBATCH ? "UniProt train set" : "UniProt test set";
	dataset.type = batchType;
	dataset.nFeatures = 4;

	std::ifstream file(filename);

	if (!file)
	{
		std::cout << "Cannot find " << filename << std::endl;
		exit(EXIT_FAILURE);
	}

	int nEntries;

	file >> dataset.nClasses >> nEntries;

	for (int i = 0; i < std::min(nEntries, numEntries); i++)
	{
		std::string pdbFilename;
		std::string chainsStr;
		int label;

		file >> pdbFilename;
		file >> chainsStr;
		file >> label;

		std::vector<char> chains(chainsStr.begin(), chainsStr.end());

		dataset.pictures.push_back(new PdbPicture(pdbFilename, chains, cellSize, jiggleAlpha, label));
	}

	file.close();

	return dataset;
}