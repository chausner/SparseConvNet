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
	dataset.nFeatures = 11;

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
		std::vector<int> labels;

		file >> pdbFilename;
		file >> chainsStr;
        
		int nLabels;
		file >> nLabels;
		for (int j = 0; j < nLabels; j++)
		{
			int label;
			file >> label;
			labels.push_back(label); 
		}

		std::vector<char> chains(chainsStr.begin(), chainsStr.end());

        PdbPicture *pic = new PdbPicture(pdbFilename, chains, cellSize, jiggleAlpha, labels);
        
        if (!pic->empty())
		    dataset.pictures.push_back(pic);
	}

	file.close();

	return dataset;
}