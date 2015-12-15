#include <string>
#include <fstream>
#include "PdbPicture.h"
#include "utilities.h"

char elementSymbolToChar(std::string elementSymbol)
{
	switch (elementSymbol[0])
	{
	case 'C':
		return 0;
	case 'O':
		return 1;
	case 'N':
		return 2;
	case 'S':
		return 3;
	default:
		return -1;
	}
}

PdbPicture::PdbPicture(std::string filename, std::vector<char> &chains, float cellSize, float jiggleAlpha, int label) 
	: Picture(label), cellSize(cellSize), jiggleAlpha(jiggleAlpha)
{
	std::ifstream file(filename.c_str());

	if (!file)
	{
		std::cout << "Cannot find " << filename << std::endl;
		exit(EXIT_FAILURE);
	}

	std::vector<std::tuple<float, float, float>> atoms;

	while (!file.eof())
	{
		std::string line;

		if (!std::getline(file, line))
			break;

		line = trim(line);

		std::string recordName = line.substr(0, 6);

		if (recordName == "ENDMDL")
			break; // only process first model
		else if (recordName == "ATOM  ")
		{
			int atom_serial_num = std::stoi(line.substr(6, 11));
			std::string atom_name = trim(line.substr(12, 16));
			std::string atom_residue_name = trim(line.substr(17, 20));
			char atom_chain_id = line[21];
			int atom_residue_seq = std::stoi(line.substr(22, 26));
			float atom_coord_x = std::stof(line.substr(30, 38));
			float atom_coord_y = std::stof(line.substr(38, 46));
			float atom_coord_z = std::stof(line.substr(46, 54));
			std::string atom_element = trim(line.substr(76, 78));

			if (std::find(chains.begin(), chains.end(), atom_chain_id) != chains.end())
			{
				atoms.push_back(std::tuple<float, float, float>(atom_coord_x, atom_coord_y, atom_coord_z));
				atomTypes.push_back(elementSymbolToChar(atom_element));
			}
		}
	}

	file.close();
    
    if (atoms.empty())
    {
		std::cout << "No atoms found in " << filename << std::endl;
		exit(EXIT_FAILURE);
    }

	points.set_size(atoms.size(), 3);

	for (int i = 0; i < atoms.size(); i++)
	{
		points(i, 0) = std::get<0>(atoms[i]);
		points(i, 1) = std::get<1>(atoms[i]);
		points(i, 2) = std::get<2>(atoms[i]);
	}
}

PdbPicture::~PdbPicture()
{
}

std::string PdbPicture::identify()
{
	return filename;
}

void PdbPicture::normalize()
{
	arma::mat pointsm = arma::min(points, 0);
	arma::mat pointsM = arma::max(points, 0);
	points = points - arma::repmat(0.5 * (pointsm + pointsM), points.n_rows, 1);
	points /= cellSize;
}

void PdbPicture::random_rotation(RNG &rng)
{
	arma::mat L, Q, R;
	L.set_size(3, 3);
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			L(i, j) = rng.uniform();
	arma::qr(Q, R, L);
	points = points * Q;
}

void PdbPicture::jiggle(RNG &rng)
{
	for (int i = 0; i < 3; i++)
		points.col(i) += rng.uniform(-jiggleAlpha, jiggleAlpha) / cellSize;
}

Picture *PdbPicture::distort(RNG &rng, batchType type)
{
	PdbPicture *pic = new PdbPicture(*this);

	pic->random_rotation(rng);
	pic->normalize();

	if (type == TRAINBATCH)
		pic->jiggle(rng);

	return pic;
}

int mapToGrid(float coord, int inputFieldSize)
{
	return (int)(coord + 0.5 * inputFieldSize);
}

void PdbPicture::codifyInputData(SparseGrid &grid, std::vector<float> &features, int &nSpatialSites, int spatialSize)
{
	for (int c = 0; c < 4; c++)
	    features.push_back(0); // Background feature

	grid.backgroundCol = nSpatialSites++;

	int startNSpatialSites = nSpatialSites;
	size_t startFeaturesSize = features.size();

	std::vector<int> featuresIndexPerSpatialSite;

	for (int i = 0; i < points.n_rows; i++)
	{
		int x = mapToGrid(points(i, 0), spatialSize);
		int y = mapToGrid(points(i, 1), spatialSize);
		int z = mapToGrid(points(i, 2), spatialSize);

		if (x < 0 || y < 0 || z < 0 ||
			x >= spatialSize || y >= spatialSize || z >= spatialSize ||
			atomTypes[i] == -1)
			continue;

		int n = x * spatialSize * spatialSize + y * spatialSize + z;

		if (grid.mp.find(n) == grid.mp.end())
		{
			featuresIndexPerSpatialSite.push_back(features.size());

			grid.mp[n] = nSpatialSites++;

			float featureVector[4] = {};
			featureVector[atomTypes[i]] = 1;

			features.push_back(featureVector[0]);
			features.push_back(featureVector[1]);
			features.push_back(featureVector[2]);
			features.push_back(featureVector[3]);

			//float featureVector[4] = {};
			//featureVector[atomTypes[i]] = 1;
			//features.insert(features.end(), featureVector, featureVector + 4);
		}
		else
		{
			int featureIndex = featuresIndexPerSpatialSite[nSpatialSites - startNSpatialSites - 1];
			features[featureIndex + atomTypes[i]]++;
		}
	}
}