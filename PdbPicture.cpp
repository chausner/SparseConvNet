#include <string>
#include <fstream>
#include "PdbPicture.h"

#include <algorithm> 
#include <functional> 
#include <cctype>
#include <locale>

static inline std::string ltrim(std::string s)
{
    size_t startpos = s.find_first_not_of(" \t");
    if (startpos != std::string::npos)
        s = s.substr(startpos);
    return s;
}

static inline std::string rtrim(std::string s)
{
    size_t endpos = s.find_last_not_of(" \t");
    if (endpos != std::string::npos)
        s = s.substr(0, endpos + 1);
    return s;
}

static inline std::string trim(std::string s)
{
	return ltrim(rtrim(s));
}

char codifyAtom(std::string elementSymbol, std::string atom_name)
{
	switch (elementSymbol[0])
	{
	case 'C':
		if (atom_name == "CA")
			return 4;
		else if (atom_name == "CB")
			return 5;
		else if (atom_name.substr(0, 2) == "CG")
			return 6;
		else if (atom_name.substr(0, 2) == "CD")
			return 7;
		else
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

float getHydrophobicity(std::string atom_residue_name)
{
    if (atom_residue_name == "ALA") return 1.8;
    else if (atom_residue_name == "ARG") return -4.5;
    else if (atom_residue_name == "ASN") return -3.5;
    else if (atom_residue_name == "ASP") return -3.5;
    else if (atom_residue_name == "CYS") return 2.5;
    else if (atom_residue_name == "GLU") return -3.5;
    else if (atom_residue_name == "GLN") return -3.5;
    else if (atom_residue_name == "GLY") return -0.4;
    else if (atom_residue_name == "HIS") return -3.2;
    else if (atom_residue_name == "ILE") return 4.5;
    else if (atom_residue_name == "LEU") return 3.8;
    else if (atom_residue_name == "LYS") return -3.9;
    else if (atom_residue_name == "MET") return 1.9;
    else if (atom_residue_name == "PHE") return 2.8;
    else if (atom_residue_name == "PRO") return -1.6;
    else if (atom_residue_name == "SER") return -0.8;
    else if (atom_residue_name == "THR") return -0.7;
    else if (atom_residue_name == "TRP") return -0.9;
    else if (atom_residue_name == "TYR") return -1.3;
    else if (atom_residue_name == "VAL") return 4.2;
    else return 0;
}

float getPolarity(std::string atom_residue_name)
{
    if (atom_residue_name == "ALA") return 0;
    else if (atom_residue_name == "ARG") return 1;
    else if (atom_residue_name == "ASN") return 1;
    else if (atom_residue_name == "ASP") return 1;
    else if (atom_residue_name == "CYS") return 0;
    else if (atom_residue_name == "GLU") return 1;
    else if (atom_residue_name == "GLN") return 1;
    else if (atom_residue_name == "GLY") return 0;
    else if (atom_residue_name == "HIS") return 1;
    else if (atom_residue_name == "ILE") return 0;
    else if (atom_residue_name == "LEU") return 0;
    else if (atom_residue_name == "LYS") return 1;
    else if (atom_residue_name == "MET") return 0;
    else if (atom_residue_name == "PHE") return 0;
    else if (atom_residue_name == "PRO") return 0;
    else if (atom_residue_name == "SER") return 1;
    else if (atom_residue_name == "THR") return 1;
    else if (atom_residue_name == "TRP") return 0;
    else if (atom_residue_name == "TYR") return 1;
    else if (atom_residue_name == "VAL") return 0;
    else return 0;
}

float getCharge(std::string atom_residue_name)
{
    if (atom_residue_name == "ALA") return 0;
    else if (atom_residue_name == "ARG") return 1;
    else if (atom_residue_name == "ASN") return 0;
    else if (atom_residue_name == "ASP") return -1;
    else if (atom_residue_name == "CYS") return 0;
    else if (atom_residue_name == "GLU") return -1;
    else if (atom_residue_name == "GLN") return 0;
    else if (atom_residue_name == "GLY") return 0;
    else if (atom_residue_name == "HIS") return 0;
    else if (atom_residue_name == "ILE") return 0;
    else if (atom_residue_name == "LEU") return 0;
    else if (atom_residue_name == "LYS") return 1;
    else if (atom_residue_name == "MET") return 0;
    else if (atom_residue_name == "PHE") return 0;
    else if (atom_residue_name == "PRO") return 0;
    else if (atom_residue_name == "SER") return 0;
    else if (atom_residue_name == "THR") return 0;
    else if (atom_residue_name == "TRP") return 0;
    else if (atom_residue_name == "TYR") return 0;
    else if (atom_residue_name == "VAL") return 0;
    else return 0;
}

PdbPicture::PdbPicture(std::string filename, std::vector<char> &chains, float cellSize, float jiggleAlpha, std::vector<int> labels) 
	: Picture(labels), filename(filename), cellSize(cellSize), jiggleAlpha(jiggleAlpha)
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
			int atom_serial_num = std::stoi(line.substr(6, 6));
			std::string atom_name = trim(line.substr(12, 4));
			std::string atom_residue_name = trim(line.substr(17, 3));
			char atom_chain_id = line[21];
			int atom_residue_seq = std::stoi(line.substr(22, 4));
			float atom_coord_x = std::stof(line.substr(30, 8));
			float atom_coord_y = std::stof(line.substr(38, 8));
			float atom_coord_z = std::stof(line.substr(46, 8));
			std::string atom_element = trim(line.substr(76, 2));

			if (std::find(chains.begin(), chains.end(), atom_chain_id) != chains.end())
			{
				atoms.push_back(std::tuple<float, float, float>(atom_coord_x, atom_coord_y, atom_coord_z));
				atomTypes.push_back(codifyAtom(atom_element, atom_name));
                hydrophobicity.push_back(getHydrophobicity(atom_residue_name));
                polarity.push_back(getPolarity(atom_residue_name));
                charge.push_back(getCharge(atom_residue_name));
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
	for (int c = 0; c < 11; c++)
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

			float featureVector[11] = {};

			featureVector[atomTypes[i]] = 1;
            featureVector[8] = hydrophobicity[i] / 10;
            featureVector[9] = polarity[i]; 
            featureVector[10] = charge[i]; 

			features.insert(features.end(), featureVector, featureVector + 11);
		}
		else
		{
			int featureIndex = featuresIndexPerSpatialSite[nSpatialSites - startNSpatialSites - 1];
			features[featureIndex + atomTypes[i]]++;
		}
	}
}