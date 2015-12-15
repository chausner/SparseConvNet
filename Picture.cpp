#include "Picture.h"
std::string Picture::identify() { return std::string(); }
Picture::Picture(std::vector<int> labels) : labels(labels) {}
Picture::~Picture() {}
