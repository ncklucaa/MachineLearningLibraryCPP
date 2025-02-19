#include "macLearning.h"
#include <fstream>
#include <sstream>

void Dataset::loadFromCSV(const std::string& filename, bool hasLabels) 
{
    std::ifstream file(filename);

    if (!file.is_open()) 
    {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string line;

    while (std::getline(file, line)) 
    {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string value;

        while (std::getline(ss, value, ',')) 
        {
            row.push_back(std::stod(value));
        }

        if (hasLabels) 
        {
            labels.push_back(row.back());
            row.pop_back();
        }
        features.push_back(row);
    }
}