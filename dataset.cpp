#include "macLearning.h"
#include <fstream>
#include <sstream>

void Dataset::loadFromCSV(const std::string& filename, bool hasLabels)
{
    std::ifstream file("normalized_data.csv");
    if (!file.is_open())
    {
        throw std::runtime_error("file could not be opened");
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