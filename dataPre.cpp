#include "macLearning.h"

void normalize(std::vector<std::vector<double>>& X) 
{
    for (size_t j = 0; j < X[0].size(); ++j) 
    {
        double minVal = X[0][j], maxVal = X[0][j];

        for (const auto& row : X) 
        {
            if (row[j] < minVal) minVal = row[j];
            if (row[j] > maxVal) maxVal = row[j];
        }

        for (auto& row : X) 
        {
            row[j] = (row[j] - minVal) / (maxVal - minVal);
        }
    }
}

void trainTestSplit(const std::vector<std::vector<double>>& X, const std::vector<double>& y,
    std::vector<std::vector<double>>& X_train, std::vector<std::vector<double>>& X_test,
    std::vector<double>& y_train, std::vector<double>& y_test, double testSize) 
{
    const auto testSamples = static_cast<size_t>(X.size() * testSize);
    for (size_t i = 0; i < X.size(); ++i) 
    {
        if (i < testSamples) 
        {
            X_test.push_back(X[i]);
            y_test.push_back(y[i]);
        }
        else 
        {
            X_train.push_back(X[i]);
            y_train.push_back(y[i]);
        }
    }
}