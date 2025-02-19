#include "macLearning.h"

double accuracy(const std::vector<double>& y_true, const std::vector<double>& y_pred) 
{
    int correct = 0;

    for (size_t i = 0; i < y_true.size(); ++i) 
    {
        if (y_true[i] == y_pred[i]) correct++;
    }
    return static_cast<double>(correct) / y_true.size();
}

double meanSquaredError(const std::vector<double>& y_true, const std::vector<double>& y_pred) 
{
    double mse = 0.0;

    for (size_t i = 0; i < y_true.size(); ++i) 
    {
        mse += (y_true[i] - y_pred[i]) * (y_true[i] - y_pred[i]);
    }
    return mse / y_true.size();
}