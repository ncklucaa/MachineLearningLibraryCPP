#include "macLearning.h"

LinearRegression::LinearRegression(double lr, int epochs)
    : learningRate(lr), epochs(epochs) {}

void LinearRegression::fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y) 
{
    const auto nFeatures = X[0].size();
    weights.resize(nFeatures, 0.0);

    for (int epoch = 0; epoch < epochs; ++epoch) 
    {
        std::vector<double> dw(nFeatures, 0.0);
        double db = 0.0;

        for (size_t i = 0; i < X.size(); ++i) 
        {
            const auto prediction = predict(X[i]);
            const auto error = prediction - y[i];

            for (size_t j = 0; j < nFeatures; ++j) 
            {
                dw[j] += error * X[i][j];
            }
            db += error;
        }

        for (size_t j = 0; j < nFeatures; ++j) 
        {
            weights[j] -= learningRate * dw[j] / static_cast<double>(X.size());
        }
        bias -= learningRate * db / static_cast<double>(X.size());
    }
}

double LinearRegression::predict(const std::vector<double>& x) const 
{
    double y = bias;

    for (size_t i = 0; i < x.size(); ++i) 
    {
        y += weights[i] * x[i];
    }
    return y;
}