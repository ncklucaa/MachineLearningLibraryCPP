#ifndef MACLEARNING_H
#define MACLEARNING_H

#include <vector>
#include <string>
#include <random> // random num gen

class Dataset 
{
public:
    std::vector<std::vector<double>> features;
    std::vector<double> labels;

    void loadFromCSV(const std::string& filename, bool hasLabels = true);
};

class LinearRegression 
{
private:
    std::vector<double> weights;
    double bias = 0.0; // initialize bias
    double learningRate;
    int epochs;

public:
    explicit LinearRegression(double lr = 0.01, int epochs = 1000);
    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y);
    double predict(const std::vector<double>& x) const; // const
};

class LogisticRegression 
{
private:
    std::vector<double> weights;
    double bias = 0.0; // initialize bias
    double learningRate;
    int epochs;

    double sigmoid(double z) const; // const

public:
    explicit LogisticRegression(double lr = 0.01, int epochs = 1000);
    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y);
    double predictProbability(const std::vector<double>& x) const; // const
    int predict(const std::vector<double>& x) const; // const
};

class KMeans 
{
private:
    int k;
    std::vector<std::vector<double>> centroids;

    double euclideanDistance(const std::vector<double>& a, const std::vector<double>& b) const; // const

public:
    explicit KMeans(int k = 3);
    void fit(const std::vector<std::vector<double>>& X, int maxIterations = 100);
    std::vector<int> predict(const std::vector<std::vector<double>>& X) const; // const
};

void normalize(std::vector<std::vector<double>>& X); // data preprocessing
void trainTestSplit(const std::vector<std::vector<double>>& X, const std::vector<double>& y,
    std::vector<std::vector<double>>& X_train, std::vector<std::vector<double>>& X_test,
    std::vector<double>& y_train, std::vector<double>& y_test, double testSize = 0.2);

double accuracy(const std::vector<double>& y_true, const std::vector<double>& y_pred); // evaluation for metrics
double meanSquaredError(const std::vector<double>& y_true, const std::vector<double>& y_pred);

#endif