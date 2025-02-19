#include "macLearning.h"
#include <random>

KMeans::KMeans(int k) : k(k) {}

double KMeans::euclideanDistance(const std::vector<double>& a, const std::vector<double>& b) const 
{
    double distance = 0.0;

    for (size_t i = 0; i < a.size(); ++i) 
    {
        distance += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(distance);
}

void KMeans::fit(const std::vector<std::vector<double>>& X, int maxIterations) 
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dist(0, X.size() - 1);

    centroids.resize(k);

    for (int i = 0; i < k; ++i) 
    {
        centroids[i] = X[dist(gen)];
    }

    for (int iter = 0; iter < maxIterations; ++iter) 
    {
        std::vector<std::vector<int>> clusters(k);

        for (size_t i = 0; i < X.size(); ++i) 
        {
            double minDist = std::numeric_limits<double>::max();
            int clusterIdx = 0;

            for (int j = 0; j < k; ++j) 
            {
                const auto dist = euclideanDistance(X[i], centroids[j]);

                if (dist < minDist) 
                {
                    minDist = dist;
                    clusterIdx = j;
                }
            }
            clusters[clusterIdx].push_back(static_cast<int>(i));
        }

        for (int j = 0; j < k; ++j) 
        {
            std::vector<double> newCentroid(X[0].size(), 0.0);

            for (const auto& idx : clusters[j]) 
            {
                for (size_t d = 0; d < X[idx].size(); ++d) 
                {
                    newCentroid[d] += X[idx][d];
                }
            }
            if (!clusters[j].empty()) 
            {
                for (size_t d = 0; d < newCentroid.size(); ++d) 
                {
                    newCentroid[d] /= clusters[j].size();
                }
            }
            centroids[j] = newCentroid;
        }
    }
}

std::vector<int> KMeans::predict(const std::vector<std::vector<double>>& X) const 
{
    std::vector<int> labels(X.size());

    for (size_t i = 0; i < X.size(); ++i) 
    {
        double minDist = std::numeric_limits<double>::max();
        int clusterIdx = 0;

        for (int j = 0; j < k; ++j) 
        {
            const auto dist = euclideanDistance(X[i], centroids[j]);

            if (dist < minDist) 
            {
                minDist = dist;
                clusterIdx = j;
            }
        }
        labels[i] = clusterIdx;
    }
    return labels;
}