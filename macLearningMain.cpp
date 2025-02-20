#include "macLearning.h"
#include <iostream>
#include <cstdlib>

int main()
{
    std::cout << "running py\n";

    int result = system("python preprocess.py");

    if (result != 0) 
    {
        std::cerr << "fail to open preprocess.py\n";
        return 1;
    }
    std::cout << "complete\n";

    Dataset data;
    data.loadFromCSV("normalized_data.csv");

    std::vector<std::vector<double>> X_train, X_test;
    std::vector<double> y_train, y_test;
    trainTestSplit(data.features, data.labels, X_train, X_test, y_train, y_test);

    LinearRegression lr;
    lr.fit(X_train, y_train);
    std::vector<double> lrPredictions;

    for (const auto& x : X_test)
    {
        lrPredictions.push_back(lr.predict(x));
    }
    std::cout << "linear regression MSE: " << meanSquaredError(y_test, lrPredictions) << std::endl;

    LogisticRegression logr;
    logr.fit(X_train, y_train);
    std::vector<double> logrPredictions;

    for (const auto& x : X_test)
    {
        logrPredictions.push_back(logr.predict(x));
    }
    std::cout << "logistic regression accuracy %: " << accuracy(y_test, logrPredictions) << std::endl;

    KMeans kmeans(2);
    kmeans.fit(data.features);
    std::vector<int> kmeansLabels = kmeans.predict(data.features);

    std::cout << "kmeans label: ";

    for (const auto& label : kmeansLabels)
    {
        std::cout << label << " ";
    }
    std::cout << std::endl;

    return 0;
}
