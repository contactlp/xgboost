#include <iostream>
#include <vector>
#include <vector>
#include <numeric>
#include <random>

template <typename VT>
void printv(std::vector<VT> v)
{
    for (int i = 0; i < v.size(); i++)
    {
        std::cout << v.at(i) << " ";
    }
    std::cout << "\n";
}

void convert_vec_to_float(std::vector<int> colsample_bytree_weight, int colsample_bytree_weight_factor, std::vector<float> &output)
{

    for (int i = 0; i < colsample_bytree_weight.size(); i++)
    {
        float fi = static_cast<float>(colsample_bytree_weight.at(i)) / static_cast<float>(colsample_bytree_weight_factor);
        output.push_back(fi);
    }
}

void normalize(std::vector<float> a, std::vector<float> &output)
{
    float sum_of_elems = 0;
    sum_of_elems = std::accumulate(a.begin(), a.end(),
                                   decltype(a)::value_type(0));
    for (int i = 0; i < a.size(); i++)
    {
        float item = a.at(i);
        float normalized = item / sum_of_elems;

        output.push_back(normalized);
    }
}

void cumulative(std::vector<float> a, std::vector<float> &output)
{
    float running_total = 0;
    for (int i = 0; i < a.size(); i++)
    {
        running_total += a.at(i);
        output.push_back(running_total);
    }
}

int find_index_less_or_equal(std::vector<float> a, float n)
{
    int index = 0;

    for (int i = 0; i < a.size(); i++)
    {
        float item = a.at(i);
        if (n >= item)
        {
            index = i + 1;
        }
        else
        {
            return index;
        }
    }
    return index;
}

float choice(std::vector<float> input, std::vector<float> p)
{
    // std::cout << "\nChoice started\n";

    if (input.size() != p.size())
    {
        std::cout << "\ninput vector and probability vector size is not the same \n";
        return -1.0;
    }
    int conversion = 1; // will be used to divide by 1.0
    // std::vector<float> float_vector = {};
    std::vector<float> normalized = {};
    std::vector<float> cumulatived = {};

    // convert_vec_to_float(p, conversion, float_vector);
    normalize(p, normalized);
    // printv(normalized);

    cumulative(normalized, cumulatived);
    // printv(cumulatived);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1); //uniform distribution between 0 and 1
    float r = dis(gen);

    int index = find_index_less_or_equal(cumulatived, r);
    float selected;
    selected = input.at(index);

    // std::cout << "\nChoice ended\n";
    return selected;
}

// int main()
// {

//     std::vector<float> x = {0.1, 0.2, 0.3, 0.4, 0.5};
//     std::vector<float> y = {0.5, 0.3, 0.2, 0.07, 0.03};

//     // std::vector<int> a = {1, 2, 3, 4, 5};
//     // int b = 1;
//     // std::vector<float> c = {};
//     // std::vector<float> d = {};
//     // std::vector<float> e = {};
//     // convert_vec_to_float(a, b, c);
//     // printv(c);
//     // normalize(c, d);
//     // printv(d);
//     // cumulative(d, e);
//     // printv(e);
//     // int index = find_index_less_or_equal(e, 0.05);
//     // std::cout << index;

//     for (int i = 0; i <= 10; i++)
//     {
//         std::cout << choice(x, y);
//         std::cout << '\n';
//     }

//     return 0;
// }
