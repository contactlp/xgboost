// cd "/home/lpatel/eclipse-workspace/xgboost/src/common" &&g++-8 -std=c++17 choice.cpp -o choice.out && ./choice.out
#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <string>
#include <iterator>
// #include <experimental/algorithm>
// #include "choice.hpp"

// template <typename VT>
// void printv(std::vector<VT> v)
// {
//     for (int i = 0; i < v.size(); i++)
//     {
//         std::cout << v.at(i) << " ";
//     }
//     std::cout << "\n";
// }

template <typename VT>
void convert_vec_to_float(std::vector<int> colsample_bytree_weight, int colsample_bytree_weight_factor, std::vector<float> &output)
{

    for (int i = 0; i < colsample_bytree_weight.size(); i++)
    {
        float fi = static_cast<float>(colsample_bytree_weight.at(i)) / static_cast<float>(colsample_bytree_weight_factor);
        output.push_back(fi);
    }
}

template <typename T>
inline void remove(std::vector<T> &v, const T &item)
{
    v.erase(std::remove(v.begin(), v.end(), item), v.end());
}

// template <typename TU>
// std::vector<TU> scale_up(std::vector<TU> a, std::vector<float> p)

// {
//     std::vector<TU> ap;

//     for (int i = 0; i < p.size(); i++)
//     {

//         TU ai = a.at(i);
//         float pif = p.at(i);

//         int pin = static_cast<int>(roundf(pif * 100000));

//         for (int i = 0; i < pin; i++)
//         {
//             ap.push_back(ai);
//         }
//     }
//     return ap;
// }

template <typename TU>
std::vector<TU> scale_up_int(std::vector<TU> a, std::vector<int> p)

{
    std::vector<TU> ap;

    for (int i = 0; i < p.size(); i++)
    {

        TU ai = a.at(i);
        int pif = p.at(i);

        // int pin = static_cast<int>(roundf(pif * 100000));
        int pin = pif;

        for (int i = 0; i < pin; i++)
        {
            ap.push_back(ai);
        }
    }
    return ap;
}

template <typename TUU>
std::vector<TUU> choice(std::vector<TUU> a, int size, bool replace, std::vector<int> p)
{
    if (replace == true)
    {
        std::cout << "Not Implemented \n\n";
        return a;
    }

    if (size >= a.size())
    {
        std::cout << "No need if choosing, subsample size is bigger or equal to sample \n\n";
        return a;
    }

    if (p.size() != a.size())
    {
        std::cout << "p and a should have the same size \n\n";
        return a;
    }

    std::vector<TUU> ap;
    // ap = scale_up(a, p);
    ap = scale_up_int(a, p);

    std::vector<TUU>
        ap_out;

    for (int i = 0; i < size; i++)
    {

        std::vector<TUU> temp;
        temp.resize(ap.size());
        // std::experimental::sample(
        //     ap.begin(),
        //     ap.end(),
        //     std::back_inserter(temp),
        //     1,
        //     std::mt19937{std::random_device{}()});

        temp = ap;
        std::random_device rd;
        std::mt19937 g(rd());

        std::shuffle(temp.begin(),
                     temp.end(),
                     g);

        ap_out.push_back(temp.at(0));
        remove(ap, temp.at(0));
    }
    return ap_out;
}

// int main()
// {

//     // std::vector<int> a{1,      2,  3, 4, 5 };
//     // std::vector<float> p{0.199, 1, 3, 0.6, 0};
//     // std::vector<int> ap_out;
//     // std::vector<int> features_weighted_out;

//     // ap_out = choice(a,3,false,p);
//     // printv (ap_out);

//     // // std::vector<float> p{0, 1, 2 ,3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
//     std::vector<int> a{1, 2, 3, 4, 5};
//     int b = 100;
//     std::vector<float> output{};
//     convert_vec_to_float(a, b, output);
//     printv(output);
// }
