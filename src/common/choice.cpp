// cd "/home/lpatel/eclipse-workspace/xgboost/src/common" &&g++-8 -std=c++17 choice.cpp -o choice.out && ./choice.out
#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <string>
#include <iterator>
#include <experimental/algorithm>


template <typename VT>
void printv(std::vector<VT> v)
{
    for (int i = 0; i < v.size(); i++)
    {
        std::cout << v.at(i) << " ";
    }
    std::cout << "\n";
}


// template <typename T>
// inline void remove(std::vector<T> &v, const T &item)
// {
//     v.erase(std::remove(v.begin(), v.end(), item), v.end());
// }


// std::vector<std::string> scale_up(std::vector<std::string> a, std::vector<float> p)

// {
//     std::vector<std::string> ap;

//     for (int i = 0; i < p.size(); i++)
//     {

//         std::string ai = a.at(i);
//         float pif = p.at(i);

//         int pin = static_cast<int>(roundf(pif * 100));

//         for (int i = 0; i < pin; i++)
//         {
//             ap.push_back(ai);
//         }
//     }
//    return ap;
// }



// std::vector<std::string> choice(std::vector<std::string> a, int size, bool replace, std::vector<float> p)
// {
//     if(replace == true){
//         std::cout << "Not Implemented";
//         return a;
//     }

//     if(size >= a.size()){
//         std::cout << "No need if choosing, subsample size is bigger or equal to sample";
//         return a;
//     }

//     if (p.size() != a.size()){
//         std::cout << "p and a should have the same size";
//         return a;
//     }

//     std::vector<std::string> ap;
//     ap = scale_up(a, p);

//     std::vector<std::string> ap_out;

//     for (int i=0; i < size;i++){

//         std::vector<std::string> temp;
//         std::sample(
//             ap.begin(),
//             ap.end(),
//             std::back_inserter(temp),
//             1,
//             std::mt19937{std::random_device{}()});

//         ap_out.push_back(temp.at(0));
//         remove(ap,temp.at(0));

//     }
//     return ap_out;
// }

// int main(){


//     std::vector<std::string> a{"height", "weight", "bmi", "sex", "race"};
//     std::vector<float> p{0.199, 1, 3, 0.6, 0};
//     std::vector<std::string> ap_out;
//     std::vector<std::string> features_weighted_out;

//     ap_out = choice(a,3,false,p);
//     printv (ap_out);

// }


