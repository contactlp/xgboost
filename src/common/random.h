/*!
 * Copyright 2015 by Contributors
 * \file random.h
 * \brief Utility related to random.
 * \author Tianqi Chen
 */
#ifndef XGBOOST_COMMON_RANDOM_H_
#define XGBOOST_COMMON_RANDOM_H_

#include <rabit/rabit.h>
#include <xgboost/logging.h>
#include <algorithm>
#include <vector>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <random>

#include "xgboost/host_device_vector.h"
#include "io.h"
// #include "choice.cpp"
// #include "choice_cummulative.cpp"

namespace xgboost
{
  namespace common
  {
    /*!
 * \brief Define mt19937 as default type Random Engine.
 */
    using RandomEngine = std::mt19937;

#if XGBOOST_CUSTOMIZE_GLOBAL_PRNG
    /*!
 * \brief An customized random engine, used to be plugged in PRNG from other systems.
 *  The implementation of this library is not provided by xgboost core library.
 *  Instead the other library can implement this class, which will be used as GlobalRandomEngine
 *  If XGBOOST_RANDOM_CUSTOMIZE = 1, by default this is switched off.
 */
    class CustomGlobalRandomEngine
    {
    public:
      /*! \brief The result type */
      using result_type = uint32_t;
      /*! \brief The minimum of random numbers generated */
      inline static constexpr result_type min()
      {
        return 0;
      }
      /*! \brief The maximum random numbers generated */
      inline static constexpr result_type max()
      {
        return std::numeric_limits<result_type>::max();
      }
      /*!
   * \brief seed function, to be implemented
   * \param val The value of the seed.
   */
      void seed(result_type val);
      /*!
   * \return next random number.
   */
      result_type operator()();
    };

    /*!
 * \brief global random engine
 */
    typedef CustomGlobalRandomEngine GlobalRandomEngine;

#else
    /*!
 * \brief global random engine
 */
    using GlobalRandomEngine = RandomEngine;
#endif // XGBOOST_CUSTOMIZE_GLOBAL_PRNG

    /*!
 * \brief global singleton of a random engine.
 *  This random engine is thread-local and
 *  only visible to current thread.
 */
    GlobalRandomEngine &GlobalRandom(); // NOLINT(*)

    /**
 * \class ColumnSampler
 *
 * \brief Handles selection of columns due to colsample_bytree, colsample_bylevel and
 * colsample_bynode parameters. Should be initialised before tree construction and to
 * reset when tree construction is completed.
 */

    class ColumnSampler
    {
      template <typename VT>
      void printv(std::vector<VT> v)
      {
        // std::cout << "\n--func printv \n";
        for (int i = 0; i < v.size(); i++)
        {
          std::cout << v.at(i) << " ";
        }
        std::cout << "\n";
      }

      // template <typename CV>
      void convert_vec_to_float(std::vector<int> colsample_bytree_weight, int colsample_bytree_weight_factor, std::vector<float> &output)
      {
        // std::cout << "\n--func convert_vec_to_float \n";
        for (int i = 0; i < colsample_bytree_weight.size(); i++)
        {
          float fi = static_cast<float>(colsample_bytree_weight.at(i)) / static_cast<float>(colsample_bytree_weight_factor);
          output.push_back(fi);
        }
      }

      // template <typename N>
      void normalize(std::vector<float> a, std::vector<float> &output)
      {
        // std::cout << "\n--func normalize \n";
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

      // template <typename CU>
      void cumulative(std::vector<float> a, std::vector<float> &output)
      {
        // std::cout << "\n--func cumulative \n";
        float running_total = 0;
        for (int i = 0; i < a.size(); i++)
        {
          running_total += a.at(i);
          output.push_back(running_total);
        }
        // std::cout << "\n--func cumulative end \n";
      }

      // template <typename FI>
      float find_index_less_or_equal(std::vector<float> a, float n)
      {
        // std::cout << "\n--func find_index_less_or_equal \n";
        // std::cout << "\n n : " << n << "length of input:" << a.size() << "\n";
        // printv(a);

        if (n > 1)
        {
          // std::cout << "\n ------------------------------------ n is bigger than cummularive sum; ERROR ";
          return a.size() - 1;
        }
        int i = 0;

        for (i; i < a.size(); i++)
        {
          float item = a.at(i);
          if (n < item)
          {
            // std::cout << "\n n < item \n"
            //          << n << "<" << item << " ; i : " << i;
            break;
          }
        }
        // std::cout << "\n n " << n << "; i" << i;
        // std::cout << "\n--func find_index_less_or_equal -- end \n";
        return i;
      }

      template <typename TUU>
      int choice_c(std::vector<TUU> input, std::vector<float> p)
      {
        // std::cout << "\n --func choice_c started \n";
        // // std::cout << "\nChoice started\n";

        if (input.size() != p.size())
        {
          // std::cout << "\ninput vector and probability vector size is not the same \n";
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
        // selected = input.at(index);

        // std::cout << "\n --func choice_c ended \n";
        return index;
      }

      // template <typename TUU>
      // std::vector<TUU> choice(std::vector<TUU> a, int size, bool replace, std::vector<int> p)

      template <typename TUU>
      std::vector<TUU> choice_n(std::vector<TUU> input, int n, bool replace, std::vector<int> p)
      {
        // std::cout << "\n--func choice_n \n";
        // std::cout << "\n input size: " << input.size() << "  p size :" << p.size() << " --\n";
        std::vector<TUU> output;

        if (input.size() != p.size())
        {
          // std::cout << "\n------------------------------------------------------------------------------input and p value length is not the same --\n ";
          // std::cout << "\n input size: " << input.size() << "  p size :" << p.size() << " --\n";
          return output;
        }

        if (input.size() < n)
        {
          // std::cout << "\n------------------------------------------------------------------------------input size: " << input.size() << "  n size : " << n << "--\n";

          return output;
        }
        std::vector<float> pf = {};
        convert_vec_to_float(p, 1, pf);
        // std::cout << "\n pf \n";
        // printv(pf);

        for (int i = 0; i < n; i++)
        {

          int index = choice_c(input, pf);
          // std::cout << "choice_c index: " << index;
          if (index >= input.size())
          {
            // std::cout << "index is bigger than length of inpt vector";
            index -= 1;
          }
          float item = input.at(index);
          output.push_back(item);
          input.erase(input.begin() + index);
          pf.erase(pf.begin() + index);
          //
        }
        return output;
      }

      std::shared_ptr<HostDeviceVector<bst_feature_t>> feature_set_tree_;
      std::map<int, std::shared_ptr<HostDeviceVector<bst_feature_t>>> feature_set_level_;
      float colsample_bylevel_{1.0f};
      float colsample_bytree_{1.0f};
      float colsample_bynode_{1.0f};
      std::vector<int> colsample_bytree_weight_{};
      int colsample_bytree_weight_factor_;
      GlobalRandomEngine rng_;

      std::shared_ptr<HostDeviceVector<bst_feature_t>> ColSample(
          std::shared_ptr<HostDeviceVector<bst_feature_t>> p_features, float colsample)
      {
        if (colsample == 1.0f)
          return p_features;
        const auto &features = p_features->HostVector();

        // printv(features);

        CHECK_GT(features.size(), 0);
        int n = std::max(1, static_cast<int>(colsample * features.size()));
        auto p_new_features = std::make_shared<HostDeviceVector<bst_feature_t>>();
        auto &new_features = *p_new_features;
        new_features.Resize(features.size());
        std::copy(features.begin(), features.end(),
                  new_features.HostVector().begin());
        std::shuffle(new_features.HostVector().begin(),
                     new_features.HostVector().end(), rng_);

        new_features.Resize(n);
        // uniform weight
        // std::vector<float> p{0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05};
        // weight2
        // std::vector<float> p{0.048, 0.024, 0.036000000000000004, 0.036000000000000004, 0.024, 0.06, 0.048, 0.012, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06};

        // std::cout << "colsample_bytree_weight_ vector used: \n";
        // printv(colsample_bytree_weight_);

        //const auto f = choice(features, n, false, colsample_bytree_weight_);
        const auto f = choice_n(features, n, false, colsample_bytree_weight_);

        // std::cout << "choice_n\n";

        p_new_features->HostVector() = f;
        const auto &fr = p_new_features->HostVector();
        // printv(fr);

        std::sort(new_features.HostVector().begin(),
                  new_features.HostVector().end());
        return p_new_features;
      }

    public:
      /**
   * \brief Column sampler constructor.
   * \note This constructor manually sets the rng seed
   */
      explicit ColumnSampler(uint32_t seed)
      {
        rng_.seed(seed);
      }

      /**
  * \brief Column sampler constructor.
  * \note This constructor synchronizes the RNG seed across processes.
  */
      ColumnSampler()
      {
        uint32_t seed = common::GlobalRandom()();
        rabit::Broadcast(&seed, sizeof(seed), 0, "seed");
        rng_.seed(seed);
      }

      /**
   * \brief Initialise this object before use.
   *
   * \param num_col
   * \param colsample_bynode
   * \param colsample_bylevel
   * \param colsample_bytree
   * \param skip_index_0      (Optional) True to skip index 0.
   */
      void Init(int64_t num_col, float colsample_bynode, float colsample_bylevel,
                float colsample_bytree, bool skip_index_0 = false,
                std::vector<int> colsample_bytree_weight = {},
                int colsample_bytree_weight_factor = 10000)
      {
        colsample_bylevel_ = colsample_bylevel;
        colsample_bytree_ = colsample_bytree;
        colsample_bynode_ = colsample_bynode;
        colsample_bytree_weight_factor_ = colsample_bytree_weight_factor;

        colsample_bytree_weight_ = colsample_bytree_weight;

        // std::cout << "\n colsample_bytree_weight_: \n";
        // std::cout << colsample_bytree_weight_factor_;
        // std::cout << "\n";

        // printv(colsample_bytree_weight_);

        if (feature_set_tree_ == nullptr)
        {
          feature_set_tree_ = std::make_shared<HostDeviceVector<bst_feature_t>>();
        }
        Reset();

        int begin_idx = skip_index_0 ? 1 : 0;
        feature_set_tree_->Resize(num_col - begin_idx);
        std::iota(feature_set_tree_->HostVector().begin(),
                  feature_set_tree_->HostVector().end(), begin_idx);

        feature_set_tree_ = ColSample(feature_set_tree_, colsample_bytree_);
      }

      /**
   * \brief Resets this object.
   */
      void Reset()
      {
        feature_set_tree_->Resize(0);
        feature_set_level_.clear();
      }

      /**
   * \brief Samples a feature set.
   *
   * \param depth The tree depth of the node at which to sample.
   * \return The sampled feature set.
   * \note If colsample_bynode_ < 1.0, this method creates a new feature set each time it
   * is called. Therefore, it should be called only once per node.
   * \note With distributed xgboost, this function must be called exactly once for the
   * construction of each tree node, and must be called the same number of times in each
   * process and with the same parameters to return the same feature set across processes.
   */
      std::shared_ptr<HostDeviceVector<bst_feature_t>> GetFeatureSet(int depth)
      {
        if (colsample_bylevel_ == 1.0f && colsample_bynode_ == 1.0f)
        {
          return feature_set_tree_;
        }

        if (feature_set_level_.count(depth) == 0)
        {
          // Level sampling, level does not yet exist so generate it
          feature_set_level_[depth] = ColSample(feature_set_tree_, colsample_bylevel_);
        }
        if (colsample_bynode_ == 1.0f)
        {
          // Level sampling
          return feature_set_level_[depth];
        }
        // Need to sample for the node individually
        return ColSample(feature_set_level_[depth], colsample_bynode_);
      }
    };

  } // namespace common
} // namespace xgboost
#endif // XGBOOST_COMMON_RANDOM_H_
