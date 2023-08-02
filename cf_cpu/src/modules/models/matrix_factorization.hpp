#pragma once

#include "model.hpp"

namespace cf
{
namespace modules  
{
namespace models
{

class MatrixFactorization : public Model
{
  public:
    MatrixFactorization(const std::shared_ptr<CFConfig> config, val_t* user_weights, val_t* item_weights, idx_t start_item_id, idx_t end_item_id);
    ~MatrixFactorization() = default;

    val_t forward_backward(idx_t user_id, idx_t pos_id, std::vector<idx_t>& neg_ids, 
        const std::shared_ptr<CFModules> cf_modules, memory::ThreadBuffer* t_buf, behavior_aggregators::BehaviorAggregator* behavior_aggregator,
        std::unordered_map<idx_t, std::vector<val_t> > &) override;
};

}
}
}