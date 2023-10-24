
#include "negative_sampler.hpp"

namespace cf
{
namespace modules
{
namespace negative_samplers
{

NegativeSampler::NegativeSampler(const std::shared_ptr<CFConfig> config, idx_t seed, idx_t tile_space)
{
    this->num_negs = config->num_negs;
    this->neg_sampler = nullptr;
}

}
}
}