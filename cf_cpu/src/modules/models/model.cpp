#include "model.hpp"

namespace cf
{
namespace modules  
{
namespace models
{

Model::Model(const std::shared_ptr<CFConfig> config, val_t* user_weights, val_t* item_weights,
             idx_t start_item_id, idx_t end_item_id,
             idx_t start_user_id, idx_t end_user_id)
{
    this->user_embedding = new embeddings::Embedding(end_user_id - start_user_id, config->emb_dim, user_weights, start_user_id, end_user_id);
    this->item_embedding = new embeddings::Embedding(end_item_id - start_item_id, config->emb_dim, item_weights, start_item_id, end_item_id);
    this->iterations = 0;
}

val_t* Model::read_embedding(embeddings::Embedding* embedding, idx_t idx, val_t* emb_buf)
{
    val_t* emb = embedding->read_weights(idx, emb_buf);
    return emb;
}

void Model::write_embedding(embeddings::Embedding* embedding, idx_t idx, val_t* emb_buf)
{
    embedding->write_weights(idx, emb_buf);
}

val_t* Model::read_gradient(embeddings::Embedding* embedding, idx_t idx, val_t* grad_buf)
{
    val_t* grad = embedding->read_grads(idx, grad_buf);
    return grad;
}

void Model::write_gradient(embeddings::Embedding* embedding, idx_t idx, val_t* grad_buf)
{
    embedding->write_grads(idx, grad_buf);
}

}
}
}