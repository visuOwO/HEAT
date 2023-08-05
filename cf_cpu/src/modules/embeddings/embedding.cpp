#include "embedding.hpp"

namespace cf
{
namespace modules  
{
namespace embeddings
{

Embedding::Embedding(idx_t num_embs, idx_t emb_dim, val_t* init_weights, idx_t start_idx, idx_t end_idx)
{
    this->weights = new ValArray(num_embs, emb_dim, init_weights);
    this->grads = new ValArray(num_embs, emb_dim, nullptr);

    this->num_embs = num_embs;
    this->emb_dim = emb_dim;    this->num_embs = num_embs;


    printf("Embedding::Embedding: num_embs = %lu, emb_dim = %lu\n", num_embs, emb_dim);

    this->start_idx = start_idx;
    this->end_idx = end_idx;

    printf("Embedding::Embedding: start_idx = %lu, end_idx = %lu\n", start_idx, end_idx);
}

val_t* Embedding::read_weights(idx_t idx, val_t* weight_buf)
{
    if (idx >= this->end_idx || idx < this->start_idx)
    {
        throw std::runtime_error("Embedding::read_weights: idx >= this->end_idx");
    }
    idx_t local_idx = idx - this->start_idx;
    std::memcpy(weight_buf, this->weights->data + (local_idx * this->emb_dim), this->emb_dim * sizeof(val_t));
    return weight_buf;
}

void Embedding::write_weights(idx_t idx, val_t* weight_buf)
{
    if (idx >= this->end_idx || idx < this->start_idx)
    {
        throw std::runtime_error("Embedding::read_weights: idx >= this->end_idx");
    }
    idx_t local_idx = idx - this->start_idx;
    std::memcpy(this->weights->data + (local_idx * this->emb_dim), weight_buf, this->emb_dim * sizeof(val_t));
}

val_t* Embedding::read_grads(idx_t idx, val_t* grad_buf)
{
    if (idx >= this->end_idx || idx < this->start_idx)
    {
        throw std::runtime_error("Embedding::read_weights: idx >= this->end_idx");
    }
    idx_t local_idx = idx - this->start_idx;
    std::memcpy(grad_buf, this->grads->data + (local_idx * this->emb_dim), this->emb_dim * sizeof(val_t));
    return grad_buf;
}

void Embedding::write_grads(idx_t idx, val_t* grad_buf)
{
    if (idx >= this->end_idx || idx < this->start_idx)
    {
        throw std::runtime_error("Embedding::read_weights: idx >= this->end_idx");
    }
    idx_t local_idx = idx - this->start_idx;
    std::memcpy(this->grads->data + (local_idx * this->emb_dim), grad_buf, this->emb_dim * sizeof(val_t));
}

void Embedding::zero_grad()
{
    idx_t num_bytes = this->num_embs * this->emb_dim * sizeof(val_t);
    par_memset(this->grads->data, 0, num_bytes);
}

}
}
}

