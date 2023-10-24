#pragma once

#include <map>
#include <string>

#include "../../splatt/base.h"
#include "../../splatt/util.h"

namespace cf
{
namespace modules
{
namespace memory
{

struct ThreadBuffer
{
    ThreadBuffer(idx_t emb_dim, idx_t num_negs, idx_t batch_size, idx_t tile_size)
    {
        this->emb_dim = emb_dim;
        this->num_negs = num_negs;

        user_emb_buf = static_cast<val_t*>(splatt_malloc(emb_dim * sizeof(val_t)));
        user_grad_buf = static_cast<val_t*>(splatt_malloc(emb_dim * sizeof(val_t)));
        pos_grad_buf = static_cast<val_t*>(splatt_malloc(emb_dim * sizeof(val_t)));
        neg_emb_buf0 = static_cast<val_t*>(splatt_malloc(emb_dim * sizeof(val_t)));
        neg_emb_buf1 = static_cast<val_t*>(splatt_malloc(num_negs * emb_dim * sizeof(val_t)));
        // neg_grad_buf = static_cast<val_t*>(splatt_malloc(num_negs * emb_dim * sizeof(val_t)));
        neg_grad_buf = static_cast<val_t*>(splatt_malloc(emb_dim * sizeof(val_t)));
        tiled_neg_emb_buf = static_cast<val_t*>(splatt_malloc(tile_size * emb_dim * sizeof(val_t)));
        pos_item_ids = static_cast<idx_t*>(splatt_malloc(batch_size * sizeof(idx_t)));
        user_ids = static_cast<idx_t*>(splatt_malloc(batch_size * sizeof(idx_t)));
        pos_emb_buf = static_cast<val_t*>(splatt_malloc(batch_size * emb_dim * sizeof(val_t)));   // positive
        user_emb_bufs = static_cast<val_t*>(splatt_malloc(batch_size * emb_dim * sizeof(val_t)));   // user


        time_map["data"] = 0.0;
        time_map["f_b"] = 0.0; // forward_backward
        time_map["forward"] = 0.0;
        time_map["backward"] = 0.0;
        time_map["read_emb"] = 0.0;
        time_map["dot"] = 0.0;
        time_map["norm"] = 0.0;
        time_map["loss"] = 0.0;
        time_map["aggr_f"] = 0.0;
        time_map["read_his"] = 0.0;
        time_map["his_mm"] = 0.0;
        time_map["grad"] = 0.0;
        time_map["reg"] = 0.0;
        time_map["write_emb"] = 0.0;
        time_map["aggr_b"] = 0.0;

    }

    ~ThreadBuffer()
    {
        splatt_free(user_grad_buf);
        splatt_free(pos_grad_buf);
        splatt_free(neg_grad_buf);
        splatt_free(user_emb_buf);
        splatt_free(pos_emb_buf);
        splatt_free(neg_emb_buf0);
        splatt_free(neg_emb_buf1);
        splatt_free(tiled_neg_emb_buf);
        splatt_free(pos_item_ids);
        splatt_free(user_ids);
        splatt_free(user_emb_bufs);
        time_map.clear();
    }

    idx_t emb_dim;
    idx_t num_negs;
    val_t* user_emb_buf;
    val_t* user_grad_buf;
    val_t* pos_emb_buf;
    val_t* pos_grad_buf;
    val_t* neg_emb_buf0;
    val_t* neg_emb_buf1;
    val_t* neg_grad_buf;
    val_t* tiled_neg_emb_buf;   // tiled negative embeddings
    idx_t* pos_item_ids;
    idx_t* user_ids;
    val_t* user_emb_bufs;   // user embeddings
    val_t* his_emb_buf;

    std::map<std::string, double> time_map;
};

}
}
}
