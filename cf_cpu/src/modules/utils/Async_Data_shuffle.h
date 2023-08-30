//
// Created by hld on 8/29/23.
//

#ifndef CF_CPU_ASYNC_DATA_SHUFFLE_H
#define CF_CPU_ASYNC_DATA_SHUFFLE_H

#include "Data_shuffle.h"
#include <unordered_map>
#include <mpi.h>

namespace cf {
    namespace modules {

        class Async_Data_shuffle : public Data_shuffle {
        public:
            Async_Data_shuffle() = default;

            ~Async_Data_shuffle() = default;

            static void shuffle_and_update_item_grads(std::unordered_map<idx_t, std::vector<val_t> >& grads,
                                                      embeddings::Embedding *item_embeddings);

            static void shuffle_embs(const std::vector<idx_t>& items, val_t *received_item_embeddings,
                                     embeddings::Embedding *item_embeddings);
        };

    } // cf
} // modules

#endif //CF_CPU_ASYNC_DATA_SHUFFLE_H
