//
// Created by hld on 7/20/23.
//

#ifndef CF_CPU_DATA_SHUFFLE_H
#define CF_CPU_DATA_SHUFFLE_H

#include <mpi.h>
#include <vector>
#include <memory>
#include <unordered_map>
#include "splatt/base.h"
#include "embeddings/embedding.hpp"
#include "cf_modules.hpp"

namespace cf {
    namespace modules {

        class Data_shuffle {
        public:
            Data_shuffle() = default;

            ~Data_shuffle() = default;

            template<class T>
            static void shuffle_grad(T *send_data, std::vector<idx_t> &cols, idx_t dst_rank, T *&recv_data,
                                     std::vector<idx_t> &recv_cols);


            static void shuffle_and_update_item_grads(std::unordered_map<idx_t, std::vector<val_t> >& grads,
                                                      embeddings::Embedding *item_embeddings, idx_t total_nums);

            static void shuffle_embs(const std::vector<idx_t>& items, val_t *received_item_embeddings,
                                     embeddings::Embedding *item_embeddings, idx_t total_nums);

            template<class T>
                    static void request_data(T *& requested_data, std::vector<idx_t> &requested_cols, idx_t dst_rank, embeddings::Embedding * item_embeddings);


            static idx_t * process_status;
            static idx_t num_items;
            static idx_t emb_dim;

            static std::shared_ptr<CFModules> cf_modules;

            static MPI_Comm comm;
            static int rank;
            static int world_size;
        private:

        };

    } // cf
} // modules

#endif //CF_CPU_DATA_SHUFFLE_H
