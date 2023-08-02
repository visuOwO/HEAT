//
// Created by hld on 7/20/23.
//

#include <map>
#include <unordered_map>
#include "Data_shuffle.h"
#include <Eigen/Dense>

namespace cf {
    namespace modules {


        // shuffle gradients from other ranks and update local embeddings
        void
        Data_shuffle::shuffle_and_update_item_grads(const std::vector<idx_t> &neg_items, val_t *neg_item_embeddings,
                                                    embeddings::Embedding *item_embeddings) {
            std::unordered_map<idx_t, std::vector<idx_t>> negs_map;
            idx_t k = total_cols / world_size;
            idx_t r = total_cols % world_size;
            for (auto &neg_item: neg_items) {
                idx_t part = neg_item / k;
                if (neg_item - part * k < r) {
                    part++;
                }
                negs_map[part].push_back(neg_item);
            }

            // update local embeddings
            for (auto i: negs_map[rank]) {
                auto *updated_item_embeddings = new val_t[emb_dim];
                item_embeddings->weights->read_row(i, updated_item_embeddings);
                cf_modules->optimizer->sparse_step(updated_item_embeddings, neg_item_embeddings + i * emb_dim);
                item_embeddings->weights->write_row(i, updated_item_embeddings);
            }

            // shuffle and update embeddings from other ranks
            for (idx_t i = 1; i < world_size; i++) {
                idx_t dst = rank ^ i;
                val_t *recv_data = nullptr;
                std::vector<idx_t> recv_cols;
                shuffle_grad(neg_item_embeddings, negs_map[dst], dst, recv_data, recv_cols);
                Eigen::Map<Eigen::Array<val_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> recv_data_arr(
                        recv_data, recv_cols.size(), emb_dim);
                for (unsigned long idx: recv_cols) {
                    auto *updated_item_embeddings = new val_t[emb_dim];
                    item_embeddings->weights->read_row(idx, updated_item_embeddings);
                    cf_modules->optimizer->sparse_step(updated_item_embeddings, recv_data + idx * emb_dim);
                    item_embeddings->weights->write_row(idx, updated_item_embeddings);
                    delete[] updated_item_embeddings;
                }
                delete[] recv_data;
            }
        }

        // get embeddings from other ranks before computing
        // input: items, received_item_embeddings (pre-allocated buffer for positive embeddings)
        // output: received_item_embeddings (positive embeddings)
        void Data_shuffle::shuffle_embs(const std::vector<idx_t>& items, val_t *received_item_embeddings,
                                        embeddings::Embedding *item_embeddings) {
            std::unordered_map<idx_t, std::vector<idx_t>> items_map;
            std::unordered_map<idx_t, std::vector<idx_t>> idx_map;
            idx_t k = total_cols / world_size;
            idx_t r = total_cols % world_size;
            for (auto i = 0; i < items.size(); i++) {
                idx_t idx = items[i];
                idx_t part = idx / k;
                if (idx - part * k < r) {
                    part++;
                }
                items_map[part].push_back(idx);
                idx_map[part].push_back(i);
            }
            std::vector<idx_t> cnts(world_size, 0);
            std::vector<val_t*> received_data(world_size, nullptr);
            for (auto i = 0; i < world_size; i++) {
                idx_t dst = rank ^ i;
                val_t *recv_data = nullptr;
                std::vector<idx_t> recv_cols;
                request_data(recv_data, items_map[i], dst, item_embeddings);
                for (auto j = 0; j < items_map[dst].size(); j++) {
                    memcpy(received_item_embeddings + idx_map[dst][j] * emb_dim, recv_data + j * emb_dim, emb_dim * sizeof(val_t));
                }
                delete[] recv_data;
            }
        }

        // request data from other ranks
        template<class T>
        void Data_shuffle::request_data(T *requested_data, std::vector<idx_t> &requested_cols, idx_t dst_rank,
                                        embeddings::Embedding *item_embeddings) {
            idx_t requested_count = requested_cols.size();
            idx_t recv_count;
            if (rank < dst_rank) {
                MPI_Send((void *) &requested_count, 1, MPI_UINT64_T, dst_rank, 0, comm);
                MPI_Send((void *) requested_cols.data(), requested_count, MPI_UINT64_T, dst_rank, 0, comm);
                MPI_Recv((void *) &recv_count, 1, MPI_UINT64_T, dst_rank, 0, comm, MPI_STATUS_IGNORE);
                std::vector<val_t> recv_cols(recv_count);
                MPI_Recv((void *) recv_cols.data(), recv_count, MPI_UINT64_T, dst_rank, 0, comm,
                         MPI_STATUS_IGNORE);

                // prepare send data
                auto *send_data = new val_t[recv_count * emb_dim];
                for (unsigned long i = 0; i < recv_count; i++) {
                    item_embeddings->weights->read_row(requested_cols[i], send_data + i * emb_dim);
                }

                MPI_Send((void *) send_data, recv_count * emb_dim, MPI_FLOAT, dst_rank, 0, comm);
                MPI_Recv((void *) requested_data, requested_count * emb_dim, MPI_FLOAT, dst_rank, 0, comm,
                         MPI_STATUS_IGNORE);
                delete[] send_data;
            } else {
                MPI_Recv((void *) &recv_count, 1, MPI_UINT64_T, dst_rank, 0, comm, MPI_STATUS_IGNORE);
                std::vector<val_t> recv_cols(recv_count);
                MPI_Recv((void *) recv_cols.data(), recv_count, MPI_UINT64_T, dst_rank, 0, comm,
                         MPI_STATUS_IGNORE);

                MPI_Send((void *) &requested_count, 1, MPI_UINT64_T, dst_rank, 0, comm);
                MPI_Send((void *) requested_cols.data(), requested_count, MPI_UINT64_T, dst_rank, 0, comm);

                auto *send_data = new val_t[recv_count * emb_dim];
                for (unsigned long i = 0; i < recv_count; i++) {
                    item_embeddings->weights->read_row(requested_cols[i], send_data + i * emb_dim);
                }

                MPI_Recv((void *) requested_data, requested_count * emb_dim, MPI_FLOAT, dst_rank, 0, comm,
                         MPI_STATUS_IGNORE);
                MPI_Send((void *) send_data, recv_count * emb_dim, MPI_FLOAT, dst_rank, 0, comm);
                delete[] send_data;
            }
        }

        // request data from other ranks



        template<class T>
        void Data_shuffle::shuffle_grad(T *send_data, std::vector<idx_t> &cols, idx_t dst_rank, T *recv_data,
                                        std::vector<idx_t> &recv_cols) {
            idx_t send_count = cols.size();
            idx_t recv_count;
            if (rank < dst_rank) {
                MPI_Send((void *) &send_count, 1, MPI_UINT64_T, dst_rank, 0, comm);
                MPI_Send((void *) cols.data(), send_count, MPI_UINT64_T, dst_rank, 0, comm);
                MPI_Send((void *) send_data, send_count * emb_dim, MPI_FLOAT, dst_rank, 0, comm);
                MPI_Recv((void *) &recv_count, 1, MPI_UINT64_T, dst_rank, 0, comm, MPI_STATUS_IGNORE);
                recv_cols.resize(recv_count);
                MPI_Recv((void *) recv_cols.data(), recv_count, MPI_UINT64_T, dst_rank, 0, comm,
                         MPI_STATUS_IGNORE);
                recv_data = new T[recv_count * emb_dim];
                MPI_Recv((void *) recv_data, recv_count * emb_dim, MPI_FLOAT, dst_rank, 0, comm, MPI_STATUS_IGNORE);
            } else {
                MPI_Recv((void *) &recv_count, 1, MPI_UINT64_T, dst_rank, 0, comm, MPI_STATUS_IGNORE);
                recv_cols.resize(recv_count);
                recv_data = new T[recv_count * emb_dim];
                MPI_Recv((void *) recv_cols.data(), recv_count, MPI_UINT64_T, dst_rank, 0, comm,
                         MPI_STATUS_IGNORE);
                MPI_Recv((void *) recv_data, recv_count * emb_dim, MPI_FLOAT, dst_rank, 0, comm, MPI_STATUS_IGNORE);
                MPI_Send((void *) &send_count, 1, MPI_UINT64_T, dst_rank, 0, comm);
                MPI_Send((void *) cols.data(), send_count, MPI_UINT64_T, dst_rank, 0, comm);
                MPI_Send((void *) send_data, send_count * emb_dim, MPI_FLOAT, dst_rank, 0, comm);
            }
        }


    } // cf
} // modules