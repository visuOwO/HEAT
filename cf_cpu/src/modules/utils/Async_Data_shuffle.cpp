//
// Created by hld on 8/29/23.
//

#include "Async_Data_shuffle.h"


namespace cf {
    namespace modules {
        void Async_Data_shuffle::shuffle_and_update_item_grads(std::unordered_map<idx_t, std::vector<val_t>> &grads,
                                                               embeddings::Embedding *item_embeddings) {
            std::unordered_map<idx_t, std::vector<idx_t>> grads_map;

            idx_t k = total_cols / world_size;
            idx_t r = total_cols % world_size;

            //printf("shuffle grads, size is: %zu\n", grads.size());
            //initialize the map
            for (auto i = 0; i < world_size; i++) {
                grads_map[i] = std::vector<idx_t>();
            }

            for (auto &entry: grads) {
                idx_t idx = entry.first;
                auto &grad = entry.second;
                idx_t part = idx / k;
                if (idx - part * k < r) {
                    part++;
                }
                grads_map[part].push_back(idx);
            }

            // update local grads
            for (auto i: grads_map[rank]) {
                auto *updated_item_embeddings = new val_t[emb_dim];
                item_embeddings->weights->read_row(i - item_embeddings->start_idx, updated_item_embeddings);
                cf_modules->optimizer->sparse_step(updated_item_embeddings, grads.at(i).data());
                item_embeddings->weights->write_row(i - item_embeddings->start_idx, updated_item_embeddings);
            }

            std::vector<MPI_Request> send_requests(world_size-1);
            std::vector<MPI_Request> recv_requests(world_size-1);

            std::unordered_map<idx_t, std::vector<idx_t>> received_idx;
            std::unordered_map<idx_t, val_t*> received_data;

            for (int i = 0; i < world_size-1; i++) {
                int dst_rank = (rank + i + 1) % world_size;
                idx_t send_size = grads_map[dst_rank].size();
                idx_t recv_size;
                //printf("rank %d, send to %d\n", rank, i);
                auto send_buffer = new val_t[grads_map[dst_rank].size() * emb_dim];
                for (idx_t j = 0; j < grads_map[dst_rank].size(); j++) {
                    memcpy(send_buffer + j * emb_dim, grads.at(grads_map[dst_rank][j]).data(), emb_dim * sizeof(val_t));
                }

                // send number of embeddings
                MPI_Isend((void*) &send_size, 1, MPI_UINT64_T, dst_rank, 0, comm, &send_requests[i]);
                MPI_Irecv((void*) &recv_size, 1, MPI_UINT64_T, dst_rank, 0, comm, &recv_requests[i]);

                received_idx[dst_rank] = std::vector<idx_t>(recv_size);
                // send index of embeddings
                MPI_Isend((void*) grads_map[dst_rank].data(), grads_map[dst_rank].size(), MPI_UINT64_T, dst_rank, 0, comm, &send_requests[i]);
                MPI_Irecv((void*) received_idx[dst_rank].data(), recv_size, MPI_UINT64_T, dst_rank, 0, comm, &recv_requests[i]);

                // send grads
                MPI_Isend((void*) send_buffer, grads_map[dst_rank].size() * emb_dim, MPI_FLOAT, dst_rank, 0, comm, &send_requests[i]);
                received_data[dst_rank] = new val_t[recv_size * emb_dim];
                MPI_Irecv((void*) received_data[dst_rank], recv_size * emb_dim, MPI_FLOAT, dst_rank, 0, comm, &recv_requests[i]);
            }
            MPI_Waitall(world_size - 1, send_requests.data(), MPI_STATUSES_IGNORE);
            MPI_Waitall(world_size - 1, recv_requests.data(), MPI_STATUSES_IGNORE);

            // update received grads
            for (auto &entry: received_idx) {
                auto dst_rank = entry.first;
                auto &idx = entry.second;
                auto *recv_data = received_data[dst_rank];
                for (auto i = 0; i < idx.size(); i++) {
                    auto *updated_item_embeddings = new val_t[emb_dim];
                    item_embeddings->weights->read_row(idx[i] - item_embeddings->start_idx, updated_item_embeddings);
                    cf_modules->optimizer->sparse_step(updated_item_embeddings, recv_data + i * emb_dim);
                    item_embeddings->weights->write_row(idx[i] - item_embeddings->start_idx, updated_item_embeddings);
                }
            }

            // free memory
            for (auto &entry: received_data) {
                delete[] entry.second;
            }
        }

        void Async_Data_shuffle::shuffle_embs(const std::vector<idx_t> &items, val_t *received_item_embeddings,
                                              embeddings::Embedding *item_embeddings) {
            std::unordered_map<idx_t, std::vector<idx_t>> items_map;
            std::unordered_map<idx_t, std::vector<idx_t>> idx_map;
            idx_t k = total_cols / world_size;
            idx_t r = total_cols % world_size;

            //printf("shuffle embs, size is: %zu\n", items.size());
            //initialize the map
            for (auto i = 0; i < world_size; i++) {
                items_map[i] = std::vector<idx_t>();
                idx_map[i] = std::vector<idx_t>();
            }

            for (auto i = 0; i < items.size(); i++) {
                idx_t idx = items[i];
                idx_t part = idx / k;
                if (idx - part * k < r) {
                    part++;
                }
                //printf("idx is %lu, part is %lu, rank is %d\n", idx, part, rank);
                items_map[part].push_back(idx);
                idx_map[part].push_back(i);
            }

            std::vector<std::vector<idx_t>> received_idx(world_size-1, std::vector<idx_t>());
            std::vector<val_t *> received_data(world_size, nullptr);
            std::vector<MPI_Request> send_requests(world_size-1);
            std::vector<MPI_Request> recv_requests(world_size-1);
            std::vector<int> recv_sizes(world_size-1);

            // using MPI_ISend/IRecv to send and receive data
            // send number of embeddings
            for (auto i = 0; i < world_size - 1; i++) {
                int dst_rank = (rank + i + 1) % world_size;
                //printf("rank %d, send to %d\n", rank, i);
                int send_size = items_map[dst_rank].size();
                MPI_Isend(&send_size, 1, MPI_INT, dst_rank, 0, comm, &send_requests[i]);
                MPI_Irecv(&recv_sizes[i], 1, MPI_INT, dst_rank, 0, comm, &recv_requests[i]);
            }
            MPI_Waitall(world_size - 1, send_requests.data(), MPI_STATUSES_IGNORE);
            MPI_Waitall(world_size - 1, recv_requests.data(), MPI_STATUSES_IGNORE);

            // send index of embeddings
            for (idx_t i = 0; i < world_size-1; i++) {
                idx_t dst_rank = (rank + i + 1) % world_size;
                //printf("rank %d, send to %d\n", rank, i);
                received_idx[i].resize(recv_sizes[i]);
                MPI_Isend((void*) items_map[dst_rank].data(), items_map[dst_rank].size(), MPI_UINT64_T, dst_rank, 0, comm, &send_requests[i]);
                MPI_Irecv((void*) received_idx[i].data(), recv_sizes[i], MPI_UINT64_T, dst_rank, 0, comm, &recv_requests[i]);
            }
            MPI_Waitall(world_size - 1, send_requests.data(), MPI_STATUSES_IGNORE);
            MPI_Waitall(world_size - 1, recv_requests.data(), MPI_STATUSES_IGNORE);

            //send embeddings
            for (auto i = 0; i < world_size-1; i++) {
                int dst_rank = (rank + i + 1) % world_size;
                //printf("rank %d, send to %d\n", rank, i);

                // copy data to send buffer
                auto *send_data = new val_t[items_map[dst_rank].size() * emb_dim];
                received_data[dst_rank] = new val_t[recv_sizes[i] * emb_dim];
                auto *recv_data = received_data[dst_rank];
                for (auto j = 0; j < items_map[dst_rank].size(); j++) {
                    idx_t idx = items_map[dst_rank][j];
                    item_embeddings->read_weights(idx-item_embeddings->start_idx, send_data + j * emb_dim);
                }
                MPI_Isend((void*) send_data, items_map[dst_rank].size() * emb_dim, MPI_FLOAT, dst_rank, 0, comm, &send_requests[i]);
                MPI_Irecv((void*) recv_data, recv_sizes[i] * emb_dim, MPI_FLOAT, dst_rank, 0, comm, &recv_requests[i]);
            }
            MPI_Waitall(world_size - 1, send_requests.data(), MPI_STATUSES_IGNORE);
            MPI_Waitall(world_size - 1, recv_requests.data(), MPI_STATUSES_IGNORE);

            // copy received data to received_item_embeddings
            for (auto i = 0; i < world_size-1; i++) {
                auto *recv_data = received_data[i];
                auto dst_rank = (rank + i + 1) % world_size;
                for (auto j = 0; j < items_map[dst_rank].size(); j++) {
                    memcpy(received_item_embeddings + idx_map[dst_rank][j] * emb_dim, recv_data + j * emb_dim,
                           emb_dim * sizeof(val_t));
                }
            }

            // copy local data to received_item_embeddings
            for (auto i = 0; i < items_map[rank].size(); i++) {
                idx_t idx = items_map[rank][i];
                item_embeddings->read_weights(idx-item_embeddings->start_idx, received_item_embeddings + idx_map[rank][i] * emb_dim);
            }

            // free memory
            for (auto i = 0; i < world_size-1; i++) {
                delete[] received_data[i];
            }
        }
    } // cf
} // modules