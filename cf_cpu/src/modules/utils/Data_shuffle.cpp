//
// Created by hld on 7/20/23.
//


#include "Data_shuffle.h"
#include "thread_buffer.hpp"
#include <Eigen/Dense>

namespace cf {
    namespace modules {


        // shuffle gradients from other ranks and update local embeddings
        __attribute__((unused)) void
        Data_shuffle::shuffle_and_update_item_grads(std::unordered_map<idx_t, std::vector<val_t> > &grads,
                                                    embeddings::Embedding *embeddings, idx_t total_nums) {
            std::unordered_map<idx_t, std::vector<idx_t>> grads_map;
            //printf("update item grads, size is: %zu\n", grads.size());
            //std::vector<std::vector<idx_t>> grads_map(world_size, std::vector<idx_t>());
            idx_t k = total_nums / world_size;
            idx_t r = total_nums % world_size;

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

                /*for (idx_t j = 0; j < emb_dim; j++) {
                    printf("%f ", grad[j]);
                }
                printf("\n");*/

            }

            // update local embeddings
            for (auto i: grads_map[rank]) {
                auto *updated_item_embeddings = new val_t[emb_dim];
                embeddings->weights->read_row(i - embeddings->start_idx, updated_item_embeddings);
                /*for (idx_t j = 0; j < emb_dim; j++) {
                    printf("%f ", updated_item_embeddings[j]);
                }
                printf("\n");*/
                cf_modules->optimizer->sparse_step(updated_item_embeddings, grads.at(i).data());
                /*for (idx_t j = 0; j < emb_dim; j++) {
                    printf("%f ", updated_item_embeddings[j]);
                }
                printf("\n");
                printf("000\n");*/

                embeddings->weights->write_row(i - embeddings->start_idx, updated_item_embeddings);
            }


            // shuffle and update embeddings from other ranks
            for (idx_t i = 1; i < world_size; i++) {
                idx_t dst = rank ^ i;
                val_t *recv_data = nullptr;
                std::vector<idx_t> recv_cols;
                auto send_buffer = new val_t[grads_map[dst].size() * emb_dim];
                for (idx_t j = 0; j < grads_map[dst].size(); j++) {
                    memcpy(send_buffer + j * emb_dim, grads.at(grads_map[dst][j]).data(), emb_dim * sizeof(val_t));
                }
                shuffle_grad(send_buffer, grads_map[dst], dst, recv_data, recv_cols);
                Eigen::Map<Eigen::Array<val_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> recv_data_arr(
                        recv_data, recv_cols.size(), emb_dim);
                for (idx_t j = 0; j < recv_cols.size(); j++) {
                    auto *updated_item_embeddings = new val_t[emb_dim];
                    embeddings->weights->read_row(recv_cols[j] - embeddings->start_idx,
                                                  updated_item_embeddings);
                    cf_modules->optimizer->sparse_step(updated_item_embeddings, recv_data + j * emb_dim);
                    embeddings->weights->write_row(recv_cols[j] - embeddings->start_idx,
                                                   updated_item_embeddings);
                    delete[] updated_item_embeddings;
                }
                delete[] recv_data;
            }
        }

        // get embeddings from other ranks before computing
        // input: items, received_item_embeddings (pre-allocated buffer for positive embeddings)
        // output: received_item_embeddings (positive embeddings)
        __attribute__((unused)) void Data_shuffle::shuffle_embs(const std::vector<idx_t> &items, val_t *received_item_embeddings,
                                        embeddings::Embedding *item_embeddings, idx_t total_nums,
                                        memory::ThreadBuffer *t_buf) {
            std::unordered_map<idx_t, std::vector<idx_t>> items_map;
            std::unordered_map<idx_t, std::vector<idx_t>> idx_map;
            idx_t k = total_nums / world_size;
            idx_t r = total_nums % world_size;

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
            std::vector<idx_t> cnts(world_size, 0);
            std::vector<val_t *> received_data(world_size, nullptr);
            for (auto i = 1; i < world_size; i++) {
                idx_t dst = rank ^ i;
                auto *recv_data = new val_t[items_map[dst].size() * emb_dim];
                std::vector<idx_t> recv_cols;
                request_data(recv_data, items_map[dst], dst, item_embeddings);
                for (auto j = 0; j < items_map[dst].size(); j++) {
                    memcpy(received_item_embeddings + idx_map[dst][j] * emb_dim, recv_data + j * emb_dim,
                           emb_dim * sizeof(val_t));
                }
            }

            // copy local embeddings
            for (auto &ii: t_buf->time_map) {
                printf("%s: %f\t", ii.first.c_str(), ii.second);
            }
            printf("\n");

            for (auto i = 0; i < items_map[rank].size(); i++) {
                idx_t idx = items_map[rank][i];
                //printf("%lu %lu\n", idx_map[rank][i], idx);
                auto *updated_item_embeddings = new val_t[emb_dim];
                item_embeddings->weights->read_row(idx - item_embeddings->start_idx, updated_item_embeddings);
                memcpy(received_item_embeddings + idx_map[rank][i] * emb_dim, updated_item_embeddings,
                       emb_dim * sizeof(val_t));
                /*for (idx_t j = 0; j < emb_dim; j++) {
                    printf("%f ", updated_item_embeddings[j]);
                }
                printf("\n");*/
                delete[] updated_item_embeddings;
            }
            for (auto &ii: t_buf->time_map) {
                printf("%s: %f\t", ii.first.c_str(), ii.second);
            }
            printf("\n");

            // free memory
            for (auto i = 0; i < world_size; i++) {
                delete[] received_data[i];
            }
        }

        void Data_shuffle::update_grad(idx_t * idx_arr, embeddings::Embedding *embeddings, val_t *data) {
            std::unordered_map<int, std::vector<idx_t>> items_map;
            std::unordered_map<int, std::vector<idx_t>> idx_map;
            idx_t k = batch_size / world_size;
            idx_t r = batch_size % world_size;

            //initialize the map
            for (auto i = 0; i < world_size; i++) {
                items_map[i] = std::vector<idx_t>();
                idx_map[i] = std::vector<idx_t>();
            }

            for (auto i = 0; i < batch_size; i++) {
                idx_t idx = idx_arr[i];
                idx_t part = idx / k;
                if (idx - part * k < r) {
                    part++;
                }
                //printf("idx is %lu, part is %lu, rank is %d\n", idx, part, rank);
                items_map[part].push_back(idx);
                idx_map[part].push_back(i);
            }

            std::vector<idx_t> cnts(world_size, 0);
            std::vector<val_t *> received_data(world_size, nullptr);

            //send grads to other ranks
            for (auto i = 1; i < world_size; i++) {
                int dst = i;
                std::vector<idx_t> recv_cols;
                if (rank == 0) {
                    int data_size = items_map[dst].size();
                    MPI_Send((void *) &data_size, 1, MPI_UINT64_T, dst, 0, comm);
                    MPI_Send((void *) items_map[dst].data(), data_size, MPI_UINT64_T, dst, 0, comm);
                    auto *send_data = new val_t[data_size * emb_dim];
                    for (idx_t j = 0; j < data_size; j++) {
                        memcpy(send_data + j * emb_dim, data + idx_map[dst][j] * emb_dim, emb_dim * sizeof(val_t));
                    }
                    MPI_Send((void *) send_data, data_size * emb_dim, MPI_FLOAT, dst, 0, comm);
                } else {
                    int data_size;
                    MPI_Recv((void *) &data_size, 1, MPI_UINT64_T, 0, 0, comm, MPI_STATUS_IGNORE);
                    recv_cols.resize(data_size);
                    MPI_Recv((void *) recv_cols.data(), data_size, MPI_UINT64_T, 0, 0, comm, MPI_STATUS_IGNORE);
                    auto grads = new val_t[data_size * emb_dim];
                    MPI_Recv((void *) grads, data_size * emb_dim, MPI_FLOAT, 0, 0, comm, MPI_STATUS_IGNORE);
                    for (idx_t j = 0; j < data_size; j++) {
                        auto *updated_item_embeddings = new val_t[emb_dim];
                        embeddings->weights->read_row(recv_cols[j] - embeddings->start_idx, updated_item_embeddings);
                        cf_modules->optimizer->sparse_step(updated_item_embeddings, grads + j * emb_dim);
                        embeddings->weights->write_row(recv_cols[j] - embeddings->start_idx, updated_item_embeddings);
                    }
                }
            }

            // update local embeddings
            if (rank == 0) {
                for (auto i = 0; i < items_map[0].size(); i++) {
                    printf("before update\n");
                    idx_t idx = items_map[0][i];
                    auto *updated_item_embeddings = new val_t[emb_dim];
                    embeddings->weights->read_row(idx - embeddings->start_idx, updated_item_embeddings);
                    for (idx_t j = 0; j < emb_dim; j++) {
                        printf("%f ", updated_item_embeddings[j]);
                    }
                    printf("\n");
                    printf("grad is\n");
                    for (idx_t j = 0; j < emb_dim; j++) {
                        printf("%f ", data[idx_map[0][i] * emb_dim + j]);
                    }
                    printf("\n");
                    cf_modules->optimizer->sparse_step(updated_item_embeddings, data + idx_map[0][i] * emb_dim);
                    embeddings->weights->write_row(idx - embeddings->start_idx, updated_item_embeddings);
                    printf("after update\n");
                    for (idx_t j = 0; j < emb_dim; j++) {
                        printf("%f ", updated_item_embeddings[j]);
                    }
                    printf("\n");
                }
            }
        }

        void Data_shuffle::request_emb(idx_t * idx_arr, embeddings::Embedding *embeddings, val_t* data) {
            std::unordered_map<idx_t, std::vector<idx_t>> items_map;
            std::unordered_map<idx_t, std::vector<idx_t>> idx_map;
            idx_t k = batch_size / world_size;
            idx_t r = batch_size % world_size;

            // examine idx array
            /*for (auto i = 0; i < batch_size; i++) {
                printf("%lu ", idx_arr[i]);
            }
            printf("\n");
            printf("test start and end idx: %lu %lu\n", embeddings->start_idx, embeddings->end_idx);*/

            //initialize the map
            for (auto i = 0; i < world_size; i++) {
                items_map[i] = std::vector<idx_t>();
                idx_map[i] = std::vector<idx_t>();
            }

            for (auto i = 0; i < batch_size; i++) {
                idx_t idx = idx_arr[i];
                idx_t part = idx / k;
                if (idx - part * k < r) {
                    part++;
                }
                //printf("idx is %lu, part is %lu, rank is %d\n", idx, part, rank);
                items_map[part].push_back(idx);
                idx_map[part].push_back(i);
            }

            std::vector<idx_t> cnts(world_size, 0);
            std::vector<val_t *> received_data(world_size, nullptr);

            //request data from other ranks
            for (auto i = 1; i < world_size; i++) {
                idx_t dst = rank ^ i;
                std::vector<idx_t> recv_cols;
                if (rank == 0) {
                    int request_size = items_map[dst].size();
                    auto * recv_data = new val_t[request_size * emb_dim];
                    MPI_Send((void *) &request_size, 1, MPI_UINT64_T, dst, 0, comm);
                    MPI_Send((void *) items_map[dst].data(), request_size, MPI_UINT64_T, dst, 0, comm);
                    MPI_Recv((void *) recv_data, request_size * emb_dim, MPI_FLOAT, dst, 0, comm, MPI_STATUS_IGNORE);
                    for (auto j = 0; j < request_size; j++) {
                        memcpy(data + idx_map[dst][j] * emb_dim, recv_data + j * emb_dim, emb_dim * sizeof(val_t));
                    }
                    delete[] recv_data;
                } else {
                    int request_size;
                    MPI_Recv((void *) &request_size, 1, MPI_UINT64_T, 0, 0, comm, MPI_STATUS_IGNORE);
                    recv_cols.resize(request_size);
                    MPI_Recv((void *) recv_cols.data(), request_size, MPI_UINT64_T, 0, 0, comm, MPI_STATUS_IGNORE);
                    auto send_data = new val_t[request_size * emb_dim];
                    for (idx_t j = 0; j < request_size; j++) {
                        embeddings->weights->read_row(recv_cols[j] - embeddings->start_idx, send_data + j * emb_dim);
                    }
                    MPI_Send((void *) send_data, request_size * emb_dim, MPI_FLOAT, 0, 0, comm);
                    delete[] send_data;
                }
            }

            //copy local embeddings
            if (rank == 0) {
                for (auto i = 0; i < items_map[0].size(); i++) {
                    idx_t idx = items_map[0][i];
                    auto *tmp = new val_t[emb_dim];
                    embeddings->weights->read_row(idx - embeddings->start_idx, tmp);
                    //printf("idx is %lu, part is %lu, rank is %d\n", idx, idx_map[0][i], rank);
                    memcpy(data + idx_map[0][i] * emb_dim, tmp, emb_dim * sizeof(val_t));
                    delete[] tmp;
                }
            }

            // examine all embeddings
            /*printf("rank %d, batch size is %d\n", rank, batch_size);
            for (auto i = 0; i < batch_size; i++) {
                for (auto j = 0; j < emb_dim; j++) {
                    printf("%f ", data[i * emb_dim + j]);
                }
                printf("\n");
            }
            printf("\n");*/
        }

        // request data from other ranks
        template<class T>
        void Data_shuffle::request_data(T *&requested_data, std::vector<idx_t> &requested_cols, idx_t dst_rank,
                                        embeddings::Embedding *item_embeddings) {
            idx_t requested_count = requested_cols.size();
            idx_t recv_count;
            if (rank < dst_rank) {
                MPI_Send((void *) &requested_count, 1, MPI_UINT64_T, dst_rank, 0, comm);
                MPI_Send((void *) requested_cols.data(), requested_count, MPI_UINT64_T, dst_rank, 0, comm);
                MPI_Recv((void *) &recv_count, 1, MPI_UINT64_T, dst_rank, 0, comm, MPI_STATUS_IGNORE);
                std::vector<idx_t> recv_cols(recv_count);
                MPI_Recv((void *) recv_cols.data(), recv_count, MPI_UINT64_T, dst_rank, 0, comm,
                         MPI_STATUS_IGNORE);

                // prepare send data
                auto *send_data = new val_t[recv_count * emb_dim];
                for (idx_t i = 0; i < recv_count; i++) {
                    item_embeddings->weights->read_row(recv_cols[i] - item_embeddings->start_idx,
                                                       send_data + i * emb_dim);
                }

                requested_data = new val_t[requested_count * emb_dim];

                MPI_Send((void *) send_data, recv_count * emb_dim, MPI_FLOAT, dst_rank, 0, comm);
                MPI_Recv((void *) requested_data, requested_count * emb_dim, MPI_FLOAT, dst_rank, 0, comm,
                         MPI_STATUS_IGNORE);

                delete[] send_data;
            } else {
                MPI_Recv((void *) &recv_count, 1, MPI_UINT64_T, dst_rank, 0, comm, MPI_STATUS_IGNORE);
                std::vector<idx_t> recv_cols(recv_count);
                MPI_Recv((void *) recv_cols.data(), recv_count, MPI_UINT64_T, dst_rank, 0, comm,
                         MPI_STATUS_IGNORE);

                MPI_Send((void *) &requested_count, 1, MPI_UINT64_T, dst_rank, 0, comm);
                MPI_Send((void *) requested_cols.data(), requested_count, MPI_UINT64_T, dst_rank, 0, comm);

                auto *send_data = new val_t[recv_count * emb_dim];
                for (unsigned long i = 0; i < recv_count; i++) {
                    item_embeddings->weights->read_row(recv_cols[i] - item_embeddings->start_idx,
                                                       send_data + i * emb_dim);
                }
                requested_data = new val_t[requested_count * emb_dim];

                MPI_Recv((void *) requested_data, requested_count * emb_dim, MPI_FLOAT, dst_rank, 0, comm,
                         MPI_STATUS_IGNORE);
                MPI_Send((void *) send_data, recv_count * emb_dim, MPI_FLOAT, dst_rank, 0, comm);
                delete[] send_data;
            }
        }


        // request data from other ranks
        template<class T>
        void Data_shuffle::shuffle_grad(T *send_data, std::vector<idx_t> &cols, idx_t dst_rank, T *&recv_data,
                                        std::vector<idx_t> &recv_cols) {
            idx_t send_count = cols.size();
            idx_t recv_count;
            //printf("rank %d, send_count is %lu\n", rank, send_count);
            if (rank < dst_rank) {
                MPI_Send((void *) &send_count, 1, MPI_UINT64_T, dst_rank, 0, comm);
                MPI_Send((void *) cols.data(), send_count, MPI_UINT64_T, dst_rank, 0, comm);
                MPI_Send((void *) send_data, send_count * emb_dim, MPI_FLOAT, dst_rank, 0, comm);
                //printf("finish sending from rank %d\n", rank);
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
                //printf("finish receiving from rank %d\n", rank);
                MPI_Send((void *) &send_count, 1, MPI_UINT64_T, dst_rank, 0, comm);
                MPI_Send((void *) cols.data(), send_count, MPI_UINT64_T, dst_rank, 0, comm);
                MPI_Send((void *) send_data, send_count * emb_dim, MPI_FLOAT, dst_rank, 0, comm);
            }
        }


    } // cf
} // modules