#include "engine.hpp"

namespace cf {
    namespace modules {
        namespace train {
            val_t Engine::train_one_epoch_new() {
                int rank;
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                int world_size;
                MPI_Comm_size(MPI_COMM_WORLD, &world_size);


                //return this->origin_train_one_epoch();

                const idx_t num_sub_epochs = this->cf_config->num_subepochs;


                auto start = std::chrono::steady_clock::now();

                const idx_t num_iterations = this->train_data->data_rows;
                const idx_t total_rows = this->cf_config->num_users;
                const idx_t total_cols = this->cf_config->num_items;
                std::cout << "total iterations: " << num_iterations << std::endl;
                std::cout << "total cols: " << total_cols << std::endl;

                std::cout << "finished initializing" << std::endl;
                // the shared memory is used to partition the columns of the embedding matrix
                // this is not good, we need to store this sequence in all processes
                idx_t *shared_data;
                idx_t *negative_partition;

                shared_data = new idx_t[total_cols];
                negative_partition = new idx_t[total_cols];
                if (rank == 0) {
                    for (idx_t i = 0; i < total_cols; i++) {
                        shared_data[i] = i;
                        negative_partition[i] = i;
                    }
                }
#if __cplusplus >= 201103L
                std::random_shuffle(shared_data, shared_data + total_cols);
#else
                std::random_shuffle(shared_data, shared_data + total_cols, std::rand);
#endif
                MPI_Bcast(shared_data, total_cols, MPI_UINT64_T, 0, MPI_COMM_WORLD);
                //std::cout << shared_data[0] << " " << shared_data[1] << " " << shared_data[2] << " "  << shared_data[3] << std::endl;
                // After this operation, each process should get a sequence of column indices,
                // which is used for partitioning the columns of the embedding matrix
                std::map<idx_t, idx_t> col_map;
                idx_t par_num = num_sub_epochs * world_size;
                //std::cout << "par_num: " << par_num << std::endl;
                for (idx_t i = 0; i < par_num; i++) {
                    idx_t q = total_cols / par_num;
                    idx_t r = total_cols % par_num;
                    idx_t par_start = i * q + (i < r);
                    idx_t par_end = (i + 1) * q + ((i + 1) < r);
                    //std::cout << "par_start: " << par_start << " par_end: " << par_end << std::endl;
                    for (idx_t j = par_start; j < par_end; j++) {
                        col_map[shared_data[j]] = i;
                        //std::cout << "col_map[" << shared_data[j] << "] = " << i << std::endl;
                    }
                }

                //std::cout << col_map[shared_data[0]] << " " << col_map[shared_data[1]] << " " << col_map[shared_data[2]] << " "  << col_map[shared_data[3]] << std::endl;
                // map the data into different sub-sub epochs
                std::map<idx_t, std::vector<std::pair<idx_t, idx_t> > > m;
                for (idx_t i = 0; i < num_iterations; i++) {
                    idx_t tmp = col_map[i];
                    idx_t user_id, item_id;
                    this->train_data->read_user_item(i, user_id, item_id);
                    m[tmp].push_back(std::make_pair(user_id, item_id));
                }

                idx_t num_negs = this->cf_config->num_negs;
                double local_loss = 0.;
                double global_loss = 0.;
                idx_t max_threads = omp_get_max_threads();
                datasets::Dataset *train_data_ptr = this->train_data.get();
                // this->positive_sampler->Data_shuffle();

                // each processor has a local aggregator_weights
                behavior_aggregators::AggregatorWeights *aggregator_weights_ptr = this->aggregator_weights.get();

                val_t *local_weights0 = aggregator_weights_ptr->weights0.data();
                val_t *global_weights0;
                idx_t dim = this->cf_config->emb_dim;
                global_weights0 = new val_t[dim * dim];

                MPI_Allreduce((void *) aggregator_weights_ptr->weights0.data(), (void *) global_weights0, dim * dim,
                              MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
                for (idx_t i = 0; i < dim * dim; ++i) {
                    global_weights0[i] /= world_size;
                }


                //std::copy(local_weights0, local_weights0 + dim * dim, global_weights0);
                if (this->cf_config->milestones.size() > 1) {
                    this->cf_modules->optimizer->scheduler_multi_step_lr(this->epoch, this->cf_config->milestones, 0.1);
                } else {
                    this->cf_modules->optimizer->scheduler_step_lr(this->epoch, this->cf_config->milestones[0], 0.1);
                }

                Eigen::initParallel();
                // each processor do the local training
                // the first level
                // select one partition from the shared_data
                //std::cout << "before sub epoch: " << totalMemoryUsed << " from process " << rank << std::endl;
                for (int sub_epoch = 0; sub_epoch < num_sub_epochs; sub_epoch++) {
                    //std::cout << "sub_epoch: " << sub_epoch << std::endl;

                    idx_t num_col_sub_epoch = total_cols / num_sub_epochs;
                    idx_t r = total_cols % num_sub_epochs;
                    //std::cout << "num_col_sub_epoch: " << num_col_sub_epoch << std::endl;
                    //the part is the partition of the first level, the partition that used to sample the positive items
                    idx_t part_start = sub_epoch * num_col_sub_epoch + (sub_epoch < r ? sub_epoch : r);
                    idx_t part_end = (sub_epoch + 1) * num_col_sub_epoch + (sub_epoch < r ? sub_epoch : r);

                    //std::cout << "part_start: " << part_start << " part_end: " << part_end << std::endl;

                    // the second level
                    // in this level, each process sample in its own partition (level 2) and update the weights
                    for (idx_t j = 0; j < world_size; j++) {

                        //sync aggregator_weights
                        /*Eigen::Map<Eigen::Array<val_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> global_weight_mat(
                                global_weights0, dim, dim);
                        aggregator_weights_ptr->weights0 = global_weight_mat;*/
                        //std::cout << "before sub-sub epoch: " << totalMemoryUsed << " from process " << rank << std::endl;
                        // Currently version without support of OpenMP
                        idx_t user_id = 0;
                        idx_t pos_id = 0;
                        std::vector<idx_t> neg_ids(num_negs);
                        idx_t seed = (this->epoch + 1) * j + this->random_gen->read();

                        negative_samplers::NegativeSampler *negative_sampler = nullptr;
                        if (this->cf_config->neg_sampler == 1) {
                            negative_sampler = static_cast<negative_samplers::NegativeSampler *>(
                                    new negative_samplers::RandomTileNegativeSampler(this->cf_config, seed));
                        } else {
                            negative_sampler = static_cast<negative_samplers::NegativeSampler *>(
                                    new negative_samplers::UniformRandomNegativeSampler(this->cf_config, seed));
                        }

                        memory::ThreadBuffer *thread_buffer = new memory::ThreadBuffer(this->cf_config->emb_dim,
                                                                                       num_negs,
                                                                                       this->cf_config->refresh_interval,
                                                                                       this->cf_config->tile_size);

                        behavior_aggregators::BehaviorAggregator behavior_aggregator(train_data_ptr,
                                                                                     aggregator_weights_ptr,
                                                                                     cf_config);


                        auto iters = m[sub_epoch * world_size + j]; //positive items in current partition


                        //Data_shuffle all the columns again for negative sampling
                        if (rank == 0) {
#if __cplusplus >= 201103L
                            std::random_shuffle(negative_partition, negative_partition + total_cols);
#else
                            std::random_shuffle(negative_partition, negative_partition + total_cols, std::rand);
#endif
                        }
                        MPI_Bcast(negative_partition, total_cols, MPI_UINT64_T, 0, MPI_COMM_WORLD);
                        idx_t part_size = total_cols / world_size;
                        random::Uniform *uniform = new random::Uniform(part_size, seed);
                        idx_t cur_parition = sub_epoch * world_size + j;


                        // determine the range of negative items
                        std::cout << part_size << std::endl;

                        int count = 0;
                        for (auto &iter: iters) {
                            //std::cout << count << std::endl;
                            user_id = iter.first;
                            pos_id = iter.second;
                            count++;
                            // sample negative items to neg_ids
                            for (idx_t k = 0; k < this->cf_config->num_negs; k++) {

                                // id is the position in negative_partition
                                idx_t id = uniform->read() + rank * part_size;
                                //if the negative item and positive item are in the same partition, we need to resample
                                while (col_map[negative_partition[id]] / num_sub_epochs ==
                                       cur_parition / num_sub_epochs) {
                                    id = uniform->read() + rank * part_size;
                                    //std::cout << "resample" << id <<std::endl;
                                    //std::cout << col_map[negative_partition[id]]  << " " << cur_parition << std::endl;
                                }
                                neg_ids[k] = id;
                            }
                            std::unordered_map<idx_t, std::vector<val_t> > emb_grads;
                            //auto tmp = this->model->forward_backward(user_id, pos_id, neg_ids, this->cf_modules,
                            //                                         thread_buffer, &behavior_aggregator, emb_grads);
                            //std::cout << "loss at process "<< rank << " is: " << tmp << std::endl;
                            //local_loss += tmp;
                        }

                        //std::cout << "start synchronize positive item embedding weights" << std::endl;
                        /*for (int k = 0; k < total_cols; k++) {
                            printf("col_map[%lu]=%lu from process %d\n", shared_data[k], col_map[shared_data[k]], rank);
                        }*/
                        // synchronize item_embedding_weights for positive items


                        /*for (idx_t k = part_start; k < part_end; k++) {
                            idx_t col_id = shared_data[k];  // col id
                            idx_t col_group = col_map[col_id];  // col group
                            int src = col_group % world_size + j;  // src processor
                            auto * tmp_weights = new val_t[dim];
                            memory::Array<val_t>* item_embedding_weights = this->model->item_embedding->weights;
                            item_embedding_weights->read_row(col_id, tmp_weights);
                            MPI_Bcast((void*) tmp_weights,dim,MPI_FLOAT,src,MPI_COMM_WORLD);
                            item_embedding_weights->write_row(col_id, tmp_weights);
                            delete[] tmp_weights;
                        }


                        //std::cout << "start synchronize negative item embedding weights" << std::endl;
                        // synchronize item_embedding_weights for negative items
                        for (idx_t k = 0; k < part_size * world_size; k++) {
                            idx_t col_id = negative_partition[k];  // col id
                            idx_t src = k / world_size;
                            auto * tmp_weights = new val_t[dim];
                            memory::Array<val_t>* item_embedding_weights = this->model->item_embedding->weights;
                            item_embedding_weights->read_row(col_id, tmp_weights);
                            MPI_Bcast((void*) tmp_weights,dim,MPI_FLOAT,src,MPI_COMM_WORLD);
                            item_embedding_weights->write_row(col_id, tmp_weights);
                            delete[] tmp_weights;
                        }

                        // get the average of weights in different processors
                        MPI_Allreduce((void *) aggregator_weights_ptr->weights0.data(), (void *) global_weights0, dim * dim,
                                      MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
                        for (idx_t i = 0; i < dim * dim; ++i) {
                            global_weights0[i] /= world_size;
                        }*/


                        //std::cout << "finished sub-sub-epoch:" << j << std::endl;
                        delete uniform;
                        delete thread_buffer;
                        delete negative_sampler;
                        //std::cout << local_loss << std::endl;
                        //std::cout << "memory used: " << totalMemoryUsed << std::endl;

                    }
                    //std::cout << "end sub-sub epoch: " << totalMemoryUsed << " from process " << rank << std::endl;

// #pragma omp parallel reduction(+ : local_loss) shared(train_data_ptr, aggregator_weights_ptr)
/* #pragma omp parallel reduction(+ : local_loss)
                    {
                        idx_t user_id = 0;
                        idx_t pos_id = 0;
                        std::vector<idx_t> neg_ids(num_negs);
                        idx_t num_threads = omp_get_num_threads();
                        idx_t thread_id = omp_get_thread_num();
                        // idx_t seed = (this->epoch + 1) * thread_id + this->random_gen->read();
                        idx_t seed = (this->epoch + 1) * thread_id;

                        negative_samplers::NegativeSampler *negative_sampler = nullptr;
                        if (this->cf_config->neg_sampler == 1) {
                            negative_sampler = static_cast<negative_samplers::NegativeSampler *>(
                                    new negative_samplers::RandomTileNegativeSampler(this->cf_config, seed));
                        } else {
                            negative_sampler = static_cast<negative_samplers::NegativeSampler *>(
                                    new negative_samplers::UniformRandomNegativeSampler(this->cf_config, seed));
                        }

                        memory::ThreadBuffer *t_buf = new memory::ThreadBuffer(this->cf_config->emb_dim, num_negs);

                        // behavior_aggregators::BehaviorAggregator* behavior_aggregator = nullptr;
                        behavior_aggregators::BehaviorAggregator behavior_aggregator(train_data_ptr,
                                                                                     aggregator_weights_ptr,
                                                                                     cf_config);

#pragma omp master
                        {
                            std::cout << "num_col_sub_epoch " << num_col_sub_epoch << " max_threads " << num_threads << std::endl;
                        }

// #pragma omp for schedule(dynamic, 4)
// for (idx_t i = 0; i < 16; ++i)
#pragma omp for schedule(dynamic, 512)
                        for (idx_t i = 0; i < num_col_sub_epoch; ++i) {
                            double start_time = omp_get_wtime();
                            idx_t train_data_idx = this->positive_sampler->read(i);
                            this->train_data->read_user_item(train_data_idx, user_id, pos_id);
                            negative_sampler->ignore_pos_sampling(user_id, pos_id, neg_ids);
                            // negative_sampler->sampling(neg_ids);
                            double end_time = omp_get_wtime();
                            t_buf->time_map["data"] = t_buf->time_map["data"] + (end_time - start_time);

                            local_loss += this->model->forward_backward(user_id, pos_id, neg_ids, this->cf_modules,
                                                                        t_buf,
                                                                        &behavior_aggregator);
                        }
                        // performance_breakdown(t_buf);
                    }*/

                    // this->cf_modules->optimizer->dense_step(this->model->user_embedding);
                    this->model->user_embedding->zero_grad();
                    // this->cf_modules->optimizer->dense_step(this->model->item_embedding);
                    this->model->item_embedding->zero_grad();


                    idx_t emb_dim = this->cf_config->emb_dim;
                    auto *global_item_embedding = new val_t[emb_dim];
                    auto *item_embedding = new val_t[emb_dim];
                    /*for (idx_t i = 0; i < this->train_data->data_cols; ++i) {

                        this->model->item_embedding->read_weights(i,item_embedding);
                        MPI_Allreduce((void *) item_embedding, (void *) global_item_embedding, emb_dim,
                                      MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
                        for (idx_t j = 0; j < emb_dim; ++j) {
                            global_item_embedding[j] /= world_size;
                        }
                        this->model->item_embedding->write_weights(i,global_item_embedding);
                    }*/
                    delete[] global_item_embedding;
                    delete[] item_embedding;
                    //std::cout << "end sub epoch: " << totalMemoryUsed << " from process " << rank << std::endl;
                }

                // get the average of weights in different processors
                /*MPI_Allreduce((void *) aggregator_weights_ptr->weights0.data(), (void *) global_weights0, dim * dim,
                              MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
                for (idx_t i = 0; i < dim * dim; ++i) {
                    global_weights0[i] /= world_size;
                }
                Eigen::Map<Eigen::Array<val_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> global_weight_mat(
                        global_weights0, dim, dim);
                aggregator_weights_ptr->weights0 = global_weight_mat;*/

                this->epoch += 1;
                idx_t total_iterations;
                MPI_Allreduce((void *) &local_loss, (void *) &global_loss, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce((void *) &num_iterations, (void *) &total_iterations, 1, MPI_UINT64_T, MPI_SUM,
                              MPI_COMM_WORLD);
                global_loss /= total_iterations;
                std::cout << "Total iterations: " << total_iterations << std::endl;
                auto end = std::chrono::steady_clock::now();
                double epoch_time =
                        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1.e-9;
                std::cout << "epoch time: " << epoch_time << " s " << std::endl;
                std::cout << "epoch " << this->epoch << " loss " << global_loss << std::endl;
                delete[] global_weights0;
                delete[] shared_data;
                delete[] negative_partition;
                return global_loss;
            }
            }
    }
}
