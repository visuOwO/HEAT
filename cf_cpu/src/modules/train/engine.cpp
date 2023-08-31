#include "engine.hpp"
#include "utils/Data_shuffle.h"
#include <mpi.h>
#include <set>

/*static size_t totalMemoryUsed = 0;

void * operator new(size_t size) {
    totalMemoryUsed += size;
    return malloc(size);
}

void operator delete(void* memory, size_t size)
{
    totalMemoryUsed -= size;
    free(memory);
}*/

namespace cf {
    namespace modules {
        idx_t Data_shuffle::total_cols;
        idx_t Data_shuffle::emb_dim;
        std::shared_ptr<CFModules> Data_shuffle::cf_modules;
        MPI_Comm Data_shuffle::comm;
        int Data_shuffle::rank;
        int Data_shuffle::world_size;
        namespace train {
            Engine::Engine(std::shared_ptr<datasets::Dataset> train_data,
                           std::shared_ptr<behavior_aggregators::AggregatorWeights> aggregator_weights,
                           std::shared_ptr<models::Model> model,
                           std::shared_ptr<CFConfig> cf_config)
                    : train_data(train_data), aggregator_weights(aggregator_weights), model(model),
                      cf_config(cf_config) {
                this->positive_sampler = std::make_shared<random::Shuffle>(cf_config->train_size);
                optimizers::Optimizer *optimizer = new optimizers::SGD(cf_config);
                this->cf_modules = std::make_shared<CFModules>(cf_config.get(), optimizer);
                this->random_gen = new random::Uniform(1024, 66);
                this->epoch = 0;
            }

            void Engine::performance_breakdown(memory::ThreadBuffer *t_buf) {
                int thread_id = omp_get_thread_num();
                if (thread_id == 0) {
                    int max_num_threads = omp_get_max_threads();
                    std::cout << "thread idx: " << thread_id << " max threads: " << max_num_threads << std::endl;

                    double epoch_time = t_buf->time_map["data"] + t_buf->time_map["f_b"];
                    double data_p = t_buf->time_map["data"] / epoch_time * 100.0;
                    double forward_p = t_buf->time_map["forward"] / epoch_time * 100.0;
                    double backward_p = t_buf->time_map["backward"] / epoch_time * 100.0;

                    double read_emb_p = t_buf->time_map["read_emb"] / t_buf->time_map["forward"] * 100.0;
                    double aggr_f_p = t_buf->time_map["aggr_f"] / t_buf->time_map["forward"] * 100.0;
                    double read_his_p = t_buf->time_map["read_his"] / t_buf->time_map["aggr_f"] * 100.0;
                    double his_mm_p = t_buf->time_map["his_mm"] / t_buf->time_map["aggr_f"] * 100.0;
                    double dot_p = t_buf->time_map["dot"] / t_buf->time_map["forward"] * 100.0;
                    double norm_p = t_buf->time_map["norm"] / t_buf->time_map["forward"] * 100.0;
                    double loss_p = t_buf->time_map["loss"] / t_buf->time_map["forward"] * 100.0;

                    double grad_p = t_buf->time_map["grad"] / t_buf->time_map["backward"] * 100.0;
                    double reg_p = t_buf->time_map["reg"] / t_buf->time_map["backward"] * 100.0;
                    double write_emb_p = t_buf->time_map["write_emb"] / t_buf->time_map["backward"] * 100.0;
                    double aggr_b_p = t_buf->time_map["aggr_b"] / t_buf->time_map["backward"] * 100.0;

                    std::cout << " epoch_time: " << epoch_time
                              << " data: " << t_buf->time_map["data"] << " data_p %: " << data_p
                              << " forward: " << t_buf->time_map["forward"] << " forward_p %: " << forward_p
                              << " backward: " << t_buf->time_map["backward"] << " backward_p %: " << backward_p
                              << std::endl
                              << " read_emb: " << t_buf->time_map["read_emb"] << " read_emb_p %: " << read_emb_p
                              << " aggr_f: " << t_buf->time_map["aggr_f"] << " aggr_f_p %: " << aggr_f_p
                              << " read_his: " << t_buf->time_map["read_his"] << " read_his_p %: " << read_his_p
                              << " his_mm: " << t_buf->time_map["his_mm"] << " his_mm_p %: " << his_mm_p
                              << " dot: " << t_buf->time_map["dot"] << " dot_p %: " << dot_p
                              << " norm: " << t_buf->time_map["norm"] << " norm_p %: " << norm_p
                              << " loss: " << t_buf->time_map["loss"] << " loss_p %: " << loss_p << std::endl
                              << " grad: " << t_buf->time_map["grad"] << " grad_p %: " << grad_p
                              << " reg: " << t_buf->time_map["reg"] << " reg_p %: " << reg_p
                              << " write_emb: " << t_buf->time_map["write_emb"] << " write_emb_p %: " << write_emb_p
                              << " aggr_b: " << t_buf->time_map["aggr_b"] << " aggr_b_p %: " << aggr_b_p
                              << std::endl;
                    std::cout << std::endl;
                }
            }

            val_t Engine::train_one_epoch() {
                int rank, world_size;
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                MPI_Comm_size(MPI_COMM_WORLD, &world_size);

                //initialize shared memory for MPI
                MPI_Comm shared_comm;
                MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shared_comm);

                MPI_Win win;
                val_t *shared_aggregated_weights;
                MPI_Win_allocate_shared(sizeof(val_t) * this->cf_config->emb_dim * this->cf_config->emb_dim,
                                        sizeof(val_t),
                                        MPI_INFO_NULL, shared_comm, &shared_aggregated_weights, &win);

                if (rank == 0) {
                    for (int i = 0; i < this->cf_config->emb_dim * this->cf_config->emb_dim; ++i) {
                        shared_aggregated_weights[i] = this->aggregator_weights->weights0.data()[i];
                    }
                }

                MPI_Win_fence(0, win);

                // copy the shared memory to local memory
                val_t *local_aggregator_weights;
                local_aggregator_weights = new val_t[this->cf_config->emb_dim * this->cf_config->emb_dim];
                memcpy(local_aggregator_weights, shared_aggregated_weights,
                       sizeof(val_t) * this->cf_config->emb_dim * this->cf_config->emb_dim);


                MPI_Win_fence(0, win);


                Data_shuffle::total_cols = this->cf_config->num_items;
                Data_shuffle::comm = MPI_COMM_WORLD;
                Data_shuffle::world_size = world_size;
                Data_shuffle::rank = rank;
                Data_shuffle::emb_dim = this->cf_config->emb_dim;
                Data_shuffle::cf_modules = this->cf_modules;

                idx_t process_status = 1;   // 1 means the process is not finished
                idx_t system_status;

                const idx_t iterations = this->train_data->data_rows;
                double local_loss = 0.;
                idx_t num_negs = this->cf_config->num_negs;

                datasets::Dataset *train_data_ptr = this->train_data.get();
                // this->positive_sampler->Data_shuffle();
                behavior_aggregators::AggregatorWeights *aggregator_weights_ptr = this->aggregator_weights.get();
                if (this->cf_config->milestones.size() > 1) {
                    this->cf_modules->optimizer->scheduler_multi_step_lr(this->epoch, this->cf_config->milestones, 0.1);
                } else {
                    this->cf_modules->optimizer->scheduler_step_lr(this->epoch, this->cf_config->milestones[0], 0.1);
                }

                Eigen::initParallel();

                idx_t user_id = 0;
                idx_t item_id = 0;
                std::vector<idx_t> neg_ids(num_negs);
                idx_t seed = (this->epoch - 1) * this->train_data->data_rows;

                // initialize nagative sampler
                negative_samplers::NegativeSampler *negative_sampler = nullptr;
                if (this->cf_config->neg_sampler == 1) {
                    negative_sampler = static_cast<negative_samplers::NegativeSampler *>(
                            new negative_samplers::RandomTileNegativeSampler(this->cf_config, seed));
                } else {
                    negative_sampler = static_cast<negative_samplers::NegativeSampler *>(
                            new negative_samplers::UniformRandomNegativeSampler(this->cf_config, seed));
                }

                memory::ThreadBuffer *t_buf = new memory::ThreadBuffer(this->cf_config->emb_dim, num_negs,
                                                                       this->cf_config->refresh_interval,
                                                                       this->cf_config->tile_size);

                // behavior_aggregators::BehaviorAggregator* behavior_aggregator = nullptr;
                behavior_aggregators::BehaviorAggregator behavior_aggregator(train_data_ptr, aggregator_weights_ptr,
                                                                             cf_config);

                // std::cout << "start training" << std::endl;
                std::unordered_map<idx_t, std::vector<val_t> > emb_grads;

                for (idx_t i = 0; i < iterations; i++) {


                    if (i % this->cf_config->refresh_interval == 0) {
                        // synchronize the process status
                        MPI_Allgather(&process_status, 1, MPI_UINT64_T, &system_status, 1, MPI_UINT64_T,
                                      MPI_COMM_WORLD);

                        if (i % this->cf_config->refresh_interval == 0) {
                            // update the global item embeddings from local gradients
                            Data_shuffle::shuffle_and_update_item_grads(emb_grads, model->item_embedding);
                        }

                        emb_grads.clear();
                        // Fetch next positive embedding for the next batch
                        std::set<idx_t> users;
                        std::vector<idx_t> pos_ids(this->cf_config->refresh_interval);
                        for (int j = 0; j < this->cf_config->refresh_interval; j++) {
                            if (i + j >= iterations) {
                                break;
                            }
                            idx_t id = this->positive_sampler->read(i + j);
                            idx_t uid, iid;
                            this->train_data->read_user_item(id, uid, iid);
                            t_buf->pos_item_ids[j] = iid;
                            if (users.find(uid) == users.end()) {
                                users.insert(uid);
                            }
                        }

                        // shuffle the positive embeddings to t_buf->tiled_pos_emb_buf
                        //printf("start shuffling pos embeddings\n");
                        Data_shuffle::shuffle_embs(std::vector<idx_t>(t_buf->pos_item_ids, t_buf->pos_item_ids +
                                                                                           this->cf_config->refresh_interval),
                                                   t_buf->pos_emb_buf,
                                                   this->model->item_embedding);


                        // shuffle historical embeddings for next batch
                        std::set<idx_t> his_items;
                        memory::Array<idx_t>* historical_items = this->train_data->historical_items;
                        auto his_id_buf = new idx_t[train_data_ptr->max_his];
                        for (auto &user: users) {
                            idx_t* his_ids = historical_items->read_row(user_id, his_id_buf);
                            for (int j = 0; j < train_data_ptr->max_his; j++) {
                                if (his_items.find(his_ids[j]) == his_items.end()) {
                                    his_items.insert(his_ids[j]);
                                }
                            }
                        }

                        behavior_aggregator.tiled_his_buf.resize(his_items.size() * this->cf_config->emb_dim);
                        behavior_aggregator.his_ids.resize(his_items.size());
                        std::copy(his_items.begin(), his_items.end(), behavior_aggregator.his_ids.data());
                        //printf("start shuffling historical embeddings\n");
                        Data_shuffle::shuffle_embs(behavior_aggregator.his_ids, behavior_aggregator.tiled_his_buf.data(), this->model->item_embedding);
                        behavior_aggregator.his_id_map.clear();
                        for (int j = 0; j < behavior_aggregator.his_ids.size(); j++) {
                            behavior_aggregator.his_id_map[behavior_aggregator.his_ids[j]] = j;
                        }

                        // assume that the negative sampler is random tile negative sampler

                        //  just a test for reading an embedding
                        //printf("test for reading an embedding\n");
                        std::vector<val_t> emb(this->cf_config->emb_dim);
                        /*this->model->item_embedding->read_weights(5+this->model->item_embedding->start_idx, emb.data());
                        for (int j = 0; j < this->cf_config->emb_dim; j++) {
                            printf("%f ", emb[j]);
                        }
                        printf("\n");*/

                        if (dynamic_cast<const negative_samplers::RandomTileNegativeSampler *>(negative_sampler) !=
                            nullptr) {
                            auto neg_tile = dynamic_cast<const negative_samplers::RandomTileNegativeSampler *>(negative_sampler)->neg_tile;
                            //printf("start shuffling and updating neg grads\n");
                            // shuffle the negative embeddings to t_buf->tiled_neg_emb_buf
                            Data_shuffle::shuffle_embs(neg_tile,t_buf->tiled_neg_emb_buf,
                                                       this->model->item_embedding);

                            /*printf("start inspecting negative embeddings\n");

                            for (auto j = 0; j < this->cf_config->tile_size; j++) {
                                printf("%lu ", neg_tile[j]);
                            }
                            printf("\n");
                            // inspect the negative embeddings
                            for (auto j = 0; j < this->cf_config->tile_size; j++) {
                                printf("negative embedding %d %d, %f %f %f %f\n", j, rank, t_buf->tiled_neg_emb_buf[j * this->cf_config->emb_dim],
                                       t_buf->tiled_neg_emb_buf[j * this->cf_config->emb_dim + 1],
                                       t_buf->tiled_neg_emb_buf[j * this->cf_config->emb_dim + 2],
                                       t_buf->tiled_neg_emb_buf[j * this->cf_config->emb_dim + 3]);
                            }

                            printf("finish inspecting negative embeddings\n");*/

                        } else {
                            throw std::runtime_error("Only support random tile negative sampler");
                        }
                    }

                    // try to examine all user embeddings
                    /*if (i % this->cf_config->refresh_interval == 0) {
                        for (int j = 0; j < this->cf_config->num_users; j++) {
                            std::vector<val_t> emb(this->cf_config->emb_dim);
                            this->model->user_embedding->weights->read_row(j-this->model->user_embedding->start_idx, emb.data());
                            for (int k = 0; k < this->cf_config->emb_dim; k++) {
                                if (emb[k] == nanf("")) {
                                    printf("nan at %d %d\n", j, k);
                                    continue;
                                }
                                printf("%f ", emb[k]);
                            }
                            printf("\n");
                        }
                        break;
                    }*/

                    // start training
                    //printf("start forward and backward\n")  ;
                    //std::cout << "iteration: " << i << std::endl;
                    idx_t train_data_idx = this->positive_sampler->read(i);
                    //std::cout << "train_data_idx: " << train_data_idx << std::endl;
                    this->train_data->read_user_item(train_data_idx, user_id, item_id);
                    //printf("user_id: %lu item_id: %lu rank: %d\n", user_id, item_id, rank);
                    negative_sampler->sampling(neg_ids);
                    auto neg_tile = dynamic_cast<const negative_samplers::RandomTileNegativeSampler *>(negative_sampler)->neg_tile;
                    auto tmp = this->model->forward_backward(user_id, i % this->cf_config->refresh_interval, neg_ids, neg_tile,
                                                          this->cf_modules, t_buf, &behavior_aggregator, emb_grads);

                    //printf("finish forward and backward\n")  ;
                    local_loss = local_loss + tmp;

                    /*printf("start examine aggregated weights\n");
                    for (int j = 0; j < this->cf_config->emb_dim; j++) {
                        printf("%f ", shared_aggregated_weights[j]);
                    }
                    printf("\n");
                    printf("finish examine aggregated weights\n");*/

                    /*printf("update the global aggregator weights from local gradients\n");
                    if (i % this->cf_config->train_size == 0) {
                        // update the global aggregator weights from local gradients
                        val_t *data = behavior_aggregator.weights0_grad_accu.data();
                        Eigen::Map<Eigen::Array<val_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> global_weight_mat(
                                shared_aggregated_weights, this->cf_config->emb_dim, this->cf_config->emb_dim);
                        global_weight_mat -= behavior_aggregator.l_r * behavior_aggregator.weights0_grad_accu;
                        behavior_aggregator.weights0_grad_accu.setZero();   // reset the gradient accumulator
                    }*/

                    //printf("local_loss is %f\n", tmp);

                }
                process_status = 0;   // 0 means the process is finished
                //local_loss = local_loss / iterations;
                // keep sharing data until all the processes are finished
                //printf("finish training epoch from rank %d\n", rank);
                while (true) {
                    MPI_Allgather(&process_status, 1, MPI_UINT64_T, &system_status, 1, MPI_UINT64_T,
                                  MPI_COMM_WORLD);
                    if (system_status == 0) {
                        break;
                    } else {
                        // Keep sending and receiving data until all the processes are finished
                        Data_shuffle::shuffle_and_update_item_grads(emb_grads,
                                                                    this->model->item_embedding);
                        Data_shuffle::shuffle_embs(std::vector<idx_t>(),
                                                   t_buf->pos_emb_buf,
                                                   this->model->item_embedding);
                        Data_shuffle::shuffle_embs(std::vector<idx_t>(),
                                                   t_buf->tiled_neg_emb_buf,
                                                   this->model->item_embedding);
                        Data_shuffle::shuffle_embs(std::vector<idx_t>(),
                                                   t_buf->tiled_neg_emb_buf,
                                                   this->model->item_embedding);
                        emb_grads.clear();
                    }
                }
                delete[] local_aggregator_weights;
                double global_loss;
                idx_t total_iteration;
                MPI_Allreduce(&local_loss, &global_loss, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(&iterations, &total_iteration, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
                global_loss = global_loss / total_iteration;
                return local_loss;
            }

            void Engine::evaluate0() {
                idx_t emb_dim = this->cf_config->emb_dim;
                idx_t num_users = this->cf_config->num_users;
                idx_t num_items = this->cf_config->num_items;
                memory::Array<val_t> *user_embedding_weights = this->model->user_embedding->weights;
                memory::Array<val_t> *item_embedding_weights = this->model->item_embedding->weights;
                Eigen::Map<Eigen::Matrix<val_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> user_emb_mat(
                        user_embedding_weights->data, num_users, emb_dim);
                Eigen::Map<Eigen::Matrix<val_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> item_emb_mat(
                        item_embedding_weights->data, num_items, emb_dim);
                this->sim_matrix = user_emb_mat * item_emb_mat.transpose();
                // this->sim_matrix = Eigen::Matrix<val_t, Eigen::Dynamic, Eigen::Dynamic>::Random(100, 100);
            }

// PyMatrix Engine::evaluate1()
// {
//     const idx_t iterations = this->test_data->data_rows;
//     idx_t emb_dim = this->cf_config->emb_dim;
//     idx_t num_negs = this->cf_config->num_negs;
//     idx_t num_users = this->cf_config->num_users;
//     idx_t num_items = this->cf_config->num_items;

//     datasets::Dataset* test_data_ptr = this->test_data.get();
//     behavior_aggregators::AggregatorWeights* aggregator_weights_ptr = this->aggregator_weights.get();

//     memory::Array<val_t>* item_embedding_weights = this->model->item_embedding->weights;

//     memory::Array<val_t>* aggr_user_embedding = new ValArray(this->cf_config->num_users, this->cf_config->emb_dim, nullptr);

// // #pragma omp parallel reduction(+ : loss) shared(test_data_ptr, aggregator_weights_ptr)
// #pragma omp parallel reduction(+ : loss)
//     {
//         idx_t user_id = 0;
//         idx_t pos_id = 0;

//         idx_t thread_id = omp_get_thread_num();
//         // idx_t seed = (this->epoch + 1) * thread_id + this->random_gen->read();
//         idx_t seed = (this->epoch + 1) * thread_id;

//         memory::ThreadBuffer* t_buf = new memory::ThreadBuffer(emb_dim, num_negs);

//         // behavior_aggregators::BehaviorAggregator* behavior_aggregator = nullptr;
//         behavior_aggregators::BehaviorAggregator behavior_aggregator(test_data_ptr, aggregator_weights_ptr, cf_config);

// // #pragma omp for schedule(dynamic, 4)
// // for (idx_t i = 0; i < 16; ++i)
// #pragma omp for schedule(dynamic, 512)
//         for (idx_t i = 0; i < iterations; ++i)
//         {
//             double start_time = omp_get_wtime();
//             idx_t test_data_idx = this->positive_sampler->read(i);
//             this->test_data->read_user_item(test_data_idx, user_id, pos_id);
//             memory::Array<val_t>* user_embedding_weights = this->model->user_embedding->weights;
//             val_t* user_emb_ptr = user_embedding_weights->read_row(user_id, t_buf->user_emb_buf);
//             behavior_aggregator.forward(user_id, user_emb_ptr, this->model->item_embedding, t_buf);
//             aggr_user_embedding->write_row(user_id, user_emb_ptr);
//         }
//     }

//     Eigen::Map<Eigen::Array<val_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> user_emb_mat(aggr_user_embedding->data, num_users, emb_dim);
//     Eigen::Map<Eigen::Array<val_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> item_emb_mat(item_embedding_weights->data, num_items, emb_dim);
//     this->sim_matrix = user_emb_mat * item_emb_mat.transpose();
//     return PyMatrix({num_users, num_items}, {num_items * sizeof(val_t), sizeof(val_t)}, this->sim_matrix.data());
// }


        }
    }
}

// rm -rf ./cf_c.cpython-38-x86_64-linux-gnu.so && make -j && cp ./cf_c.cpython-38-x86_64-linux-gnu.so ../cf/ 