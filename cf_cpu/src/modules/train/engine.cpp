#include "engine.hpp"
#include "utils/Data_shuffle.h"
#include <set>
#include <unordered_set>

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
        idx_t Data_shuffle::num_items;
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

                std::unordered_map<std::string, double> time_map;
                time_map["shuffle_and_update_item_grads"] = 0.0;
                time_map["shuffle_embs"] = 0.0;
                time_map["shuffle_his_embs"] = 0.0;
                time_map["forward_backward"] = 0.0;
                time_map["shuffle_negs"] = 0.0;
                time_map["init"] = 0.0;

                double start_time, end_time;

                start_time = MPI_Wtime();

                Data_shuffle::num_items = this->cf_config->num_items;
                Data_shuffle::comm = MPI_COMM_WORLD;
                Data_shuffle::world_size = world_size;
                Data_shuffle::rank = rank;
                Data_shuffle::emb_dim = this->cf_config->emb_dim;
                Data_shuffle::cf_modules = this->cf_modules;

                idx_t process_status = 1;   // 1 means the process is not finished
                idx_t system_status;

                idx_t mini_batch_size = cf_config->mini_batch_size;
                int num_thread = 2;
                idx_t batch_size = mini_batch_size * num_thread;

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

                std::vector<idx_t> neg_ids(num_negs);
                idx_t seed = (this->epoch - 1) * this->train_data->data_rows;
                idx_t tile_space = this->model->item_embedding->end_idx - this->model->item_embedding->start_idx + 1;

                // initialize nagative sampler
                negative_samplers::NegativeSampler *negative_sampler = nullptr;
                if (this->cf_config->neg_sampler == 1) {
                    negative_sampler = static_cast<negative_samplers::NegativeSampler *>(
                            new negative_samplers::RandomTileNegativeSampler(this->cf_config, seed, tile_space));
                } else {
                    negative_sampler = static_cast<negative_samplers::NegativeSampler *>(
                            new negative_samplers::UniformRandomNegativeSampler(this->cf_config, seed));
                }

                memory::ThreadBuffer *t_buf = new memory::ThreadBuffer(this->cf_config->emb_dim, num_negs,
                                                                       batch_size,
                                                                       this->cf_config->tile_size);

                // behavior_aggregators::BehaviorAggregator* behavior_aggregator = nullptr;
                behavior_aggregators::BehaviorAggregator behavior_aggregator(train_data_ptr, aggregator_weights_ptr,
                                                                             cf_config);

                std::unordered_map<idx_t, std::vector<val_t> > item_emb_grads;
                std::unordered_map<idx_t, std::vector<val_t> > user_emb_grads;


                end_time = MPI_Wtime();
                time_map["init"] += end_time - start_time;

                for (idx_t i = 0; i < iterations; i += batch_size) {

                    MPI_Allreduce(&process_status, &system_status, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);

                    start_time = MPI_Wtime();
                    Data_shuffle::shuffle_and_update_item_grads(item_emb_grads, model->item_embedding,
                                                                this->cf_config->num_items);
                    Data_shuffle::shuffle_and_update_item_grads(user_emb_grads, model->user_embedding,
                                                                this->cf_config->num_users);
                    end_time = MPI_Wtime();
                    time_map["shuffle_and_update_item_grads"] += end_time - start_time;
                    item_emb_grads.clear();
                    user_emb_grads.clear();

                    start_time = MPI_Wtime();
                    for (int j = 0; j < batch_size; j++) {
                        if (i + j >= iterations) {
                            break;
                        }
                        idx_t id = this->positive_sampler->read(i + j);
                        idx_t uid, iid;
                        this->train_data->read_user_item(id, uid, iid);
                        t_buf->pos_item_ids[j] = iid;
                        t_buf->user_ids[j] = uid;
                        if (item_emb_grads.find(iid) == item_emb_grads.end()) {
                            item_emb_grads[iid] = std::vector<val_t>(this->cf_config->emb_dim, 0.0);
                        }
                        if (user_emb_grads.find(uid) == user_emb_grads.end()) {
                            user_emb_grads[uid] = std::vector<val_t>(this->cf_config->emb_dim, 0.0);
                        }
                    }

                    //

                    // shuffle the positive embeddings to t_buf->tiled_pos_emb_buf
                    //printf("start shuffling pos embeddings\n");
                    Data_shuffle::shuffle_embs(std::vector<idx_t>(t_buf->pos_item_ids, t_buf->pos_item_ids +
                                                                                       batch_size),
                                               t_buf->pos_emb_buf,
                                               this->model->item_embedding, this->cf_config->num_items);

                    Data_shuffle::shuffle_embs(std::vector<idx_t>(t_buf->user_ids, t_buf->user_ids +
                                                                                   batch_size),
                                               t_buf->user_emb_buf,
                                               this->model->user_embedding, this->cf_config->num_users);

                    end_time = MPI_Wtime();
                    time_map["shuffle_embs"] += end_time - start_time;

                    start_time = MPI_Wtime();
#pragma omp parallel for schedule(static, mini_batch_size) num_threads(num_thread) reduction(+:local_loss) shared(i, behavior_aggregator, t_buf, item_emb_grads)

                    for (idx_t j = 0; j < batch_size; j++) {
                        idx_t user_id = 0;
                        idx_t item_id = 0;
                        if (i + j >= iterations) {
                            continue;
                        }
                        idx_t train_data_idx = this->positive_sampler->read(i + j);
                        this->train_data->read_user_item(train_data_idx, user_id, item_id);
                        negative_sampler->sampling(neg_ids);
                        auto tmp = this->model->forward_backward(user_id, (i + j) % batch_size,
                                                                 neg_ids,
                                                                 std::vector<idx_t>(),
                                                                 this->cf_modules, t_buf, &behavior_aggregator,
                                                                 item_emb_grads, user_emb_grads);

                        local_loss = local_loss + tmp;
                    }
                    end_time = MPI_Wtime();
                    time_map["forward_backward"] += end_time - start_time;
                }
                process_status = 0;   // 0 means the process is finished
                item_emb_grads.clear();
                while (true) {
                    MPI_Allreduce(&process_status, &system_status, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
                    //printf("system_status: %d\n", system_status) ;
                    if (system_status == 0) {
                        break;
                    } else {
                        // Keep sending and receiving data until all the processes are finished
                        Data_shuffle::shuffle_and_update_item_grads(item_emb_grads,
                                                                    this->model->item_embedding,
                                                                    this->cf_config->num_items);
                        Data_shuffle::shuffle_and_update_item_grads(user_emb_grads,
                                                                    this->model->user_embedding,
                                                                    this->cf_config->num_users);
                        Data_shuffle::shuffle_embs(std::vector<idx_t>(),
                                                   t_buf->pos_emb_buf,
                                                   this->model->item_embedding, this->cf_config->num_items);
                        Data_shuffle::shuffle_embs(std::vector<idx_t>(),
                                                   t_buf->tiled_neg_emb_buf,
                                                   this->model->item_embedding, this->cf_config->num_items);
                        item_emb_grads.clear();
                    }
                }

                // delete[] local_aggregator_weights;
                double global_loss;
                idx_t total_iteration;
                MPI_Allreduce(&local_loss, &global_loss, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(&iterations, &total_iteration, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
                global_loss = global_loss / total_iteration;

                // free memory
                delete negative_sampler;
                delete t_buf;

                printf("init time: %f\n", time_map["init"]);
                printf("shuffle_and_update_item_grads time: %f\n", time_map["shuffle_and_update_item_grads"]);
                printf("shuffle_embs time: %f\n", time_map["shuffle_embs"]);
                printf("shuffle_his_embs time: %f\n", time_map["shuffle_his_embs"]);
                printf("forward_backward time: %f\n", time_map["forward_backward"]);
                printf("shuffle_negs time: %f\n", time_map["shuffle_negs"]);

                return global_loss;
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