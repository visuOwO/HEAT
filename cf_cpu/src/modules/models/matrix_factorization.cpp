#include <unordered_map>
#include "matrix_factorization.hpp"

namespace cf {
    namespace modules {
        namespace models {

            MatrixFactorization::MatrixFactorization(const std::shared_ptr<CFConfig> config, val_t *user_weights,
                                                     val_t *item_weights, idx_t start_item_id, idx_t end_item_id)
                    : Model(config, user_weights, item_weights, start_item_id, end_item_id) {
            }


            val_t MatrixFactorization::forward_backward(idx_t user_id, idx_t pos_id, std::vector<idx_t> &neg_ids,
                                                        std::vector<idx_t> neg_tile,
                                                        const std::shared_ptr<CFModules> cf_modules,
                                                        memory::ThreadBuffer *t_buf,
                                                        behavior_aggregators::BehaviorAggregator *behavior_aggregator,
                                                        std::unordered_map<idx_t, std::vector<val_t> > &remote_item_embeddings_grads // to store gradients of embeddings
            ) {
                //printf("MatrixFactorization::forward_backward\n");
                double start_time = omp_get_wtime();
                double forward_time = start_time;
                double f_b_time = start_time;

                const int emb_dim = cf_modules->cf_config->emb_dim;
                const int num_negs = cf_modules->cf_config->num_negs;

                // compute user positive dot products
                memory::Array<val_t> *user_embedding_weights = this->user_embedding->weights;
                memory::Array<val_t> *user_embedding_grads = this->user_embedding->grads;
                memory::Array<val_t> *item_embedding_weights = this->item_embedding->weights;
                memory::Array<val_t> *item_embedding_grads = this->item_embedding->grads;

                auto * user_emb_buf = new val_t[emb_dim];
                val_t *user_emb_ptr = user_embedding_weights->read_row(user_id-this->user_embedding->start_idx, user_emb_buf);

                /*printf("test embedding\n");
                std::vector<val_t> emb(emb_dim, 0.0);
                user_embedding_weights->read_row(3-this->user_embedding->start_idx, emb.data());
                printf("%f %f %f %f\n", emb[0], emb[1], emb[2], emb[3]);
                if (user_id == 3) {
                    printf("%f %f %f %f\n", user_emb_ptr[0], user_emb_ptr[1], user_emb_ptr[2], user_emb_ptr[3]);
                }
                printf("finish test embedding\n");

                printf("test aggregator weights\n");
                std::vector<val_t> weights0(emb_dim * emb_dim, 0.0);
                printf("%f %f %f %f\n", behavior_aggregator->aggregator_weights->weights0(0, 0),
                       behavior_aggregator->aggregator_weights->weights0(0, 1),
                       behavior_aggregator->aggregator_weights->weights0(1, 0),
                       behavior_aggregator->aggregator_weights->weights0(1, 1));
                printf("finish test aggregator weights\n");*/

                // val_t* pos_emb_ptr = item_embedding_weights->read_row(pos_id, t_buf->pos_emb_buf);

                val_t *pos_emb_ptr = t_buf->pos_emb_buf + pos_id * emb_dim;


                if (remote_item_embeddings_grads.find(t_buf->pos_item_ids[pos_id]) ==
                    remote_item_embeddings_grads.end()) {
                    remote_item_embeddings_grads.insert(
                            {t_buf->pos_item_ids[pos_id], std::vector<val_t>(emb_dim, 0.0)});    // initialize gradients to 0
                }

                //printf("MatrixFactorization::forward_backward: user_id=%lu, pos_id=%lu, neg_ids.size()=%zu\n", user_id, pos_id, neg_ids.size());

                /*for (auto i = 0; i < emb_dim; i++) {
                    printf("pos_emb_ptr[%d]=%f\n", i, pos_emb_ptr[i]);
                }
                printf("\n");

                for (auto i = 0; i < num_negs; i++) {
                    printf("neg_ids[%d]=%lu\n", i, neg_ids[i]);
                }*/

                double end_time = omp_get_wtime();
                t_buf->time_map["read_emb"] = t_buf->time_map["read_emb"] + (end_time - start_time);
                start_time = end_time;

                behavior_aggregator->forward(user_id, user_emb_ptr, this->item_embedding, t_buf);
                end_time = omp_get_wtime();
                t_buf->time_map["aggr_f"] = t_buf->time_map["aggr_f"] + (end_time - start_time);
                start_time = end_time;

                //printf("BehaviorAggregator finish forward\n");

                Eigen::Map<Eigen::Matrix<val_t, 1, Eigen::Dynamic, Eigen::RowMajor>> user_emb(user_emb_ptr, 1, emb_dim);
                Eigen::Map<Eigen::Matrix<val_t, 1, Eigen::Dynamic, Eigen::RowMajor>> pos_emb(pos_emb_ptr, 1, emb_dim);


                val_t user_user_dot = user_emb.dot(user_emb);
                val_t pos_pos_dot = pos_emb.dot(pos_emb);
                val_t user_pos_dot = user_emb.dot(pos_emb);
                end_time = omp_get_wtime();
                t_buf->time_map["dot"] = t_buf->time_map["dot"] + (end_time - start_time);
                start_time = end_time;

                // compute user positive cosine similarity, partial gradient
                // machine epsilon
                const val_t eps = 1e-8;
                val_t user_norm = std::sqrt(std::max(user_user_dot, eps));
                val_t pos_norm = std::sqrt(std::max(pos_pos_dot, eps));
                val_t user_norm3 = user_norm * user_norm * user_norm;
                val_t pos_norm3 = pos_norm * pos_norm * pos_norm;

                val_t r_u3_p = 1 / (user_norm3 * pos_norm);
                val_t r_u_p3 = 1 / (user_norm * pos_norm3);
                Eigen::Matrix<val_t, 1, Eigen::Dynamic, Eigen::RowMajor> u_p_cos_u_grad =
                        (user_user_dot * pos_emb - user_pos_dot * user_emb) * r_u3_p;
                Eigen::Matrix<val_t, 1, Eigen::Dynamic, Eigen::RowMajor> u_p_cos_p_grad =
                        -(pos_pos_dot * user_emb - user_pos_dot * pos_emb) * r_u_p3;
                end_time = omp_get_wtime();
                t_buf->time_map["norm"] = t_buf->time_map["norm"] + (end_time - start_time);

                start_time = end_time;

                //printf("calculate gradient\n");

                // examine negative embeddings
                /*printf("start examine negative embeddings\n");
                  for (idx_t neg_idx = 0; neg_idx < num_negs; ++neg_idx) {
                    idx_t neg_id = neg_ids[neg_idx];
                    printf("neg_id=%lu\n", neg_id);
                    for (idx_t i = 0; i < emb_dim; i++) {
                        printf("%f ", t_buf->tiled_neg_emb_buf[neg_idx * emb_dim + i]);
                    }
                    printf("\n");
                }
                printf("finish examine negative embeddings\n");*/

                // compute user negative dot products
                for (idx_t neg_idx = 0; neg_idx < num_negs; ++neg_idx) {
                    idx_t neg_id = neg_ids[neg_idx];
                    //printf("neg_id=%lu\n", neg_id);

                    // val_t* neg_emb_ptr = item_embedding_weights->read_row(neg_id, t_buf->neg_emb_buf0);
                    val_t *neg_emb_ptr = t_buf->neg_emb_buf1;

                    // memcpy(t_buf->neg_emb_buf1 + neg_idx * emb_dim, neg_emb_ptr, emb_dim * sizeof(val_t));

                    // read negative embedding from tiled memory
                    // maybe it's a waste of time to read from tiled memory, we can just use the embedding in the buffer
                    memcpy(t_buf->neg_emb_buf1 + neg_idx * emb_dim, t_buf->tiled_neg_emb_buf + neg_id * emb_dim,
                           emb_dim * sizeof(val_t));

                    /*for (auto i = 0; i < emb_dim; i++) {
                        printf("neg_emb_ptr[%lu][%d]=%f\n", neg_idx, i, neg_emb_ptr[neg_idx * emb_dim + i]);
                    }
                    printf("\n");*/

                    //val_t* neg_grad_ptr = item_embedding_grads->read_row(neg_id, t_buf->neg_grad_buf);
                    //Eigen::Map<Eigen::Matrix<val_t, 1, Eigen::Dynamic, Eigen::RowMajor>> neg_emb_grad(neg_grad_ptr, 1, emb_dim);
                }

                // start examine embeddings
                /*printf("start examine embeddings\n");
                printf("user_id=%lu, pos_id=%lu, neg_id=%lu\n", user_id, pos_id, neg_ids[0]);
                printf("%f %f %f %f\n", user_emb_ptr[0], user_emb_ptr[1], user_emb_ptr[2], user_emb_ptr[3]);
                printf("%f %f %f %f\n", pos_emb_ptr[0], pos_emb_ptr[1], pos_emb_ptr[2], pos_emb_ptr[3]);
                printf("%f %f %f %f\n", t_buf->neg_emb_buf1[0], t_buf->neg_emb_buf1[1], t_buf->neg_emb_buf1[2], t_buf->neg_emb_buf1[3]);
                printf("finish examine embeddings\n");*/

                end_time = omp_get_wtime();
                t_buf->time_map["read_emb"] = t_buf->time_map["read_emb"] + (end_time - start_time);
                start_time = end_time;

                //printf("calculate negative dot products\n") ;

                Eigen::Map<Eigen::Matrix<val_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> neg_embs(
                        t_buf->neg_emb_buf1, num_negs, emb_dim);
                Eigen::Matrix<val_t, 1, Eigen::Dynamic> user_neg_dots = user_emb * neg_embs.transpose();
                Eigen::Matrix<val_t, 1, Eigen::Dynamic> pos_neg_dots = pos_emb * neg_embs.transpose();
                Eigen::Matrix<val_t, 1, Eigen::Dynamic> neg_neg_dots = (neg_embs.array() *
                                                                        neg_embs.array()).transpose().colwise().sum();
                end_time = omp_get_wtime();
                t_buf->time_map["dot"] = t_buf->time_map["dot"] + (end_time - start_time);
                start_time = end_time;

                // examine positive embedding
                /*printf("start examine positive embedding\n");
                  for (idx_t i = 0; i < emb_dim; i++) {
                    printf("%f ", pos_emb_ptr[i]);
                }
                printf("\n");
                printf("finish examine positive embedding\n");*/



                // compute user negative cosine similarity, score
                //printf("calculate negative cosine similarity\n") ;
                val_t user_pos_cos = user_pos_dot / (user_norm * pos_norm);
                Eigen::Array<val_t, 1, Eigen::Dynamic> selected_negs_negs_dots = (neg_neg_dots.array() < eps).select(
                        eps, neg_neg_dots.array());
                Eigen::Array<val_t, 1, Eigen::Dynamic> neg_norm = selected_negs_negs_dots.sqrt();
                Eigen::Array<val_t, 1, Eigen::Dynamic> neg_norm3 = neg_norm * neg_norm * neg_norm;
                Eigen::Array<val_t, 1, Eigen::Dynamic> user_negs_cos = user_neg_dots.array() / (user_norm * neg_norm);
                Eigen::Array<val_t, 1, Eigen::Dynamic> score = user_negs_cos - user_pos_cos;
                end_time = omp_get_wtime();
                t_buf->time_map["norm"] = t_buf->time_map["norm"] + (end_time - start_time);
                start_time = end_time;

                //printf("calculate score\n") ;
                double score_mul = 1.0 / 0.07;
                // double score_mul = 1.0;
                score *= score_mul;
                val_t max_score = score.maxCoeff();
                Eigen::Array<val_t, 1, Eigen::Dynamic> exp_score = (score - max_score).exp();
                val_t exp_score_sum = exp_score.sum();
                exp_score_sum += std::exp(-1.0 * max_score);
                val_t loss = max_score + std::log(exp_score_sum);
                Eigen::Array<val_t, 1, Eigen::Dynamic> loss_grad = (exp_score / exp_score_sum) * score_mul;
                end_time = omp_get_wtime();
                t_buf->time_map["loss"] = t_buf->time_map["loss"] + (end_time - start_time);
                t_buf->time_map["forward"] = t_buf->time_map["forward"] + (end_time - forward_time);
                start_time = end_time;
                double backward_time = end_time;
                // Eigen::Array<val_t, 1, Eigen::Dynamic> loss = 1.0 / (1.0 + score.abs().exp());
                // Eigen::Array<val_t, 1, Eigen::Dynamic> loss_grad = score_mul * loss * (1.0 - loss);

                //printf("calculate loss\n") ;

                val_t *user_grad_ptr = user_embedding_grads->read_row(user_id, t_buf->user_grad_buf);
                //val_t* pos_grad_ptr = item_embedding_grads->read_row(pos_id, t_buf->pos_grad_buf);
                val_t *pos_grad_ptr = remote_item_embeddings_grads[t_buf->pos_item_ids[pos_id]].data();
                /*printf("pos_ids and id are: %lu %lu\n", t_buf->pos_item_ids[pos_id], pos_id);
                for (auto &i: remote_item_embeddings_grads) {
                    printf("%lu ", i.first);
                }
                printf("\n");
                std::cout << remote_item_embeddings_grads[t_buf->pos_item_ids[pos_id]].size() << std::endl;
                printf("%f\n",pos_grad_ptr[0]);*/
                Eigen::Map<Eigen::Matrix<val_t, 1, Eigen::Dynamic, Eigen::RowMajor>> user_emb_grad(user_grad_ptr, 1,
                                                                                                   emb_dim);
                Eigen::Map<Eigen::Matrix<val_t, 1, Eigen::Dynamic, Eigen::RowMajor>> pos_emb_grad(pos_grad_ptr, 1,
                                                                                                  emb_dim);

                // Eigen::Matrix<val_t, 1, Eigen::Dynamic> user_emb_grad = Eigen::Matrix<val_t, 1, Eigen::Dynamic>::Zero(1, emb_dim);
                // Eigen::Matrix<val_t, 1, Eigen::Dynamic> pos_emb_grad = Eigen::Matrix<val_t, 1, Eigen::Dynamic>::Zero(1, emb_dim);
                // Eigen::Matrix<val_t, 1, Eigen::Dynamic> neg_emb_grad = Eigen::Matrix<val_t, 1, Eigen::Dynamic>::Zero(1, emb_dim);
                const val_t l2 = cf_modules->cf_config->l2;

                //printf("calculate neg gradient\n") ;
                for (idx_t neg_idx = 0; neg_idx < neg_ids.size(); ++neg_idx) {
                    idx_t neg_id = neg_ids[neg_idx];
                    // val_t* neg_emb_ptr = item_embedding_weights->read_row(neg_id, t_buf->neg_emb_buf0);
                    // Eigen::Map<Eigen::Matrix<val_t, 1, Eigen::Dynamic, Eigen::RowMajor>> neg_emb(neg_emb_ptr, 1, emb_dim);

                    //val_t* neg_grad_ptr = item_embedding_grads->read_row(neg_id, t_buf->neg_grad_buf);
                    if (remote_item_embeddings_grads.find(neg_tile[neg_id]) == remote_item_embeddings_grads.end()) {
                        remote_item_embeddings_grads.insert(
                                {neg_tile[neg_id], std::vector<val_t>(emb_dim, 0.0)});    // initialize gradients to 0
                    }
                    val_t *neg_grad_ptr = remote_item_embeddings_grads[neg_tile[neg_id]].data();
                    Eigen::Map<Eigen::Matrix<val_t, 1, Eigen::Dynamic, Eigen::RowMajor>> neg_emb_grad(neg_grad_ptr, 1,
                                                                                                      emb_dim);

                    //printf("test1\n");
                    val_t r_u3_n = 1 / (user_norm3 * neg_norm(0, neg_idx));
                    val_t r_u_n3 = 1 / (user_norm * neg_norm3(0, neg_idx));
                    Eigen::Matrix<val_t, 1, Eigen::Dynamic> u_n_cos_u_grad =
                            (user_user_dot * neg_embs.row(neg_idx) - user_neg_dots(0, neg_idx) * user_emb) * r_u3_n;
                    Eigen::Matrix<val_t, 1, Eigen::Dynamic> u_n_cos_n_grad =
                            (neg_neg_dots(0, neg_idx) * user_emb - user_neg_dots(0, neg_idx) * neg_embs.row(neg_idx)) *
                            r_u_n3;

                    // printf("test2\n");

                    //printf("neg_emb is %f %f %f %f\n", neg_embs(neg_idx, 0), neg_embs(neg_idx, 1), neg_embs(neg_idx, 2),
                           //neg_embs(neg_idx, 3));
                    // printf("loss_grad(0, neg_idx)=%f\n", loss_grad(0, neg_idx));

                    /*printf("%f\n", user_grad_ptr[0]);
                    printf("%f\n", pos_grad_ptr[0]);
                    printf("%f\n", neg_grad_ptr[0]);*/

                    // regularize negative embedding
                    // neg_emb_grad += l2 * neg_embs.row(neg_idx);

                    user_emb_grad += loss_grad(0, neg_idx) * (u_n_cos_u_grad - u_p_cos_u_grad);
                    pos_emb_grad += loss_grad(0, neg_idx) * u_p_cos_p_grad;
                    neg_emb_grad += loss_grad(0, neg_idx) * u_n_cos_n_grad;


                    //cf_modules->optimizer->sparse_step(neg_embs.row(neg_idx).data(), neg_grad_ptr);


                    // item_embedding_weights->write_row(neg_id, neg_embs.row(neg_idx).data());
                    // item_embedding_grads->write_row(neg_id, neg_grad_ptr);
                }

                //printf("test3\n");
                behavior_aggregator->backward(user_grad_ptr);
                end_time = omp_get_wtime();
                t_buf->time_map["aggr_b"] = t_buf->time_map["aggr_b"] + (end_time - start_time);
                start_time = end_time;
                //printf("test4\n");

                //printf("calculate aggr\n") ;

                // val_t neg_scale = 1.0 / neg_ids.size();
                // #pragma omp simd
                // for (idx_t i = 0; i < emb_dim; ++i)
                // {
                //     user_grad_ptr[i] *= neg_scale;
                //     pos_grad_ptr[i] *= neg_scale;
                // }

                // user_emb_grad += l2 * user_emb;
                cf_modules->optimizer->sparse_step(user_emb_ptr, user_grad_ptr);

                // pos_emb_grad += l2 * pos_emb;
                cf_modules->optimizer->sparse_step(pos_emb_ptr, pos_grad_ptr);

                // examine the weights that write to the embedding
                /*printf("start examine the weights that write to the embedding\n");
                printf("user_id=%lu, pos_id=%lu, neg_id=%lu\n", user_id, pos_id, neg_ids[0]);
                printf("%f %f %f %f\n", user_emb_ptr[0], user_emb_ptr[1], user_emb_ptr[2], user_emb_ptr[3]);
                printf("finish examine the weights that write to the embedding\n");*/

                user_embedding_weights->write_row(user_id, user_emb_ptr);
                user_embedding_grads->write_row(user_id, user_grad_ptr);
                //item_embedding_weights->write_row(pos_id, pos_emb_ptr);
                //item_embedding_grads->write_row(pos_id, pos_grad_ptr);


                end_time = omp_get_wtime();
                t_buf->time_map["backward"] = t_buf->time_map["backward"] + (end_time - backward_time);
                t_buf->time_map["f_b"] = t_buf->time_map["f_b"] + (end_time - f_b_time);

                // free memory
                delete[] user_emb_buf;

                return loss;
            }

        }
    }
}