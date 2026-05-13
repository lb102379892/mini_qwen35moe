/**
 * @file graph.cpp
 * @brief Qwen3.5-MoE 模型前向传播计算图构建实现
 * 
 * 该文件实现了 Qwen3.5-MoE 大语言模型的推理计算图构建，包括：
 * 1. Prefill 阶段：处理初始上下文输入，写入 KV 缓存
 * 2. Decode 阶段：逐 token 生成，支持多序列并行
 * 3. DeltaNet 层：Qwen3.5 特有的高效序列建模层
 * 4. MoE 层：混合专家前馈网络
 */
#include "graph/graph.h"
#include <algorithm>
#include <cstdlib>

namespace {
bool env_flag_enabled(const char* name) {
    const char* value = std::getenv(name);
    return value && value[0] != '\0' && value[0] != '0';
}

uint32_t decode_cache_capacity(uint32_t required, uint32_t context_len) {
    (void) required;
    return context_len;
}

void set_mask_data(ggml_tensor* tensor, const std::vector<float>& mask) {
    if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> mask_f16(mask.size());
        ggml_fp32_to_fp16_row(mask.data(), mask_f16.data(), static_cast<int64_t>(mask.size()));
        ggml_backend_tensor_set(tensor, mask_f16.data(), 0, mask_f16.size() * sizeof(ggml_fp16_t));
        return;
    }
    ggml_backend_tensor_set(tensor, mask.data(), 0, mask.size() * sizeof(float));
}
}

/**
 * @brief 构造函数：初始化 Qwen35moeForwardPass 对象
 */
Qwen35moeForwardPass::Qwen35moeForwardPass() {
}

/**
 * @brief 析构函数：释放 ggml 上下文资源
 */
Qwen35moeForwardPass::~Qwen35moeForwardPass() {
    if (ctx_) {
        ggml_free(ctx_);
    }
}

/**
 * @brief 初始化前向传播器
 * 
 * @param context_len 上下文长度
 * @param max_batch_size 最大批处理大小
 * @param model 模型对象指针
 * @return 0 表示成功
 * 
 * 主要初始化工作：
 * 1. 初始化 ggml 上下文
 * 2. 构建层映射表（区分 Full Attention 层和 DeltaNet 层）
 * 3. 创建 KV 缓存（用于 Full Attention 层）
 * 4. 创建 DeltaNet 状态（用于 DeltaNet 层）
 */
int Qwen35moeForwardPass::init(const uint32_t context_len, const uint32_t max_batch_size, std::shared_ptr<Qwen35moeModel> model) {
    model_ = model;
    use_flash_attention_ = env_flag_enabled("QWEN35MOE_FLASH_ATTN");
    auto& m = model_->meta_->qwen35moe;

    context_len_ = context_len;
    max_batch_size_ = max_batch_size;

    // 预分配持久化缓冲区用于存储计算图元数据
    ctx_buffer_.resize(FP_GRAPH_SIZE_METADATA);

    struct ggml_init_params params = {
        .mem_size   = ctx_buffer_.size(),
        .mem_buffer = ctx_buffer_.data(),
        .no_alloc   = true,  // 使用预分配缓冲区，不允许自动分配
    };
    ctx_ = ggml_init(params);

    // 初始化层映射表：将物理层索引映射到 KV/DeltaNet 层索引
    kv_layer_map_.assign(m.block_count, -1);  // Full Attention 层映射
    dn_layer_map_.assign(m.block_count, -1);  // DeltaNet 层映射
    int kv_idx = 0, dn_idx = 0;
    for (uint32_t il = 0; il < m.block_count; ++il) {
        if (is_full_attention_layer(il))
            kv_layer_map_[il] = kv_idx++;  // 标记为 Full Attention 层
        else
            dn_layer_map_[il] = dn_idx++;  // 标记为 DeltaNet 层
    }

    // 创建 KV 缓存：用于存储注意力机制的 Key/Value 张量
    // Qwen3.5-MoE 有 10 个 Full Attention 层
    ggml_backend_t cache_backend = model_->get_curr_backend();

    const int n_kv_layers = kv_idx;  // Full Attention 层数量
    const uint32_t n_embd_k = static_cast<uint32_t>(m.head_count_kv * m.key_length);  // K 维度
    const uint32_t n_embd_v = static_cast<uint32_t>(m.head_count_kv * m.value_length); // V 维度
    kv_cache_ = std::make_unique<simple_kv_cache>(
        static_cast<uint32_t>(n_kv_layers),  // KV 层数
        context_len_,                         // 上下文长度
        max_batch_size_,                      // 最大批大小
        n_embd_k, n_embd_v,                   // K/V 维度
        GGML_TYPE_F16, GGML_TYPE_F16,         // 数据类型
        cache_backend                         // 后端设备
    );

    // 创建 DeltaNet 状态：用于存储 DeltaNet 层的循环状态
    // Qwen3.5-MoE 有 30 个 DeltaNet 层
    const int n_dn_layers = dn_idx;
    const uint32_t d_inner       = m.inner_size;     // 内部维度 = 4096
    const uint32_t num_v_heads   = m.time_step_rank; // V 头数量 = 32
    const uint32_t num_k_heads   = m.group_count;    // K 头数量 = 16
    const uint32_t head_v_dim    = d_inner / num_v_heads; // 每个 V 头维度 = 128
    const uint32_t conv_channels = d_inner + 2 * num_k_heads * m.state_size; // 卷积通道数 = 8192

    DeltaNetStateParams dn_state_hp {
        static_cast<uint32_t>(n_dn_layers),
        max_batch_size_,
        head_v_dim,
        m.state_size,  // head_k_dim = 128
        num_v_heads,
        conv_channels,
        m.conv_kernel, // 卷积核大小 = 4
        cache_backend
    };
    dn_state_ = std::make_unique<DeltaNetState>(dn_state_hp);

    return 0;
}

void Qwen35moeForwardPass::set_flash_attention_enabled(bool enabled) {
    use_flash_attention_ = enabled;
}

void Qwen35moeForwardPass::configure_device_sampling(int top_k, float temperature) {
    sampling_top_k_ = top_k;
    sampling_temperature_ = temperature;
}

/**
 * @brief 重置 ggml 上下文
 * 
 * 在每次构建新的计算图前调用，确保上下文干净
 */
void Qwen35moeForwardPass::reset_context() {
    if (ctx_) {
        ggml_free(ctx_);
    }
    cached_decode_graph_ = nullptr;
    cached_decode_graph_allocated_ = false;
    cached_decode_kv_capacity_ = 0;
    cached_decode_scratch_pos_ = 0;
    cached_decode_last_mask_pos_ = -1;
    cached_decode_mask_tensors_.clear();
    cached_decode_mask_f32_.clear();
    cached_decode_mask_f16_.clear();
    struct ggml_init_params params = {
        .mem_size   = ctx_buffer_.size(),
        .mem_buffer = ctx_buffer_.data(),
        .no_alloc   = true, 
    };
    ctx_ = ggml_init(params);
}

void Qwen35moeForwardPass::reset_sequence(uint32_t slot_idx) {
    if (slot_idx >= max_batch_size_) {
        throw std::runtime_error("Qwen35moeForwardPass::reset_sequence: slot_idx out of range");
    }
    if (kv_cache_) {
        kv_cache_->clear_slot(slot_idx);
    }
    if (dn_state_) {
        dn_state_->clear_slot(slot_idx);
    }
    if (slot_idx < snapkv_seq_pos_.size()) {
        snapkv_seq_pos_[slot_idx] = 0;
    }
}

void Qwen35moeForwardPass::prepare_cached_decode_graph(ggml_backend_sched_t scheduler, uint32_t slot_idx, uint32_t kv_capacity) {
    if (scheduler == nullptr) {
        throw std::runtime_error("Qwen35moeForwardPass::prepare_cached_decode_graph: scheduler is null");
    }
    if (context_len_ == 0) {
        throw std::runtime_error("Qwen35moeForwardPass::prepare_cached_decode_graph: context length is zero");
    }
    if (slot_idx >= max_batch_size_) {
        throw std::runtime_error("Qwen35moeForwardPass::prepare_cached_decode_graph: slot_idx out of range");
    }
    if (kv_capacity == 0) {
        throw std::runtime_error("Qwen35moeForwardPass::prepare_cached_decode_graph: kv_capacity cannot be zero");
    }
    if (kv_capacity > context_len_) {
        throw std::runtime_error("Qwen35moeForwardPass::prepare_cached_decode_graph: kv_capacity exceeds context_len_");
    }

    const uint32_t saved_pos = kv_cache_ ? kv_cache_->get_pos(slot_idx) : 0;
    const uint32_t saved_snap = slot_idx < snapkv_seq_pos_.size() ? snapkv_seq_pos_[slot_idx] : 0;
    const uint32_t scratch_pos = kv_capacity - 1;

    if (kv_cache_) {
        kv_cache_->set_pos(scratch_pos, slot_idx);
    }
    if (slot_idx < snapkv_seq_pos_.size()) {
        snapkv_seq_pos_[slot_idx] = 0;
    }

    std::vector<int32_t> dummy_tokens(1, 0);
    cached_decode_graph_ = build_prefill_graph(dummy_tokens, static_cast<int>(scratch_pos), slot_idx);
    cached_decode_slot_ = slot_idx;
    cached_decode_kv_capacity_ = kv_capacity;
    cached_decode_scratch_pos_ = scratch_pos;
    cached_decode_last_mask_pos_ = -1;
    cached_decode_mask_tensors_.clear();
    for (uint32_t il = 0; il < model_->meta_->qwen35moe.block_count; ++il) {
        if (!is_full_attention_layer(il)) {
            continue;
        }
        char name[32];
        std::snprintf(name, sizeof(name), "kq_mask.%u", il);
        ggml_tensor* kq_mask = ggml_graph_get_tensor(cached_decode_graph_, name);
        if (kq_mask) {
            cached_decode_mask_tensors_.push_back(kq_mask);
        }
    }

    if (kv_cache_) {
        kv_cache_->set_pos(saved_pos, slot_idx);
    }
    if (slot_idx < snapkv_seq_pos_.size()) {
        snapkv_seq_pos_[slot_idx] = saved_snap;
    }

    ggml_backend_sched_reset(scheduler);
    if (!ggml_backend_sched_alloc_graph(scheduler, cached_decode_graph_)) {
        throw std::runtime_error("Qwen35moeForwardPass::prepare_cached_decode_graph: failed to allocate graph");
    }
    cached_decode_graph_allocated_ = true;
}

/**
 * @brief 设置缓存解码图的输入数据（图复用模式）
 * 
 * 该函数用于解码阶段（Decode），在预构建的计算图上更新输入数据，实现图复用。
 * 相比于每次解码都重建图，这种方式可以显著减少 CPU 开销，提高 GPU 利用率。
 * 
 * @param gf 预构建的解码计算图（由 build_cached_decode_graph 创建）
 * @param token 当前要解码的单个 token ID
 * @param pos 当前 token 在序列中的位置（从 0 开始）
 * 
 * @throws std::runtime_error 如果图中缺少必要的输入张量
 * 
 * 实现步骤：
 * 1. 更新 token 输入张量
 * 2. 更新位置输入张量
 * 3. 动态更新注意力掩码（确保因果性）
 */
void Qwen35moeForwardPass::set_cached_decode_inputs(ggml_cgraph* gf, int32_t token, int pos) {
    // ==================== 步骤1：更新 Token 输入 ====================
    // 从计算图中获取名为 "tokens" 的输入张量
    ggml_tensor* tok_t = ggml_graph_get_tensor(gf, "tokens");
    if (!tok_t) {
        throw std::runtime_error("qwen36: 'tokens' tensor missing from cached decode graph");
    }
    // 将当前 token ID 写入张量（解码阶段每次只处理一个 token）
    ggml_backend_tensor_set(tok_t, &token, 0, sizeof(int32_t));

    // ==================== 步骤2：更新位置输入 ====================
    // 从计算图中获取名为 "inp_pos" 的位置张量
    ggml_tensor* pos_t = ggml_graph_get_tensor(gf, "inp_pos");
    if (!pos_t) {
        throw std::runtime_error("qwen36: 'inp_pos' tensor missing from cached decode graph");
    }
    // 将当前位置写入张量
    ggml_backend_tensor_set(pos_t, &pos, 0, sizeof(int32_t));

    // ==================== 步骤3：动态更新注意力掩码 ====================
    // scratch_pos 是上下文缓冲区的最后一个位置，用于特殊用途（如 KV 缓存优化）
    const uint32_t scratch_pos = cached_decode_scratch_pos_;

    if (cached_decode_mask_tensors_.empty()) {
        for (uint32_t il = 0; il < model_->meta_->qwen35moe.block_count; ++il) {
            if (!is_full_attention_layer(il)) {
                continue;
            }
            char name[32];
            std::snprintf(name, sizeof(name), "kq_mask.%u", il);
            ggml_tensor* kq_mask = ggml_graph_get_tensor(gf, name);
            if (kq_mask) {
                cached_decode_mask_tensors_.push_back(kq_mask);
            }
        }
    }

    uint32_t prepared_n_kv = 0;
    bool f16_ready = false;
    const bool can_incremental =
        cached_decode_last_mask_pos_ >= 0 && pos == cached_decode_last_mask_pos_ + 1;

    for (ggml_tensor* kq_mask : cached_decode_mask_tensors_) {
        const uint32_t n_kv = static_cast<uint32_t>(kq_mask->ne[0]);
        if (prepared_n_kv != n_kv) {
            const bool need_full_refresh =
                !can_incremental ||
                cached_decode_mask_f32_.size() != n_kv ||
                cached_decode_last_mask_pos_ < 0;
            if (need_full_refresh) {
                cached_decode_mask_f32_.assign(n_kv, -INFINITY);
                const uint32_t valid_prev = pos > 0 ? static_cast<uint32_t>(pos) : 0;
                for (uint32_t j = 0; j < valid_prev && j < n_kv; ++j) {
                    cached_decode_mask_f32_[j] = 0.0f;
                }
                if (scratch_pos < n_kv) {
                    cached_decode_mask_f32_[scratch_pos] = 0.0f;
                }
            } else {
                const uint32_t newly_visible = static_cast<uint32_t>(pos - 1);
                if (newly_visible < n_kv) {
                    cached_decode_mask_f32_[newly_visible] = 0.0f;
                }
            }
            prepared_n_kv = n_kv;
            f16_ready = false;
        }

        if (kq_mask->type == GGML_TYPE_F16) {
            if (!f16_ready) {
                cached_decode_mask_f16_.resize(cached_decode_mask_f32_.size());
                ggml_fp32_to_fp16_row(
                    cached_decode_mask_f32_.data(),
                    cached_decode_mask_f16_.data(),
                    static_cast<int64_t>(cached_decode_mask_f32_.size())
                );
                f16_ready = true;
            }
            ggml_backend_tensor_set(
                kq_mask,
                cached_decode_mask_f16_.data(),
                0,
                cached_decode_mask_f16_.size() * sizeof(ggml_fp16_t)
            );
        } else {
            ggml_backend_tensor_set(
                kq_mask,
                cached_decode_mask_f32_.data(),
                0,
                cached_decode_mask_f32_.size() * sizeof(float)
            );
        }
    }
    cached_decode_last_mask_pos_ = pos;
}

/**
 * @brief 构建 Prefill 阶段计算图
 * 
 * Prefill 阶段用于处理初始上下文输入，将所有 token 的 KV 值写入缓存
 * 
 * @param tokens 输入的 token 序列
 * @param pos 起始位置（通常为 0）
 * @param slot_idx 槽位索引（默认为 0）
 * @return 构建好的计算图
 * 
 * 处理流程：
 * 1. Token Embedding 查找
 * 2. 创建位置编码张量
 * 3. 遍历所有 Transformer 层：
 *    - Pre-attention RMSNorm
 *    - Full Attention 或 DeltaNet 层
 *    - 残差连接
 *    - Pre-FFN RMSNorm
 *    - MoE FFN 层
 *    - 残差连接
 * 4. 最终归一化 + LM Head
 */
ggml_cgraph* Qwen35moeForwardPass::build_prefill_graph(const std::vector<int32_t>& tokens, int pos, uint32_t slot_idx) {
    reset_context();

    const uint32_t n_tok = static_cast<uint32_t>(tokens.size());
    if (static_cast<uint32_t>(pos) + n_tok > context_len_) {
        throw std::runtime_error(
            "Qwen35moeForwardPass::build_prefill_graph: input exceeds context_len_"
        );
    }

    ggml_cgraph* gf = new_graph();

    auto& m = model_->meta_->qwen35moe;
    const uint32_t d_inner = m.inner_size;
    const uint32_t num_v_heads = m.time_step_rank;
    const uint32_t num_k_heads = m.group_count;
    const uint32_t head_v_dim = d_inner / num_v_heads;
    const uint32_t conv_channels = d_inner + 2 * num_k_heads * m.state_size;
    
    // DeltaNet 参数配置
    DeltaNetStateParams dn_hp {
        0, 0,             // n_dn_layers / n_slots 在 helper 中不使用
        head_v_dim,
        m.state_size,
        num_v_heads,
        conv_channels,
        m.conv_kernel,
        nullptr
    };

    // 1. Token Embedding：将 token ID 转换为词向量
    ggml_tensor* inpL = embedding(gf, tokens);
    set_tensor_name(inpL, "inpL");

    // 2. 位置张量（所有注意力层共享）
    ggml_tensor* inp_pos = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, n_tok);
    ggml_set_input(inp_pos);  // 标记为输入张量
    set_tensor_name(inp_pos, "inp_pos");
    ggml_build_forward_expand(gf, inp_pos);

    // 3. Transformer 层循环
    for (uint32_t il = 0; il < m.block_count; ++il) {
        ggml_tensor* inpSA = inpL;  // 保存残差连接的输入

        // ── Pre-attention RMSNorm ───────────────────────────────────────────
        struct ggml_tensor* attn_norm_weight = model_->get_attn_norm_weight(il);
        ggml_tensor* cur = build_norm(gf, inpL, attn_norm_weight, il);

        // ── Attention 或 DeltaNet 层 ────────────────────────────────────────
        if (is_full_attention_layer(il)) {
            // Full Attention 层：使用标准多头注意力 + KV 缓存
            int kv_idx = kv_layer_map_[il];

            // Gated Attention：联合 Q+Gate 投影，Q 权重输出
            // [(n_embd_head*2)*n_head, n_tokens]
            struct ggml_tensor* attn_q_weight = model_->get_attn_q_weight(il);
            struct ggml_tensor* attn_q_norm_weight = model_->get_attn_q_norm_weight(il);
            struct ggml_tensor* attn_k_weight = model_->get_attn_k_weight(il);
            struct ggml_tensor* attn_k_norm_weight = model_->get_attn_k_norm_weight(il);
            struct ggml_tensor* attn_v_weight = model_->get_attn_v_weight(il);
            struct ggml_tensor* attn_output_weight = model_->get_attn_output_weight(il);
            cur = build_gated_attention(
                ctx_, gf, kv_cache_.get(), cur, inp_pos,
                kv_idx, n_tok, slot_idx, il, attn_q_weight, 
                attn_q_norm_weight, attn_k_weight, attn_k_norm_weight, attn_v_weight, attn_output_weight,
                m.key_length, m.head_count, m.head_count_kv, m.dimension_count, m.freq_base,
                static_cast<int>(context_len_), m.layer_norm_rms_epsilon
            );
        } else {
            // DeltaNet 层：使用门控 delta 网络进行高效序列建模
            uint32_t dn_idx = static_cast<uint32_t>(dn_layer_map_[il]);
            cur = build_dn_layer(ctx_, gf, cur, dn_state_.get(), dn_hp, num_k_heads, 
                m.embedding_length, dn_idx, n_tok, slot_idx, m.layer_norm_rms_epsilon, il
            );
        }

        // ── 残差连接 1 (Attention / DeltaNet) ───────────────────────────────
        cur = ggml_add(ctx_, cur, inpSA);

        // ── Pre-FFN RMSNorm ─────────────────────────────────────────────────
        ggml_tensor* ffn_inp = cur;
        struct ggml_tensor* post_attention_norm = model_->get_post_attention_norm_weight(il);
        cur = build_norm(gf, cur, post_attention_norm, il);

        // ── MoE FFN：混合专家前馈网络 ───────────────────────────────────────
        cur = build_moe_layer(ctx_, gf, cur, il);

        // ── 残差连接 2 (FFN) ────────────────────────────────────────────────
        cur = ggml_add(ctx_, cur, ffn_inp);
        set_tensor_name(cur, "layer_out", il);

        inpL = cur;  // 更新输入为当前层输出
    }

    // 4. 最终归一化 + LM Head（输出 logits）
    build_output_head(gf, inpL);

    return gf;
}

/**
 * @brief 构建 Decode 阶段计算图（支持多序列并行）
 * 
 * Decode 阶段用于逐 token 生成，支持多序列并行处理
 * 每个序列每次只生成一个 token，但多个序列可以并行处理
 * 
 * @param tokens 每个序列当前的 token（batch 大小）
 * @param slots 每个 token 对应的槽位索引
 * @param positions 每个序列的当前位置
 * @return 构建好的计算图
 * 
 * 处理流程：
 * 1. Token Embedding
 * 2. 创建位置编码张量（每个 batch 元素一个）
 * 3. 创建 KV gather mask 和 gather indices（用于从稀疏缓存中收集 KV）
 * 4. 遍历 Transformer 层（与 Prefill 类似，但使用批处理版本）
 * 5. 最终归一化 + LM Head
 */
ggml_cgraph* Qwen35moeForwardPass::build_decoding_graph(const std::vector<int32_t>& tokens, const std::vector<uint32_t>& slots, const std::vector<int32_t>&  positions) {
    reset_context();

    ggml_cgraph* gf = new_graph();

    auto& m = model_->meta_->qwen35moe;
    const uint32_t n_batch = static_cast<uint32_t>(tokens.size());  // 批大小 = 序列数量

    // 从元数据派生 DeltaNet 参数
    const uint32_t d_inner = m.inner_size;
    const uint32_t num_v_heads = m.time_step_rank;
    const uint32_t num_k_heads = m.group_count;
    const uint32_t head_v_dim = d_inner / num_v_heads;
    const uint32_t conv_channels = d_inner + 2 * num_k_heads * m.state_size;
    DeltaNetParams dn_hp {
        static_cast<int>(m.embedding_length),
        static_cast<int>(head_v_dim * num_v_heads),
        static_cast<int>(m.state_size),
        static_cast<int>(num_k_heads),
        static_cast<int>(num_v_heads),
        static_cast<int>(head_v_dim),
        static_cast<int>(conv_channels),
        static_cast<int>(m.conv_kernel),
        m.layer_norm_rms_epsilon
    };

    // 1. Token Embedding
    ggml_tensor* inpL = embedding(gf, tokens);
    set_tensor_name(inpL, "inpL");

    // 2. 位置张量（每个 batch 元素一个位置）
    ggml_tensor* inp_pos = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, n_batch);
    ggml_set_input(inp_pos);
    set_tensor_name(inp_pos, "inp_pos");
    ggml_build_forward_expand(gf, inp_pos);

    // 3. KV gather mask — 所有注意力层共享
    // 计算最大物理缓存位置，确定 KV 长度
    uint32_t max_physical = 0;
    for (uint32_t s : slots) {
        uint32_t phys = get_physical_cache_pos(s);
        if (phys > max_physical) 
            max_physical = phys;
    }
    const uint32_t n_kv_len = max_physical + 1;  // +1 用于即将写入的 token
    
    // 创建 KV mask: [n_kv_len, 1, 1, n_batch]
    ggml_tensor* kq_mask = ggml_new_tensor_4d(ctx_, use_flash_attention_ ? GGML_TYPE_F16 : GGML_TYPE_F32, n_kv_len, 1, 1, n_batch);
    ggml_set_input(kq_mask);
    set_tensor_name(kq_mask, "kq_mask_b");
    ggml_build_forward_expand(gf, kq_mask);

    // 创建 gather indices: [n_batch * n_kv_len]
    // 用于从稀疏 KV 缓存中收集正确的 KV 对
    ggml_tensor* gather_indices = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, static_cast<int64_t>(n_batch * n_kv_len));
    ggml_set_input(gather_indices);
    set_tensor_name(gather_indices, "gather_indices");

    // 4. Transformer 层循环
    for (uint32_t il = 0; il < m.block_count; ++il) {
        ggml_tensor* inpSA = inpL;

        // Pre-attention RMSNorm
        struct ggml_tensor* attn_norm_weight = model_->get_attn_norm_weight(il);
        ggml_tensor* cur = build_norm(gf, inpL, attn_norm_weight, il);

        if (is_full_attention_layer(il)) {
            // Full Attention 层：使用批处理版本
            int kv_idx = kv_layer_map_[il];

            struct ggml_tensor* attn_q_weight = model_->get_attn_q_weight(il);
            struct ggml_tensor* attn_q_norm_weight = model_->get_attn_q_norm_weight(il);
            struct ggml_tensor* attn_k_weight = model_->get_attn_k_weight(il);
            struct ggml_tensor* attn_k_norm_weight = model_->get_attn_k_norm_weight(il);
            struct ggml_tensor* attn_v_weight = model_->get_attn_v_weight(il);
            struct ggml_tensor* attn_output_weight = model_->get_attn_output_weight(il);
            cur = build_gated_batched_attention(
                ctx_, gf, kv_cache_.get(), cur, inp_pos,
                kq_mask, gather_indices,
                kv_idx, slots, positions, il,
                attn_q_weight, attn_q_norm_weight,
                attn_k_weight, attn_k_norm_weight,
                attn_v_weight, attn_output_weight,
                m.key_length, m.head_count, m.head_count_kv,
                m.dimension_count, m.freq_base,
                static_cast<int>(context_len_),
                m.layer_norm_rms_epsilon
            );
        } else {
            // DeltaNet 层：解码阶段，每个槽位一个 token
            uint32_t dn_idx = static_cast<uint32_t>(dn_layer_map_[il]);
            DecodeArgs da{slots};
            PrefillArgs pa_unused{1, 0};

            cur = build_all_deltanet_layer(ctx_, gf, cur, dn_idx, Phase::Decode, pa_unused, &da, dn_state_.get(), dn_hp, il);
        }

        // 残差连接 1
        cur = ggml_add(ctx_, cur, inpSA);

        // Pre-FFN RMSNorm + MoE
        ggml_tensor* ffn_inp = cur;
        struct ggml_tensor* post_attention_norm = model_->get_post_attention_norm_weight(il);
        cur = build_norm(gf, cur, post_attention_norm, il);
        cur = build_moe_layer(ctx_, gf, cur, il);

        // 残差连接 2
        cur = ggml_add(ctx_, cur, ffn_inp);
        inpL = cur;
    }

    // 最终归一化 + LM Head
    build_output_head(gf, inpL);
    return gf;
}

/**
 * @brief 设置 Prefill 阶段的输入数据
 * 
 * @param gf 计算图
 * @param tokens 输入 token 序列
 * @param pos 起始位置
 * 
 * 设置内容：
 * 1. Token ID 张量
 * 2. 位置 ID 张量
 * 3. 因果注意力掩码（Causal mask）
 */
void Qwen35moeForwardPass::set_inputs(ggml_cgraph* gf, const std::vector<int32_t>& tokens, int pos) {
    const uint32_t n_tok = static_cast<uint32_t>(tokens.size());

    // 1. 设置 Token 张量
    ggml_tensor* tok_t = ggml_graph_get_tensor(gf, "tokens");
    if (!tok_t) 
        throw std::runtime_error("qwen36: 'tokens' tensor missing from graph");
    ggml_backend_tensor_set(tok_t, tokens.data(), 0, n_tok * sizeof(int32_t));

    // 2. 设置位置 ID 张量
    ggml_tensor* pos_t = ggml_graph_get_tensor(gf, "inp_pos");
    if (!pos_t) 
        throw std::runtime_error("qwen36: 'inp_pos' tensor missing from graph");

    std::vector<int32_t> pos_data(n_tok);
    for (uint32_t i = 0; i < n_tok; ++i) 
        pos_data[i] = pos + static_cast<int>(i);  // 位置从 pos 开始递增
    ggml_backend_tensor_set(pos_t, pos_data.data(), 0, n_tok * sizeof(int32_t));

    // 3. 设置因果注意力掩码（仅用于 Full Attention 层）
    // 掩码命名格式: "kq_mask.{physical_il}"
    for (uint32_t il = 0; il < model_->meta_->qwen35moe.block_count; ++il) {
        if (!is_full_attention_layer(il)) 
            continue;

        char name[32];
        std::snprintf(name, sizeof(name), "kq_mask.%u", il);
        ggml_tensor* kq_mask = ggml_graph_get_tensor(gf, name);
        if (!kq_mask) 
            continue;  // 如果 KV 缓存为空，掩码可能不存在

        const uint32_t n_kv = static_cast<uint32_t>(kq_mask->ne[0]);
        std::vector<float> mask(n_kv * n_tok);
        // 因果掩码：token 只能关注前面的 token
        for (uint32_t t = 0; t < n_tok; ++t) {
            const uint32_t q_pos = static_cast<uint32_t>(pos) + t;
            for (uint32_t j = 0; j < n_kv; ++j)
                mask[t * n_kv + j] = (j <= q_pos) ? 0.0f : -INFINITY;
        }
        set_mask_data(kq_mask, mask);
    }
}

/**
 * @brief 设置 Decode 阶段的批处理输入数据
 * 
 * @param gf 计算图
 * @param tokens 每个序列当前的 token
 * @param slots 槽位索引数组
 * @param positions 每个序列的当前位置
 * 
 * 设置内容：
 * 1. Token ID 张量
 * 2. 位置 ID 张量（每个序列一个）
 * 3. 共享 KV 掩码
 * 4. Gather indices（用于从稀疏缓存收集 KV）
 */
void Qwen35moeForwardPass::set_batched_inputs(ggml_cgraph* gf, const std::vector<int32_t>& tokens,
    const std::vector<uint32_t>& slots, const std::vector<int32_t>&  positions) {
    const uint32_t n_batch = static_cast<uint32_t>(tokens.size());

    // 1. 设置 Token 张量
    ggml_tensor* tok_t = ggml_graph_get_tensor(gf, "tokens");
    if (!tok_t) throw std::runtime_error("qwen36: 'tokens' tensor missing from graph");
    ggml_backend_tensor_set(tok_t, tokens.data(), 0, n_batch * sizeof(int32_t));

    // 2. 设置位置 ID 张量（每个 batch 槽位一个位置）
    ggml_tensor* pos_t = ggml_graph_get_tensor(gf, "inp_pos");
    if (!pos_t) throw std::runtime_error("qwen36: 'inp_pos' tensor missing from graph");
    ggml_backend_tensor_set(pos_t, positions.data(), 0, n_batch * sizeof(int32_t));

    // 3. 设置共享 KV 掩码 [n_kv_len, 1, 1, n_batch]
    ggml_tensor* kq_mask = ggml_graph_get_tensor(gf, "kq_mask_b");
    if (kq_mask) {
        const uint32_t n_kv = static_cast<uint32_t>(kq_mask->ne[0]);
        std::vector<float> mask(n_kv * n_batch, -INFINITY);
        for (uint32_t b = 0; b < n_batch; ++b) {
            const uint32_t q_pos = static_cast<uint32_t>(positions[b]);
            for (uint32_t j = 0; j <= q_pos && j < n_kv; ++j)
                mask[b * n_kv + j] = 0.0f;  // 允许关注当前位置之前的 token
        }
        set_mask_data(kq_mask, mask);
    }

    // 4. 设置 Gather indices [n_batch * n_kv_len]
    // 用于从稀疏 KV 缓存中收集正确的 KV 对
    ggml_tensor* gi = ggml_graph_get_tensor(gf, "gather_indices");
    if (gi) {
        const uint32_t n_kv   = static_cast<uint32_t>(gi->ne[0]) / n_batch;
        std::vector<int32_t> idx(n_batch * n_kv);
        for (uint32_t b = 0; b < n_batch; ++b) {
            const uint32_t slot = slots[b];
            for (uint32_t j = 0; j < n_kv; ++j)
                idx[b * n_kv + j] = static_cast<int32_t>(slot * n_kv + j);
        }
        ggml_backend_tensor_set(gi, idx.data(), 0, idx.size() * sizeof(int32_t));
    }
}

/**
 * @brief 执行 Prefill 阶段推理
 * 
 * @param tokens 输入 token 序列
 * @param pos 起始位置
 * @param slot_idx 槽位索引
 * @param scheduler 后端调度器
 * @return 输出 logits
 * 
 * 执行流程：
 * 1. 重置调度器
 * 2. 构建计算图
 * 3. 分配计算图内存
 * 4. 设置输入数据
 * 5. 执行计算
 * 6. 更新缓存位置
 * 7. 返回输出 logits
 */
std::vector<float> Qwen35moeForwardPass::run_prefill(const std::vector<int32_t>& tokens, int pos, 
    uint32_t slot_idx, ggml_backend_sched_t scheduler) {
    if (scheduler != nullptr) {
        // 默认：整体路径（子类可以重写以支持 TQ）
        ggml_backend_sched_reset(scheduler);
        ggml_cgraph* gf = build_prefill_graph(tokens, pos, slot_idx);
        if (!ggml_backend_sched_alloc_graph(scheduler, gf)) {
            throw std::runtime_error("Qwen35moeForwardPass::run_prefill: failed to allocate graph");
        }
        set_inputs(gf, tokens, pos);
        ggml_backend_sched_graph_compute(scheduler, gf);
        advance_cache(tokens.size(), slot_idx);
        return get_output_logits(gf);
    } else {
        // 直接执行（不使用调度器）
        auto backend = model_->get_curr_backend();
        auto buf_type = ggml_backend_get_default_buffer_type(backend);
        ggml_gallocr_t allocr_prefill = ggml_gallocr_new(buf_type);
        ggml_cgraph* gf = build_prefill_graph(tokens, pos, slot_idx);
        ggml_gallocr_alloc_graph(allocr_prefill, gf);
        set_inputs(gf, tokens, pos);
        ggml_backend_graph_compute(backend, gf);
        advance_cache(tokens.size(), slot_idx);
        auto result =  get_output_logits(gf);
        if (allocr_prefill) 
            ggml_gallocr_free(allocr_prefill);
        return result;
    }
}

TopKSampleCandidates Qwen35moeForwardPass::run_prefill_topk(const std::vector<int32_t>& tokens, int pos,
    uint32_t slot_idx, ggml_backend_sched_t scheduler) {
    if (sampling_top_k_ <= 0 || sampling_temperature_ <= 0.0f) {
        throw std::runtime_error("Qwen35moeForwardPass::run_prefill_topk: device sampling not configured");
    }
    if (tokens.empty()) {
        throw std::runtime_error("Qwen35moeForwardPass::run_prefill_topk: empty token input");
    }

    if (scheduler != nullptr) {
        ggml_backend_sched_reset(scheduler);
        ggml_cgraph* gf = build_prefill_graph(tokens, pos, slot_idx);
        if (!ggml_backend_sched_alloc_graph(scheduler, gf)) {
            throw std::runtime_error("Qwen35moeForwardPass::run_prefill_topk: failed to allocate graph");
        }
        set_inputs(gf, tokens, pos);
        ggml_backend_sched_graph_compute(scheduler, gf);
        advance_cache(tokens.size(), slot_idx);
        return get_output_topk_candidates(gf, static_cast<uint32_t>(tokens.size() - 1));
    } else {
        auto backend = model_->get_curr_backend();
        auto buf_type = ggml_backend_get_default_buffer_type(backend);
        ggml_gallocr_t allocr_prefill = ggml_gallocr_new(buf_type);
        ggml_cgraph* gf = build_prefill_graph(tokens, pos, slot_idx);
        ggml_gallocr_alloc_graph(allocr_prefill, gf);
        set_inputs(gf, tokens, pos);
        ggml_backend_graph_compute(backend, gf);
        advance_cache(tokens.size(), slot_idx);
        auto result = get_output_topk_candidates(gf, static_cast<uint32_t>(tokens.size() - 1));
        if (allocr_prefill) {
            ggml_gallocr_free(allocr_prefill);
        }
        return result;
    }
}

std::vector<float> Qwen35moeForwardPass::run_decode_cached(int32_t token, int pos,
    uint32_t slot_idx, ggml_backend_sched_t scheduler) {
    if (scheduler == nullptr) {
        std::vector<int32_t> token_vec = { token };
        return run_prefill(token_vec, pos, slot_idx, scheduler);
    }
    if (static_cast<uint32_t>(pos) >= context_len_) {
        throw std::runtime_error("Qwen35moeForwardPass::run_decode_cached: position exceeds context_len_");
    }
    const uint32_t required_kv = static_cast<uint32_t>(pos) + 1;
    if (cached_decode_graph_ == nullptr || cached_decode_slot_ != slot_idx ||
        !cached_decode_graph_allocated_ || required_kv > cached_decode_kv_capacity_) {
        prepare_cached_decode_graph(scheduler, slot_idx, decode_cache_capacity(required_kv, context_len_));
    }

    set_cached_decode_inputs(cached_decode_graph_, token, pos);
    ggml_backend_sched_graph_compute(scheduler, cached_decode_graph_);

    if (kv_cache_) {
        kv_cache_->copy_pos(cached_decode_scratch_pos_, static_cast<uint32_t>(pos), slot_idx);
    }
    advance_cache(1, slot_idx);
    return get_output_logits(cached_decode_graph_);
}

TopKSampleCandidates Qwen35moeForwardPass::run_decode_cached_topk(int32_t token, int pos,
    uint32_t slot_idx, ggml_backend_sched_t scheduler) {
    if (sampling_top_k_ <= 0 || sampling_temperature_ <= 0.0f) {
        throw std::runtime_error("Qwen35moeForwardPass::run_decode_cached_topk: device sampling not configured");
    }
    if (scheduler == nullptr) {
        std::vector<int32_t> token_vec = { token };
        return run_prefill_topk(token_vec, pos, slot_idx, scheduler);
    }
    if (static_cast<uint32_t>(pos) >= context_len_) {
        throw std::runtime_error("Qwen35moeForwardPass::run_decode_cached_topk: position exceeds context_len_");
    }
    const uint32_t required_kv = static_cast<uint32_t>(pos) + 1;
    if (cached_decode_graph_ == nullptr || cached_decode_slot_ != slot_idx ||
        !cached_decode_graph_allocated_ || required_kv > cached_decode_kv_capacity_) {
        prepare_cached_decode_graph(scheduler, slot_idx, decode_cache_capacity(required_kv, context_len_));
    }

    set_cached_decode_inputs(cached_decode_graph_, token, pos);
    ggml_backend_sched_graph_compute(scheduler, cached_decode_graph_);

    if (kv_cache_) {
        kv_cache_->copy_pos(cached_decode_scratch_pos_, static_cast<uint32_t>(pos), slot_idx);
    }
    advance_cache(1, slot_idx);
    return get_output_topk_candidates(cached_decode_graph_, 0);
}

uint32_t Qwen35moeForwardPass::get_cache_pos(uint32_t slot_idx) const {
    uint32_t seq = snapkv_get_seq_pos(slot_idx);
    if (seq > 0) return seq;
    return kv_cache_ ? kv_cache_->get_pos(slot_idx) : 0;
}

ggml_cgraph* Qwen35moeForwardPass::new_graph() {
    return ggml_new_graph_custom(ctx_, FP_GRAPH_SIZE, false);
}

/**
 * @brief Token Embedding 查找
 * 
 * @param gf 计算图
 * @param tokens 输入 token ID 序列
 * @return 嵌入向量张量 [n_embd, n_tokens]
 * 
 * 实现步骤：
 * 1. 创建 token ID 张量
 * 2. 使用 ggml_get_rows 进行嵌入查找（相当于 embedding lookup）
 */
ggml_tensor* Qwen35moeForwardPass::embedding(ggml_cgraph* gf, const std::vector<int32_t>& tokens) {
    const size_t n_tokens = tokens.size();

    // 1. 创建 1D token ID 张量
    struct ggml_tensor* tokens_tensor = ggml_new_tensor_1d(
        ctx_,
        GGML_TYPE_I32,
        n_tokens
    );
    
    ggml_set_input(tokens_tensor);  // 标记为输入张量
    set_tensor_name(tokens_tensor, "tokens");
    ggml_build_forward_expand(gf, tokens_tensor);

    // 2. 使用 ggml_get_rows 进行嵌入查找
    // 从嵌入矩阵中提取对应 token 的嵌入向量
    struct ggml_tensor* token_embedding = model_->get_token_embedding_weight();
    ggml_tensor * cur = ggml_get_rows(
        ctx_,
        token_embedding,
        tokens_tensor
    );

    set_tensor_name(cur, "embed_lookup");
    return cur;
}

/**
 * @brief 构建 MoE（混合专家）FFN 层
 * 
 * 在 Pre-FFN 归一化之后应用，返回 FFN 输出（残差连接之前）
 * 
 * @param ctx ggml 上下文
 * @param gf 计算图
 * @param input 输入张量 [n_embd, n_tokens]
 * @param il 物理层索引
 * @return MoE 层输出 [n_embd, n_tokens]
 * 
 * MoE 层结构：
 * 1. 路由 (Routing)：为每个 token 选择 top-k 专家
 * 2. Expert Dispatch：将 token 分配到选中的专家
 * 3. Weighted Sum：加权汇总专家输出
 * 4. Shared Expert：共享专家（可选）
 */
ggml_tensor* Qwen35moeForwardPass::build_moe_layer(
    ggml_context* ctx,
    ggml_cgraph*  gf,
    ggml_tensor*  input,
    int il
)
{
    // 获取共享专家权重
    struct ggml_tensor* ffn_gate_shexp = model_->get_ffn_gate_shexp_weight(il);
    struct ggml_tensor* ffn_up_shexp = model_->get_ffn_up_shexp_weight(il);
    struct ggml_tensor* ffn_down_shexp = model_->get_ffn_down_shexp_weight(il);
    if (!ffn_gate_shexp || !ffn_up_shexp || !ffn_down_shexp) {
        throw std::runtime_error("moe_layer: has_shared_expert=true but shared expert weights are null");
    }
    
    // 获取门控输入和专家权重
    struct ggml_tensor* ffn_gate_inp = model_->get_ffn_gate_inp_weight(il);
    struct ggml_tensor* ffn_gate_exps = model_->get_ffn_gate_exps_weight(il);
    struct ggml_tensor* ffn_up_exps = model_->get_ffn_up_exps_weight(il);
    struct ggml_tensor* ffn_down_exps = model_->get_ffn_down_exps_weight(il);
    struct ggml_tensor* ffn_gate_inp_shexp = model_->get_ffn_gate_inp_shexp_weight(il);

    auto& m = model_->meta_->qwen35moe;
    // input: [n_embd, n_tokens]
    const int64_t n_embd   = input->ne[0];
    const int64_t n_tokens = input->ne[1];
    const int     n_exp    = m.expert_count;      // 专家数量
    const int     top_k    = m.expert_used_count; // 每个 token 选择的专家数
    const int64_t ffn_dim  = m.expert_feed_forward_length;  // FFN 维度

    // ── 1. 路由 logits 和 Top-k 门控 ────────────────────────────────────────
    // 计算每个 token 到每个专家的路由分数
    // logits: [n_experts, n_tokens]
    ggml_tensor* logits = ggml_mul_mat(ctx, ffn_gate_inp, input);
    set_tensor_name(logits, "moe_logits", il);

    // 获取 top-k 专家的索引
    // sorted_idx: [n_experts, n_tokens] I32 — 按降序排序
    ggml_tensor* sorted_idx = ggml_argsort(ctx, logits, GGML_SORT_ORDER_DESC);
    // expert_idx: [top_k, n_tokens] I32 — 只取前 top-k
    ggml_tensor* expert_idx = ggml_view_2d(ctx, sorted_idx,
        top_k, n_tokens,
        sorted_idx->nb[1],
        0
    );
    set_tensor_name(expert_idx, "moe_idx", il);

    // 收集 top-k 专家对应的实际 logit 值
    // 为了使用 ggml_get_rows，将 logits 重塑为 [1, n_experts, n_tokens]
    ggml_tensor* logits_3d = ggml_reshape_3d(ctx, logits, 1, n_exp, n_tokens);
    ggml_tensor* expert_logits = ggml_get_rows(ctx, logits_3d, expert_idx);
    // expert_logits: [1, top_k, n_tokens]，重塑为 2D 进行 softmax
    expert_logits = ggml_reshape_2d(ctx, expert_logits, top_k, n_tokens);

    // 对每个 token 的 top-k 权重应用 softmax 归一化
    ggml_tensor* expert_weights = ggml_soft_max(ctx, expert_logits);
    set_tensor_name(expert_weights, "moe_weights", il);

    // ── 2. 通过 ggml_mul_mat_id 分发到专家 ──────────────────────────────────
    //
    // ggml_mul_mat_id: 批处理矩阵乘法，每个 token 使用不同的专家权重矩阵
    // 签名: (W [in, out, n_exp], x [in, n_tok], idx [top_k, n_tok])
    // 返回: [out, top_k, n_tok]

    // 将输入重塑为 [in, 1, n_tok] 以适配 ggml_mul_mat_id
    ggml_tensor* input_3d = ggml_reshape_3d(ctx, input, n_embd, 1, n_tokens);

    // Gate 投影: 每个 token × 其 top_k 专家的 gate 权重
    ggml_tensor* exp_gate_out = ggml_mul_mat_id(ctx, ffn_gate_exps, input_3d, expert_idx);
    set_tensor_name(exp_gate_out, "moe_exp_gate", il);
    // [ffn_dim, top_k, n_tokens]

    // Up 投影
    ggml_tensor* exp_up_out = ggml_mul_mat_id(ctx, ffn_up_exps, input_3d, expert_idx);
    set_tensor_name(exp_up_out, "moe_exp_up", il);

    // SwiGLU 激活: silu(gate) * up
    ggml_tensor* exp_act = ggml_mul(ctx, ggml_silu(ctx, exp_gate_out), exp_up_out);
    set_tensor_name(exp_act, "moe_exp_act", il);
    // [ffn_dim, top_k, n_tokens]

    // Down 投影
    // exp_act: [ffn_dim, top_k, n_tokens] — 需要重塑以适配 ggml_mul_mat_id
    ggml_tensor* exp_down_out = ggml_mul_mat_id(ctx, ffn_down_exps, exp_act, expert_idx);
    set_tensor_name(exp_down_out, "moe_exp_down", il);
    // [n_embd, top_k, n_tokens]

    // ── 3. 专家输出的加权求和 ───────────────────────────────────────────────

    // expert_weights: [top_k, n_tokens] — 重塑为 [1, top_k, n_tokens] 用于广播
    ggml_tensor* w_expanded = ggml_reshape_3d(ctx, expert_weights, 1, top_k, n_tokens);

    // exp_down_out: [n_embd, top_k, n_tokens]
    // 将每个专家输出乘以其路由权重
    ggml_tensor* weighted = ggml_mul(ctx, exp_down_out, w_expanded);
    set_tensor_name(weighted, "moe_weighted", il);

    // 在 top_k 维度上求和 → [n_embd, n_tokens]
    ggml_tensor* routed_out = ggml_view_2d(ctx, weighted, n_embd, n_tokens, weighted->nb[2], 0);
    for (int k = 1; k < top_k; ++k) {
        ggml_tensor* expert_k = ggml_view_2d(ctx, weighted,
            n_embd, n_tokens, weighted->nb[2],
            static_cast<size_t>(k) * weighted->nb[1]
        );
        routed_out = ggml_add(ctx, routed_out, expert_k);
    }
    set_tensor_name(routed_out, "moe_routed_out", il);

    // ── 4. 共享专家（可选）───────────────────────────────────────────────────
    // 共享专家: 对所有 token 应用标准 SwiGLU FFN
    ggml_tensor* sh_gate_out = ggml_mul_mat(ctx, ffn_gate_shexp, input);
    ggml_tensor* sh_up_out   = ggml_mul_mat(ctx, ffn_up_shexp,   input);
    ggml_tensor* sh_act      = ggml_mul(ctx, ggml_silu(ctx, sh_gate_out), sh_up_out);
    ggml_tensor* sh_down_out = ggml_mul_mat(ctx, ffn_down_shexp, sh_act);
    set_tensor_name(sh_down_out, "moe_shared_out", il);

    // 每个 token 的标量门控: ffn_gate_inp_shexp 是 [n_embd]
    // mul_mat 得到 [1, n_tokens], sigmoid 映射到 (0,1)
    ggml_tensor* sh_gate_logit = ggml_mul_mat(ctx, ffn_gate_inp_shexp, input);
    ggml_tensor* sh_gate       = ggml_sigmoid(ctx, sh_gate_logit);
    ggml_tensor* sh_contribution = ggml_mul(ctx, sh_down_out, sh_gate);
    set_tensor_name(sh_contribution, "moe_shared_contrib", il);

    // 合并专家输出和共享专家输出
    ggml_tensor* combined = ggml_add(ctx, routed_out, sh_contribution);
    set_tensor_name(combined, "moe_combined", il);

    return combined;
}

/**
 * @brief 构建 RMS 归一化层
 * 
 * RMSNorm 公式: output = weight * (input / sqrt(mean(input^2) + eps))
 * 
 * @param ctx ggml 上下文
 * @param cur 输入张量
 * @param weight 缩放权重
 * @param eps epsilon（防止除零）
 * @param il 层索引
 * @return 归一化后的张量
 */
ggml_tensor* Qwen35moeForwardPass::build_rms_norm(
    ggml_context* ctx,
    ggml_tensor*  cur,
    ggml_tensor*  weight,
    float         eps,
    int           il
)
{
    cur = ggml_rms_norm(ctx, cur, eps);
    set_tensor_name(cur, "cur_rms_normed", il);
    cur = ggml_mul(ctx, cur, weight);  // 乘以可学习的缩放权重
    return cur;
}

/**
 * @brief 构建 SwiGLU FFN 层
 * 
 * SwiGLU 公式: down @ (silu(gate @ x) * (up @ x))
 * 
 * @param ctx ggml 上下文
 * @param gf 计算图（未使用，保持 API 一致性）
 * @param cur 输入张量
 * @param gate gate 权重矩阵
 * @param up up 权重矩阵
 * @param down down 权重矩阵
 * @param il 层索引
 * @return FFN 输出
 */
ggml_tensor* Qwen35moeForwardPass::build_ffn_swiglu(
    ggml_context* ctx,
    ggml_cgraph*  /*gf*/,
    ggml_tensor*  cur,
    ggml_tensor*  gate,
    ggml_tensor*  up,
    ggml_tensor*  down,
    int           il
)
{
    char name[128];

    // Up 投影
    ggml_tensor* tmp = ggml_mul_mat(ctx, up, cur);
    snprintf(name, sizeof(name), "ffn_up.%d", il);
    set_tensor_name(tmp, name);

    // Gate 投影
    cur = ggml_mul_mat(ctx, gate, cur);
    snprintf(name, sizeof(name), "ffn_gate.%d", il);
    set_tensor_name(cur, name);

    // SwiGLU 激活: silu(gate) * up
    cur = ggml_swiglu_split(ctx, cur, tmp);
    snprintf(name, sizeof(name), "ffn_swiglu.%d", il);
    set_tensor_name(cur, name);

    // Down 投影
    cur = ggml_mul_mat(ctx, down, cur);
    return cur;
}

/**
 * @brief 构建归一化层（封装函数）
 * 
 * 调用 build_rms_norm，使用模型配置的 epsilon
 * 
 * @param gf 计算图
 * @param cur 输入张量
 * @param mw 权重张量
 * @param il 层索引
 * @return 归一化后的张量
 */
ggml_tensor* Qwen35moeForwardPass::build_norm(
    ggml_cgraph* gf,
    ggml_tensor* cur,
    ggml_tensor* mw,
    int il
)
{
    return build_rms_norm(ctx_, cur, mw, model_->meta_->qwen35moe.layer_norm_rms_epsilon, il);
}

/**
 * @brief 构建多头注意力 (MHA) 层
 * 
 * 标准多头注意力计算：Attention(Q,K,V) = softmax(Q*K^T / sqrt(d_k)) * V
 * 
 * @param ctx ggml 上下文
 * @param gf 计算图（未直接使用）
 * @param q Query 张量
 * @param k Key 张量
 * @param v Value 张量
 * @param kq_mask 注意力掩码
 * @param sinks sink token（未使用）
 * @param kq_scale 缩放因子（1/sqrt(d_k)）
 * @param pos 位置（未使用）
 * @param il 层索引
 * @return 注意力输出
 */
ggml_tensor* Qwen35moeForwardPass::build_attn_mha(
    ggml_context* ctx,
    ggml_cgraph*  gf,
    ggml_tensor*  q,
    ggml_tensor*  k,
    ggml_tensor*  v,
    ggml_tensor*  kq_mask,
    ggml_tensor*  sinks,
    float         kq_scale,
    uint32_t      pos,
    int           il
)
{
    (void)gf; (void)pos; // 保持 API 对称性，未直接使用

    const bool v_trans = v->nb[1] > v->nb[2];
    (void)v_trans;

    const auto n_stream = k->ne[3];

    // 重塑 Q 张量
    q = ggml_reshape_4d(ctx, q, q->ne[0], q->ne[1], q->ne[2]/n_stream, n_stream);
    set_tensor_name(q, "q_reshaped", il);

    // 调整张量维度顺序以适配注意力计算
    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    set_tensor_name(q, "q_permuted", il);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    set_tensor_name(k, "k_permuted", il);

    if (use_flash_attention_ && kq_mask && kq_mask->type == GGML_TYPE_F16) {
        ggml_tensor* v_fa = ggml_permute(ctx, v, 0, 2, 1, 3);
        set_tensor_name(v_fa, "v_flash", il);

        ggml_tensor* kqv = ggml_flash_attn_ext(ctx, q, k, v_fa, kq_mask, kq_scale, 0.0f, 0.0f);
        set_tensor_name(kqv, "kqv_flash", il);
        if (sinks) {
            ggml_flash_attn_ext_add_sinks(kqv, sinks);
        }

        ggml_tensor* cur = ggml_cont_2d(ctx, kqv, kqv->ne[0] * kqv->ne[1], kqv->ne[2] * kqv->ne[3]);
        set_tensor_name(cur, "attn_flash_recombined", il);
        return cur;
    }

    v = ggml_permute(ctx, v, 1, 2, 0, 3);
    set_tensor_name(v, "v_permuted", il);
    v = ggml_cont(ctx, v);  // 确保连续内存
    set_tensor_name(v, "v_cont", il);

    ggml_tensor* cur;
    {
        // Q * K^T
        ggml_tensor* kq = ggml_mul_mat(ctx, k, q);
        set_tensor_name(kq, "kq", il);

        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);  // 使用 F32 精度

        // softmax(Q*K^T / scale + mask)
        kq = ggml_soft_max_ext(ctx, kq, kq_mask, kq_scale, 0);
        set_tensor_name(kq, "kq_soft", il);

        ggml_soft_max_add_sinks(kq, sinks);  // 添加 sink token

        // attention * V
        ggml_tensor* kqv = ggml_mul_mat(ctx, v, kq);
        set_tensor_name(kqv, "kqv", il);

        // 调整输出维度顺序
        cur = ggml_permute(ctx, kqv, 0, 2, 1, 3);
        set_tensor_name(cur, "kqv_permuted", il);

        // 合并为 2D 张量
        cur = ggml_cont_2d(ctx, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
        set_tensor_name(cur, "attn_recombined", il);
    }

    return cur;
}

/**
 * @brief 构建 Qwen3.5/Qwen3.6 的门控注意力层
 * 
 * Gated Attention 特点：
 * 1. 联合 Q+Gate 投影
 * 2. Q/K 独立 RMS 归一化
 * 3. Partial RoPE 位置编码
 * 4. 输出端 Sigmoid 门控
 * 
 * @param ctx ggml 上下文
 * @param gf 计算图
 * @param kv_cache KV 缓存
 * @param cur 输入张量
 * @param inp_pos 位置张量
 * @param kv_cache_layer KV 缓存层索引
 * @param n_tokens token 数量
 * @param slot_idx 槽位索引
 * @param il 层索引
 * @param w_q Q 投影权重
 * @param w_q_norm Q 归一化权重
 * @param w_k K 投影权重
 * @param w_k_norm K 归一化权重
 * @param w_v V 投影权重
 * @param w_out 输出投影权重
 * @param n_embd_head 每头维度
 * @param n_head Q 头数
 * @param n_head_kv KV 头数
 * @param n_rot RoPE 旋转维度
 * @param freq_base RoPE 频率基数
 * @param context_length 上下文长度
 * @param rms_norm_eps RMS 归一化 epsilon
 * @return 注意力输出
 */
ggml_tensor* Qwen35moeForwardPass::build_gated_attention(
    ggml_context*    ctx,
    ggml_cgraph*     gf,
    simple_kv_cache* kv_cache,
    ggml_tensor*     cur,
    ggml_tensor*     inp_pos,
    int              kv_cache_layer,
    uint32_t         n_tokens,
    uint32_t         slot_idx,
    int              il,
    ggml_tensor*     w_q,
    ggml_tensor*     w_q_norm,
    ggml_tensor*     w_k,
    ggml_tensor*     w_k_norm,
    ggml_tensor*     w_v,
    ggml_tensor*     w_out,
    int              n_embd_head,
    int              n_head,
    int              n_head_kv,
    int              n_rot,
    float            freq_base,
    int              context_length,
    float            rms_norm_eps
)
{
    // A. 联合 Q+Gate 投影
    // 输出: [(n_embd_head*2)*n_head, n_tokens]
    ggml_tensor* Qcur_full = ggml_mul_mat(ctx, w_q, cur);
    set_tensor_name(Qcur_full, "Qcur_full", il);

    // B. 通过跨步视图提取 Q
    // Q 和 Gate 交替存储，Q 在偶数位置
    ggml_tensor* Qcur = ggml_view_3d(ctx, Qcur_full,
        n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    set_tensor_name(Qcur, "Qcur", il);

    // Q RMS 归一化
    Qcur = build_rms_norm(ctx, Qcur, w_q_norm, rms_norm_eps, il);
    set_tensor_name(Qcur, "Qcur_normed", il);

    // C. K 和 V 投影
    ggml_tensor* Kcur = ggml_mul_mat(ctx, w_k, cur);
    ggml_tensor* Vcur = ggml_mul_mat(ctx, w_v, cur);

    Kcur = ggml_reshape_3d(ctx, Kcur, n_embd_head, n_head_kv, n_tokens);
    Vcur = ggml_reshape_3d(ctx, Vcur, n_embd_head, n_head_kv, n_tokens);

    // K RMS 归一化
    Kcur = build_rms_norm(ctx, Kcur, w_k_norm, rms_norm_eps, il);
    set_tensor_name(Kcur, "Kcur_normed", il);

    // D. 提取 Gate
    // Gate 在奇数位置（偏移 n_embd_head）
    ggml_tensor* gate = ggml_view_3d(ctx, Qcur_full,
        n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx, gate, n_embd_head * n_head, n_tokens);
    set_tensor_name(gate, "gate", il);

    // E. Partial RoPE（部分旋转位置编码）
    Qcur = ggml_rope_ext(ctx, Qcur, inp_pos, nullptr,
        n_rot, GGML_ROPE_TYPE_NEOX,
        context_length, freq_base,
        1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    Kcur = ggml_rope_ext(ctx, Kcur, inp_pos, nullptr,
        n_rot, GGML_ROPE_TYPE_NEOX,
        context_length, freq_base,
        1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

    // F. KV 缓存写入 + 完整历史读取
    const float    kq_scale  = 1.0f / sqrtf(float(n_embd_head));
    const uint32_t cache_pos = kv_cache->get_pos(slot_idx);
    const uint32_t n_kv      = cache_pos + n_tokens;

    // 写入 KV 缓存
    ggml_build_forward_expand(gf, kv_cache->cpy_k(ctx, Kcur, kv_cache_layer, slot_idx));
    ggml_build_forward_expand(gf, kv_cache->cpy_v(ctx, Vcur, kv_cache_layer, slot_idx));

    // 读取完整的 KV 历史
    ggml_tensor* k_full = kv_cache->get_k(ctx, kv_cache_layer, n_kv, slot_idx);
    ggml_tensor* v_full = kv_cache->get_v(ctx, kv_cache_layer, n_kv, slot_idx);

    ggml_tensor* k_view = ggml_view_3d(ctx, k_full,
        n_embd_head, n_head_kv, n_kv,
        n_embd_head * k_full->nb[0], k_full->nb[1], 0);
    ggml_tensor* v_view = ggml_view_3d(ctx, v_full,
        n_embd_head, n_head_kv, n_kv,
        n_embd_head * v_full->nb[0], v_full->nb[1], 0);

    // 创建因果掩码
    ggml_tensor* kq_mask = ggml_new_tensor_2d(ctx, use_flash_attention_ ? GGML_TYPE_F16 : GGML_TYPE_F32, n_kv, n_tokens);
    set_tensor_name(kq_mask, "kq_mask", il);
    ggml_build_forward_expand(gf, kq_mask);

    // 计算多头注意力
    cur = build_attn_mha(ctx, gf, Qcur, k_view, v_view, kq_mask, nullptr, kq_scale, cache_pos, il);

    // G. Sigmoid 门控
    cur = ggml_mul(ctx, cur, ggml_sigmoid(ctx, gate));
    set_tensor_name(cur, "attn_gated", il);

    // H. 输出投影
    cur = ggml_mul_mat(ctx, w_out, cur);
    set_tensor_name(cur, "attn_output", il);

    return cur;
}

/**
 * @brief 构建批处理版本的门控注意力层（Decode 阶段）
 * 
 * 与 build_gated_attention 类似，但支持多序列并行处理
 * 
 * @param ctx ggml 上下文
 * @param gf 计算图
 * @param kv_cache KV 缓存
 * @param cur 输入张量
 * @param inp_pos 位置张量
 * @param kq_mask 注意力掩码
 * @param gather_indices KV gather 索引
 * @param kv_cache_layer KV 缓存层索引
 * @param slots 槽位索引数组
 * @param positions 位置数组
 * @param il 层索引
 * @param w_q Q 投影权重
 * @param w_q_norm Q 归一化权重
 * @param w_k K 投影权重
 * @param w_k_norm K 归一化权重
 * @param w_v V 投影权重
 * @param w_out 输出投影权重
 * @param n_embd_head 每头维度
 * @param n_head Q 头数
 * @param n_head_kv KV 头数
 * @param n_rot RoPE 旋转维度
 * @param freq_base RoPE 频率基数
 * @param context_length 上下文长度
 * @param rms_norm_eps RMS 归一化 epsilon
 * @return 注意力输出
 */
ggml_tensor* Qwen35moeForwardPass::build_gated_batched_attention(
    ggml_context*                ctx,
    ggml_cgraph*                 gf,
    simple_kv_cache*             kv_cache,
    ggml_tensor*                 cur,
    ggml_tensor*                 inp_pos,
    ggml_tensor*                 kq_mask,
    ggml_tensor*                 gather_indices,
    int                          kv_cache_layer,
    const std::vector<uint32_t>& slots,
    const std::vector<int32_t>&  positions,
    int                          il,
    ggml_tensor*                 w_q,
    ggml_tensor*                 w_q_norm,
    ggml_tensor*                 w_k,
    ggml_tensor*                 w_k_norm,
    ggml_tensor*                 w_v,
    ggml_tensor*                 w_out,
    int                          n_embd_head,
    int                          n_head,
    int                          n_head_kv,
    int                          n_rot,
    float                        freq_base,
    int                          context_length,
    float                        rms_norm_eps
)
{
    const size_t n_batch = slots.size();

    // A. 联合 Q+Gate 投影 → [(n_embd_head*2)*n_head, n_batch]
    ggml_tensor* Qcur_full = ggml_mul_mat(ctx, w_q, cur);

    // B. 通过跨步视图提取 Q → [n_embd_head, n_head, n_batch]
    ggml_tensor* Qcur = ggml_view_3d(ctx, Qcur_full,
        n_embd_head, n_head, n_batch,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0
    );

    Qcur = build_rms_norm(ctx, Qcur, w_q_norm, rms_norm_eps, il);

    // C. K 和 V 投影
    ggml_tensor* Kcur = ggml_mul_mat(ctx, w_k, cur);
    ggml_tensor* Vcur = ggml_mul_mat(ctx, w_v, cur);

    Kcur = ggml_reshape_3d(ctx, Kcur, n_embd_head, n_head_kv, n_batch);
    Vcur = ggml_reshape_3d(ctx, Vcur, n_embd_head, n_head_kv, n_batch);

    Kcur = build_rms_norm(ctx, Kcur, w_k_norm, rms_norm_eps, il);

    // D. 提取 Gate → [n_embd_head*n_head, n_batch]
    ggml_tensor* gate = ggml_view_3d(ctx, Qcur_full,
        n_embd_head, n_head, n_batch,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head
    );
    gate = ggml_cont_2d(ctx, gate, n_embd_head * n_head, n_batch);

    // E. Partial RoPE
    Qcur = ggml_rope_ext(ctx, Qcur, inp_pos, nullptr,
        n_rot, GGML_ROPE_TYPE_NEOX,
        context_length, freq_base,
        1.0f, 0.0f, 1.0f, 32.0f, 1.0f
    );
    Kcur = ggml_rope_ext(ctx, Kcur, inp_pos, nullptr,
        n_rot, GGML_ROPE_TYPE_NEOX,
        context_length, freq_base,
        1.0f, 0.0f, 1.0f, 32.0f, 1.0f
    );

    // F. 逐槽位写入 KV 缓存
    const int n_embd_k = n_head_kv * n_embd_head;
    const int n_embd_v = n_head_kv * n_embd_head;

    ggml_tensor* k_storage_fmt = ggml_reshape_3d(ctx, Kcur, n_embd_k, 1, n_batch);
    ggml_tensor* v_storage_fmt = ggml_reshape_3d(ctx, Vcur, n_embd_v, 1, n_batch);

    for (size_t b = 0; b < n_batch; ++b) {
        ggml_tensor* k_slice = ggml_view_2d(ctx, k_storage_fmt, n_embd_k, 1, k_storage_fmt->nb[1], b * k_storage_fmt->nb[2]);
        ggml_tensor* v_slice = ggml_view_2d(ctx, v_storage_fmt, n_embd_v, 1, v_storage_fmt->nb[1], b * v_storage_fmt->nb[2]);

        ggml_build_forward_expand(gf, kv_cache->cpy_k(ctx, k_slice, kv_cache_layer, slots[b]));
        ggml_build_forward_expand(gf, kv_cache->cpy_v(ctx, v_slice, kv_cache_layer, slots[b]));
    }

    // G. Gather 所有槽位的 KV
    uint32_t max_pos = 0;
    for (int32_t p : positions) {
        if (p > (int32_t)max_pos) 
            max_pos = (uint32_t)p;
    }
    uint32_t n_kv_len = max_pos + 1;
    ggml_tensor* k_gathered = kv_cache->gather_k(ctx, gf, kv_cache_layer, gather_indices, n_batch, n_kv_len);
    ggml_tensor* v_gathered = kv_cache->gather_v(ctx, gf, kv_cache_layer, gather_indices, n_batch, n_kv_len);

    ggml_tensor* k_view = ggml_view_4d(ctx, k_gathered,
        n_embd_head, n_head_kv, n_kv_len, n_batch,
        n_embd_head * k_gathered->nb[0],
        k_gathered->nb[1],
        k_gathered->nb[2], 0
    );
    ggml_tensor* v_view = ggml_view_4d(ctx, v_gathered,
        n_embd_head, n_head_kv, n_kv_len, n_batch,
        n_embd_head * v_gathered->nb[0],
        v_gathered->nb[1],
        v_gathered->nb[2], 0
    );

    // H. 注意力计算
    const float kq_scale = 1.0f / sqrtf(float(n_embd_head));
    cur = build_attn_mha(ctx, gf, Qcur, k_view, v_view, kq_mask, nullptr, kq_scale, 0, il);

    // I. Sigmoid 门控
    cur = ggml_mul(ctx, cur, ggml_sigmoid(ctx, gate));

    // J. 输出投影
    cur = ggml_mul_mat(ctx, w_out, cur);

    return cur;
}

/**
 * @brief 构建 DeltaNet 层（核心实现）
 * 
 * DeltaNet 是 Qwen3.5 引入的高效序列建模层，替代传统 Transformer 中的多头注意力机制。
 * 它结合了以下核心组件：
 * 
 * 1. **因果卷积 (Causal Conv1d)**：捕捉局部序列依赖，类似于注意力的窗口机制
 * 2. **门控 Delta 规则 (Gated Delta Rule)**：一种高效的状态更新机制
 * 3. **循环状态 (Recurrent State)**：维护长期依赖的记忆
 * 
 * DeltaNet 的优势：
 * - 时间复杂度 O(n)，远低于标准注意力的 O(n²)
 * - 内存效率更高，无需存储完整的 KV 矩阵
 * - 适合长序列建模
 * 
 * @param ctx ggml 上下文
 * @param gf 计算图
 * @param cur 输入张量 [n_embd, n_tokens]
 * @param dn_state DeltaNet 状态（包含卷积状态和循环状态）
 * @param dn_idx DeltaNet 层索引（在所有 DeltaNet 层中的位置）
 * @param slot_idx 槽位索引（用于多序列并行）
 * @param n_tokens 当前批次的 token 数量
 * @param w_qkv QKV 混合投影权重 [n_embd, conv_channels]
 * @param w_gate 输出门控权重 [n_embd, d_inner]
 * @param w_beta beta 门控权重 [n_embd, num_v_heads]
 * @param w_a alpha/decay 权重 [n_embd, num_v_heads]
 * @param w_dt_bias dt bias 权重 [num_v_heads]
 * @param w_a_log A_log 权重 [num_v_heads]（用于状态衰减）
 * @param w_conv 卷积核权重
 * @param w_norm RMS 归一化权重 [head_v_dim]
 * @param w_out 输出投影权重 [n_embd, d_inner]
 * @param n_embd 嵌入维度
 * @param d_inner 内部维度 (head_v_dim * num_v_heads)
 * @param head_k_dim K 头维度
 * @param num_k_heads K 头数量
 * @param num_v_heads V 头数量
 * @param head_v_dim V 头维度
 * @param conv_channels 卷积通道数
 * @param conv_kernel 卷积核大小
 * @param rms_norm_eps RMS 归一化 epsilon
 * @param il 物理层索引（用于命名）
 * @return DeltaNet 输出 [n_embd, n_tokens]
 */
ggml_tensor* Qwen35moeForwardPass::build_deltanet_layer(
    ggml_context*   ctx,
    ggml_cgraph*    gf,
    ggml_tensor*    cur,
    DeltaNetState*  dn_state,
    uint32_t        dn_idx,
    uint32_t        slot_idx,
    uint32_t        n_tokens,
    ggml_tensor*    w_qkv,
    ggml_tensor*    w_gate,
    ggml_tensor*    w_beta,
    ggml_tensor*    w_a,
    ggml_tensor*    w_dt_bias,
    ggml_tensor*    w_a_log,
    ggml_tensor*    w_conv,
    ggml_tensor*    w_norm,
    ggml_tensor*    w_out,
    int             n_embd,
    int             d_inner,
    int             head_k_dim,
    int             num_k_heads,
    int             num_v_heads,
    int             head_v_dim,
    int             conv_channels,
    int             conv_kernel,
    float           rms_norm_eps,
    int             il)
{
    // 转换为 int64_t 类型以避免溢出
    const int64_t n_seq_tokens = static_cast<int64_t>(n_tokens);
    const int64_t n_seqs = 1;  // Prefill 阶段处理单个序列

    // =========================================================================
    // 阶段 1: 输入投影 (Input Projections)
    // =========================================================================
    // QKV 混合投影: 将输入嵌入映射到卷积通道空间
    // 输入: cur [n_embd, n_tokens]
    // 权重: w_qkv [n_embd, conv_channels]
    // 输出: qkv_mixed [conv_channels, n_tokens]
    ggml_tensor* qkv_mixed = ggml_mul_mat(ctx, w_qkv, cur);
    qkv_mixed = ggml_reshape_3d(ctx, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    set_tensor_name(qkv_mixed, "dn_qkv", il);

    // Z 门控投影: 用于最后的门控归一化，控制信息流动
    // 输入: cur [n_embd, n_tokens]
    // 权重: w_gate [n_embd, d_inner]
    // 输出: z [d_inner, n_tokens]
    ggml_tensor* z = ggml_mul_mat(ctx, w_gate, cur);
    set_tensor_name(z, "dn_z", il);

    // =========================================================================
    // 阶段 2: Beta 和 Decay Gate 投影
    // =========================================================================
    // Beta 门控: 控制新信息的接受程度（类似注意力中的权重）
    // 经过 sigmoid 激活后取值范围为 (0, 1)
    // 输入: cur [n_embd, n_tokens]
    // 权重: w_beta [n_embd, num_v_heads]
    // 输出: beta [1, num_v_heads, n_tokens, 1]
    ggml_tensor* beta = ggml_mul_mat(ctx, w_beta, cur);
    beta = ggml_reshape_4d(ctx, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    beta = ggml_sigmoid(ctx, beta);
    set_tensor_name(beta, "dn_beta", il);

    // Decay Gate: 控制循环状态的衰减速率，决定历史信息的遗忘程度
    // 公式: decay = softplus(alpha @ x + dt_bias) * A_log
    // softplus 确保输出非负，A_log 是可学习的对数衰减因子
    // 输入: cur [n_embd, n_tokens]
    // 权重: w_a [n_embd, num_v_heads]
    // 输出: decay_gate [1, num_v_heads, n_tokens, 1]
    ggml_tensor* alpha = ggml_mul_mat(ctx, w_a, cur);
    alpha = ggml_reshape_3d(ctx, alpha, num_v_heads, n_seq_tokens, n_seqs);
    ggml_tensor* alpha_biased = ggml_add(ctx, alpha, w_dt_bias);  // 添加时间步偏置
    ggml_tensor* alpha_sp     = ggml_softplus(ctx, alpha_biased);  // softplus 激活
    ggml_tensor* decay_gate   = ggml_mul(ctx, alpha_sp, w_a_log);  // 乘以对数衰减因子
    decay_gate = ggml_reshape_4d(ctx, decay_gate, 1, num_v_heads, n_seq_tokens, n_seqs);
    set_tensor_name(decay_gate, "dn_decay", il);

    // =========================================================================
    // 阶段 3: 因果卷积 (Causal Conv1d)
    // =========================================================================
    // 获取该层的卷积状态张量（存储所有槽位的卷积状态）
    ggml_tensor* conv_all = dn_state->conv_tensor(dn_idx);
    const int64_t conv_state_elems = static_cast<int64_t>(conv_kernel - 1) * conv_channels;

    // 提取当前槽位的滑动窗口状态
    // 卷积状态存储了前 (conv_kernel - 1) 个 token 的信息，用于构建因果卷积窗口
    ggml_tensor* conv_states = ggml_view_1d(ctx, conv_all, conv_state_elems, static_cast<size_t>(slot_idx) * conv_all->nb[1]);
    conv_states = ggml_reshape_3d(ctx, conv_states, conv_kernel - 1, conv_channels, n_seqs);
    set_tensor_name(conv_states, "dn_conv_st", il);

    // 沿时间轴拼接历史状态和新的 QKV token
    // conv_input: [conv_kernel, conv_channels, n_seqs]
    ggml_tensor* qkv_t = ggml_transpose(ctx, qkv_mixed);
    ggml_tensor* conv_input = ggml_concat(ctx, conv_states, qkv_t, 0);
    set_tensor_name(conv_input, "dn_conv_in", il);

    // 更新卷积状态：保留最后 (conv_kernel - 1) 个 token
    // 这是实现因果性的关键：只保留最近的历史，防止信息泄露
    ggml_tensor* last_conv = ggml_view_3d(ctx, conv_input,
        conv_kernel - 1, conv_channels, n_seqs,
        conv_input->nb[1], conv_input->nb[2],
        static_cast<size_t>(conv_input->ne[0] - (conv_kernel - 1)) * ggml_element_size(conv_input)
    );
    ggml_tensor* conv_dst = ggml_view_1d(ctx, conv_all, conv_state_elems, static_cast<size_t>(slot_idx) * conv_all->nb[1]);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, last_conv, conv_dst));

    // 执行深度卷积 + SiLU 激活
    // 深度卷积：每个通道独立卷积，参数共享，计算高效
    ggml_tensor* conv_out = ggml_ssm_conv(ctx, conv_input, w_conv);
    conv_out = ggml_silu(ctx, conv_out);
    set_tensor_name(conv_out, "dn_conv_out", il);

    // =========================================================================
    // 阶段 4: 拆分卷积输出为 Q, K, V
    // =========================================================================
    // QKV 在卷积输出中是连续存储的，需要通过视图拆分
    // 布局: [Q | K | V] = [head_k_dim*num_k_heads | head_k_dim*num_k_heads | head_v_dim*num_v_heads]
    const int64_t qkv_dim  = static_cast<int64_t>(head_k_dim) * num_k_heads * 2 + static_cast<int64_t>(head_v_dim) * num_v_heads;
    const int64_t nb1_qkv  = ggml_row_size(conv_out->type, qkv_dim);

    ggml_tensor* q_conv = ggml_view_4d(ctx, conv_out,
        head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
        ggml_row_size(conv_out->type, head_k_dim),
        nb1_qkv, nb1_qkv * n_seq_tokens, 0
    );

    ggml_tensor* k_conv = ggml_view_4d(ctx, conv_out,
        head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
        ggml_row_size(conv_out->type, head_k_dim),
        nb1_qkv, nb1_qkv * n_seq_tokens,
        static_cast<size_t>(head_k_dim) * num_k_heads * ggml_element_size(conv_out)
    );

    ggml_tensor* v_conv = ggml_view_4d(ctx, conv_out,
        head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
        ggml_row_size(conv_out->type, head_v_dim),
        nb1_qkv, nb1_qkv * n_seq_tokens,
        ggml_row_size(conv_out->type, 2LL * head_k_dim * num_k_heads)
    );

    // =========================================================================
    // 阶段 5: Q 和 K 的 L2 归一化
    // =========================================================================
    // L2 归一化使 Q 和 K 的范数为 1，有助于稳定训练和推理
    // 避免因范数差异导致的梯度不稳定
    q_conv = ggml_l2_norm(ctx, q_conv, rms_norm_eps);
    k_conv = ggml_l2_norm(ctx, k_conv, rms_norm_eps);

    // GQA 风格的头部匹配：如果 K 头数不等于 V 头数，重复 K/Q 头
    // 例如: num_k_heads=16, num_v_heads=32，则每个 K 头对应 2 个 V 头
    // 这样可以减少计算量同时保持表达能力
    if (num_k_heads != num_v_heads) {
        q_conv = ggml_repeat_4d(ctx, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    // =========================================================================
    // 阶段 6: 门控 Delta Net 循环（核心操作）
    // =========================================================================
    // 获取循环状态张量（存储所有槽位的循环状态）
    ggml_tensor* rec_all = dn_state->recurrent_tensor(dn_idx);
    const int64_t rec_slot_floats = static_cast<int64_t>(head_v_dim) * head_k_dim * num_v_heads;

    ggml_tensor* S = ggml_view_1d(ctx, rec_all, rec_slot_floats, static_cast<size_t>(slot_idx) * rec_all->nb[1]);
    S = ggml_reshape_4d(ctx, S, head_v_dim, head_k_dim, num_v_heads, n_seqs);
    set_tensor_name(S, "dn_state_in", il);

    // 调用融合的门控 Delta Net 操作
    // Delta 规则: S_{t+1} = (1 - decay) * S_t + beta * outer(K, V)
    // 这是 DeltaNet 的核心：将新的 K-V 对加权加入状态，同时按 decay 因子衰减旧状态
    // 输出打包了两部分内容：
    //   1. per-token output: [head_v_dim, num_v_heads, n_tokens, n_seqs]
    //   2. final state:      [head_v_dim, head_k_dim, num_v_heads, n_seqs]
    ggml_tensor* result = ggml_gated_delta_net(ctx, q_conv, k_conv, v_conv, decay_gate, beta, S);
    set_tensor_name(result, "dn_gdn_result", il);

    // 提取 per-token 输出（结果的第一部分）
    ggml_tensor* output = ggml_view_4d(ctx, result,
        head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
        ggml_row_size(result->type, head_v_dim),
        ggml_row_size(result->type, head_v_dim * num_v_heads),
        ggml_row_size(result->type, head_v_dim * num_v_heads * n_seq_tokens),
        0
    );
    set_tensor_name(output, "dn_delta_out", il);

    // 提取并写回新的循环状态（结果的第二部分）
    ggml_tensor* new_state = ggml_view_4d(ctx, result,
        head_v_dim, head_k_dim, num_v_heads, n_seqs,
        ggml_row_size(result->type, head_v_dim),
        ggml_row_size(result->type, head_v_dim * head_k_dim),
        ggml_row_size(result->type, head_v_dim * head_k_dim * num_v_heads),
        ggml_row_size(result->type, static_cast<int64_t>(head_v_dim) * num_v_heads * n_seq_tokens * n_seqs)
    );

    ggml_tensor* rec_dst = ggml_view_1d(ctx, rec_all, rec_slot_floats, static_cast<size_t>(slot_idx) * rec_all->nb[1]);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_reshape_1d(ctx, new_state, rec_slot_floats), rec_dst));

    // =========================================================================
    // 阶段 7: 门控 RMSNorm
    // =========================================================================
    // 对输出进行归一化，并应用门控机制
    // 类似于 Transformer 中的 pre-attention norm，但增加了门控来控制信息流
    // w_norm 是 [head_v_dim]，通过广播应用到所有 head 和 token
    ggml_tensor* z_4d   = ggml_reshape_4d(ctx, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);
    ggml_tensor* normed = ggml_rms_norm(ctx, output, rms_norm_eps);
    normed = ggml_mul(ctx, normed, w_norm);
    ggml_tensor* z_silu = ggml_silu(ctx, z_4d);
    ggml_tensor* gated  = ggml_mul(ctx, normed, z_silu);
    set_tensor_name(gated, "dn_gated", il);

    // =========================================================================
    // 阶段 8: 输出投影
    // =========================================================================
    // 将内部维度投影回嵌入维度
    // gated: [head_v_dim, num_v_heads, n_tokens, n_seqs]
    // flat: [head_v_dim * num_v_heads, n_tokens, n_seqs]
    // out_proj: [n_embd, n_tokens]
    ggml_tensor* flat = ggml_reshape_3d(ctx, gated, static_cast<int64_t>(head_v_dim) * num_v_heads, n_seq_tokens, n_seqs);
    ggml_tensor* out_proj = ggml_mul_mat(ctx, w_out, flat);
    set_tensor_name(out_proj, "dn_output", il);
    out_proj = ggml_reshape_2d(ctx, out_proj, n_embd, n_seq_tokens * n_seqs);

    return out_proj;
}

/**
 * @brief 构建 DeltaNet 层（封装函数）
 * 
 * 为物理层 il 构建 DeltaNet 子图，封装了权重获取逻辑
 * 
 * @param ctx ggml 上下文
 * @param gf 计算图
 * @param cur 输入张量
 * @param dn_state DeltaNet 状态
 * @param state_hp DeltaNet 状态参数
 * @param num_k_heads K 头数量
 * @param n_embd 嵌入维度
 * @param dn_idx DeltaNet 层索引
 * @param n_tokens token 数量
 * @param slot_idx 槽位索引
 * @param rms_norm_eps RMS 归一化 epsilon
 * @param il 物理层索引
 * @return DeltaNet 输出
 */
ggml_tensor* Qwen35moeForwardPass::build_dn_layer(
    ggml_context*   ctx,
    ggml_cgraph*    gf,
    ggml_tensor*    cur,
    DeltaNetState*  dn_state,
    const DeltaNetStateParams& state_hp,
    uint32_t num_k_heads,
    uint32_t n_embd,
    uint32_t dn_idx,
    uint32_t n_tokens,
    uint32_t slot_idx,
    float    rms_norm_eps,
    int      il
)
{
    // 获取 DeltaNet 层的各项权重
    struct ggml_tensor* attn_qkv_weight = model_->get_attn_qkv_weight(il);
    struct ggml_tensor* attn_gate_weight = model_->get_attn_gate_weight(il);
    struct ggml_tensor* ssm_beta_weight = model_->get_ssm_beta_weight(il);
    struct ggml_tensor* ssm_alpha_weight = model_->get_ssm_alpha_weight(il);
    struct ggml_tensor* ssm_dt_bias = model_->get_ssm_dt_weight(il);
    struct ggml_tensor* ssm_a = model_->get_ssm_a_weight(il);
    struct ggml_tensor* ssm_conv1d_weight = model_->get_ssm_conv1d_weight(il);
    struct ggml_tensor* ssm_norm_weight = model_->get_ssm_norm_weight(il);
    struct ggml_tensor* ssm_out_weight = model_->get_ssm_out_weight(il);
    
    // 调用核心 DeltaNet 构建函数
    return build_deltanet_layer(
        ctx, gf, cur, dn_state,
        dn_idx, slot_idx, n_tokens,
        attn_qkv_weight,
        attn_gate_weight,
        ssm_beta_weight,
        ssm_alpha_weight,
        ssm_dt_bias,
        ssm_a,
        ssm_conv1d_weight,
        ssm_norm_weight,
        ssm_out_weight,
        static_cast<int>(n_embd),                                          // n_embd
        static_cast<int>(state_hp.head_v_dim * state_hp.num_v_heads),      // d_inner
        static_cast<int>(state_hp.head_k_dim),
        static_cast<int>(num_k_heads),
        static_cast<int>(state_hp.num_v_heads),
        static_cast<int>(state_hp.head_v_dim),
        static_cast<int>(state_hp.conv_channels),
        static_cast<int>(state_hp.conv_kernel),
        rms_norm_eps,
        il
    );
}

/**
 * @brief DeltaNet 层统一构建入口
 * 
 * 根据阶段（Prefill/Decode）调用不同的构建函数
 * 
 * @param ctx ggml 上下文
 * @param gf 计算图
 * @param input 输入张量
 * @param dn_idx DeltaNet 层索引
 * @param phase 阶段（Prefill/Decode）
 * @param prefill_args Prefill 参数
 * @param decode_args Decode 参数
 * @param dn_state DeltaNet 状态
 * @param hp DeltaNet 参数
 * @param il 物理层索引
 * @return DeltaNet 输出 [n_embd, n_tokens/n_batch]
 */
ggml_tensor* Qwen35moeForwardPass::build_all_deltanet_layer(
    ggml_context*      ctx,
    ggml_cgraph*       gf,
    ggml_tensor*       input,
    uint32_t           dn_idx,
    Phase              phase,
    const PrefillArgs& prefill_args,
    const DecodeArgs*  decode_args,
    DeltaNetState*     dn_state,
    const DeltaNetParams&     hp,
    int      il
)
{
    if (phase == Phase::Prefill) {
        return build_all_deltanet_layer_prefill(ctx, gf, input, dn_idx, prefill_args, dn_state, hp, il);
    } else {
        if (!decode_args) {
            throw std::runtime_error("DeltaNetLayer::build: decode_args must be non-null for Phase::Decode");
        }
        
        return build_all_deltanet_layer_decode(ctx, gf, input, dn_idx, *decode_args, dn_state, hp, il);
    }
}

/**
 * @brief Prefill 阶段的 DeltaNet 构建
 * 
 * @param ctx ggml 上下文
 * @param gf 计算图
 * @param input 输入张量
 * @param dn_idx DeltaNet 层索引
 * @param prefill_args Prefill 参数
 * @param dn_state DeltaNet 状态
 * @param hp DeltaNet 参数
 * @param il 物理层索引
 * @return DeltaNet 输出
 */
ggml_tensor* Qwen35moeForwardPass::build_all_deltanet_layer_prefill(
    ggml_context*      ctx,
    ggml_cgraph*       gf,
    ggml_tensor*       input,
    uint32_t           dn_idx,
    const PrefillArgs& prefill_args,
    DeltaNetState*     dn_state,
    const DeltaNetParams&     hp,
    int      il
)
{
    // 从输入形状推导 n_embd
    const int n_embd = static_cast<int>(input->ne[0]);

    // 获取权重并调用核心构建函数
    struct ggml_tensor* attn_qkv_weight = model_->get_attn_qkv_weight(il);
    struct ggml_tensor* attn_gate_weight = model_->get_attn_gate_weight(il);
    struct ggml_tensor* ssm_beta_weight = model_->get_ssm_beta_weight(il);
    struct ggml_tensor* ssm_alpha_weight = model_->get_ssm_alpha_weight(il);
    struct ggml_tensor* ssm_dt_bias = model_->get_ssm_dt_weight(il);
    struct ggml_tensor* ssm_a = model_->get_ssm_a_weight(il);
    struct ggml_tensor* ssm_conv1d_weight = model_->get_ssm_conv1d_weight(il);
    struct ggml_tensor* ssm_norm_weight = model_->get_ssm_norm_weight(il);
    struct ggml_tensor* ssm_out_weight = model_->get_ssm_out_weight(il);
    return build_deltanet_layer(
        ctx, gf, input, dn_state, dn_idx, prefill_args.slot_idx, prefill_args.n_tokens,
        attn_qkv_weight, attn_gate_weight, ssm_beta_weight, ssm_alpha_weight, ssm_dt_bias,
        ssm_a, ssm_conv1d_weight, ssm_norm_weight, ssm_out_weight,
        n_embd, hp.d_inner,
        hp.head_k_dim, hp.num_k_heads, hp.num_v_heads, hp.head_v_dim,
        hp.conv_channels, hp.conv_kernel, hp.rms_norm_eps,
        static_cast<int>(dn_idx)
    );
}

/**
 * @brief Decode 阶段的 DeltaNet 构建（支持多序列并行）
 * 
 * Decode 阶段每个槽位处理一个 token，但支持多槽位并行
 * 
 * @param ctx ggml 上下文
 * @param gf 计算图
 * @param input 输入张量
 * @param dn_idx DeltaNet 层索引
 * @param decode_args Decode 参数
 * @param dn_state DeltaNet 状态
 * @param hp DeltaNet 参数
 * @param il 物理层索引
 * @return DeltaNet 输出
 */
ggml_tensor* Qwen35moeForwardPass::build_all_deltanet_layer_decode(
    ggml_context*     ctx,
    ggml_cgraph*      gf,
    ggml_tensor*      input,
    uint32_t          dn_idx,
    const DecodeArgs& decode_args,
    DeltaNetState*     dn_state,
    const DeltaNetParams&     hp,
    int      il
)
{
    // Decode: 批量单 token，每个槽位一个
    // 为了保持状态写入正确，我们逐个槽位处理，然后拼接输出
    // 当 n_batch == 1 时，直接调用 prefill
    const int n_embd  = static_cast<int>(input->ne[0]);
    const int n_batch = static_cast<int>(decode_args.slots.size());

    if (n_batch == 0) {
        throw std::runtime_error("DeltaNetLayer::build_decode: slots must be non-empty");
    }

    // 单批次优化：直接调用 prefill
    if (n_batch == 1) {
        PrefillArgs pa;
        pa.n_tokens = 1;
        pa.slot_idx = decode_args.slots[0];
        return build_all_deltanet_layer_prefill(ctx, gf, input, dn_idx, pa, dn_state, hp, il);
    }

    // 多槽位 decode：逐个槽位切片输入，独立处理，然后拼接
    std::vector<ggml_tensor*> slot_outs;
    slot_outs.reserve(n_batch);

    for (int b = 0; b < n_batch; ++b) {
        // 提取第 b 个槽位的 token: input[:, b]
        ggml_tensor* token_in = ggml_view_2d(ctx, input,
            n_embd, 1,
            input->nb[1],
            static_cast<size_t>(b) * input->nb[1]
        );

        PrefillArgs pa;
        pa.n_tokens = 1;
        pa.slot_idx = decode_args.slots[b];
        ggml_tensor* out_b = build_all_deltanet_layer_prefill(ctx, gf, token_in, dn_idx, pa, dn_state, hp, il);
        slot_outs.push_back(out_b);
    }

    // 沿 token 维度拼接所有槽位的输出
    ggml_tensor* combined = slot_outs[0];
    for (int b = 1; b < n_batch; ++b) {
        combined = ggml_concat(ctx, combined, slot_outs[b], 1);
    }
    
    return combined;
}

/**
 * @brief 设置张量名称（带层索引）
 * 
 * @param tensor 张量指针
 * @param name 基础名称
 * @param il 层索引（-1 表示不带索引）
 */
void Qwen35moeForwardPass::set_tensor_name(ggml_tensor* tensor, const char* name, int il) const {
    if (il != -1) {
        char new_name[128];
        snprintf(new_name, sizeof(new_name), "%s.%d", name, il);
        ggml_set_name(tensor, new_name);
    } else {
        ggml_set_name(tensor, name);
    }
}

/**
 * @brief 构建输出头（LM Head）
 * 
 * 包含最终归一化和线性投影到 vocab 空间
 * 
 * @param gf 计算图
 * @param cur 输入张量（Transformer 最后一层输出）
 */
void Qwen35moeForwardPass::build_output_head(ggml_cgraph* gf, ggml_tensor* cur) {
    // 最终 RMS 归一化
    struct ggml_tensor* output_norm = model_->get_output_norm_weight();
    cur = build_norm(gf, cur, output_norm, -1);
    set_tensor_name(cur, "final_norm");

    // LM Head 投影
    // 如果有独立的输出权重则使用，否则复用嵌入矩阵（权重共享）
    struct ggml_tensor* output = model_->get_output_weight();
    struct ggml_tensor* token_embedding = model_->get_token_embedding_weight();
    if (output != nullptr) {
        cur = ggml_mul_mat(ctx_, output, cur);
    } else {
        cur = ggml_mul_mat(ctx_, token_embedding, cur);
    }
    ggml_set_name(cur, "logits");
    ggml_build_forward_expand(gf, cur);

    if (sampling_top_k_ > 0 && sampling_temperature_ > 0.0f) {
        const float inv_temp = 1.0f / sampling_temperature_;
        if (!std::isfinite(inv_temp)) {
            throw std::runtime_error("invalid sampling temperature for device-side scaling");
        }
        ggml_tensor* scaled_logits = ggml_scale(ctx_, cur, inv_temp);
        set_tensor_name(scaled_logits, "logits_scaled");
        ggml_build_forward_expand(gf, scaled_logits);

        const int64_t n_vocab = scaled_logits->ne[0];
        int64_t k = sampling_top_k_;
        if (k > n_vocab) {
            k = n_vocab;
        }
        if (k > 0) {
            ggml_tensor* topk_indices = ggml_top_k(ctx_, scaled_logits, static_cast<int32_t>(k));
            set_tensor_name(topk_indices, "sample_topk_idx");
            ggml_build_forward_expand(gf, topk_indices);
        }
    }
}

/**
 * @brief 判断某层是否为 Full Attention 层
 * 
 * Qwen3.5-MoE 使用混合架构：大部分层使用 DeltaNet，每隔一定间隔使用 Full Attention
 * 模式：layers (interval-1), 2*(interval-1)+1, ... 即 layer_idx % interval == interval-1
 * 
 * @param layer_idx 物理层索引
 * @return true 表示是 Full Attention 层，false 表示是 DeltaNet 层
 */
bool Qwen35moeForwardPass::is_full_attention_layer(uint32_t layer_idx) const {
    return (layer_idx % model_->meta_->qwen35moe.full_attention_interval) == (model_->meta_->qwen35moe.full_attention_interval - 1);
}

/**
 * @brief 获取槽位的物理缓存位置
 * 
 * @param slot_idx 槽位索引
 * @return 物理缓存位置
 */
uint32_t Qwen35moeForwardPass::get_physical_cache_pos(uint32_t slot_idx) const {
    return kv_cache_ ? kv_cache_->get_pos(slot_idx) : 0;
}

// =========================================================================
// 缓存管理相关函数
// =========================================================================

/**
 * @brief 推进缓存位置（KV 缓存和 SnapKV）
 * 
 * 在推理完成后调用，将缓存指针向前移动 n_tokens 个位置
 * 
 * @param n_tokens 处理的 token 数量
 * @param slot_idx 槽位索引
 */
void Qwen35moeForwardPass::advance_cache(uint32_t n_tokens, uint32_t slot_idx) {
    // 推进 KV 缓存位置
    if (kv_cache_) 
        kv_cache_->advance(n_tokens, slot_idx);
    // DeltaNet 状态在计算图中自动更新，无需手动推进
    // 推进 SnapKV 的逻辑序列位置
    snapkv_advance_seq_pos(slot_idx, n_tokens);
}

/**
 * @brief 获取 SnapKV 的逻辑序列位置
 * 
 * SnapKV 是一个优化的 KV 缓存策略，用于减少内存带宽消耗
 * 返回 0 表示该槽位未激活 SnapKV
 * 
 * @param slot_idx 槽位索引
 * @return 逻辑序列位置（0 表示未激活）
 */
uint32_t Qwen35moeForwardPass::snapkv_get_seq_pos(uint32_t slot_idx) const {
    return (slot_idx < snapkv_seq_pos_.size()) ? snapkv_seq_pos_[slot_idx] : 0;
}

/**
 * @brief 推进 SnapKV 的逻辑序列位置
 * 
 * 与 advance_cache 配合使用，更新 SnapKV 的位置计数器
 * 
 * @param slot_idx 槽位索引
 * @param n_tokens 推进的 token 数量
 */
void Qwen35moeForwardPass::snapkv_advance_seq_pos(uint32_t slot_idx, uint32_t n_tokens) {
    // 只有当槽位有效且 SnapKV 已激活（位置 > 0）时才推进
    if (slot_idx < snapkv_seq_pos_.size() && snapkv_seq_pos_[slot_idx] > 0)
        snapkv_seq_pos_[slot_idx] += n_tokens;
}

/**
 * @brief 从 GPU 获取输出 logits
 * 
 * 从计算图中提取 "logits" 张量，并将其数据从 GPU 复制到 CPU
 * 
 * @param gf 计算图
 * @return logits 数据（浮点数组）
 */
std::vector<float> Qwen35moeForwardPass::get_output_logits(ggml_cgraph* gf) {
    // 从图中获取名为 "logits" 的张量
    ggml_tensor* logits = ggml_graph_get_tensor(gf, "logits");
    if (!logits) {
        throw std::runtime_error("logits tensor not found in graph");
    }
    
    // 计算 logits 数据大小并分配内存
    size_t logits_size = ggml_nbytes(logits);
    std::vector<float> logits_result(logits_size / sizeof(float));
    
    // 从 GPU 后端复制数据到 CPU
    ggml_backend_tensor_get(logits, logits_result.data(), 0, logits_size);
    
    return logits_result;
}

TopKSampleCandidates Qwen35moeForwardPass::get_output_topk_candidates(ggml_cgraph* gf, uint32_t token_col) {
    ggml_tensor* logits = ggml_graph_get_tensor(gf, "logits_scaled");
    if (!logits) {
        throw std::runtime_error("logits_scaled tensor not found in graph");
    }
    ggml_tensor* topk_idx = ggml_graph_get_tensor(gf, "sample_topk_idx");
    if (!topk_idx) {
        throw std::runtime_error("sample_topk_idx tensor not found in graph");
    }
    if (topk_idx->type != GGML_TYPE_I32) {
        throw std::runtime_error("sample_topk_idx tensor type mismatch");
    }
    if (token_col >= static_cast<uint32_t>(topk_idx->ne[1])) {
        throw std::runtime_error("token column out of range for sample_topk_idx");
    }

    const uint32_t k = static_cast<uint32_t>(topk_idx->ne[0]);
    TopKSampleCandidates result;
    result.token_ids.resize(k);
    result.logits.resize(k);

    const size_t idx_offset = static_cast<size_t>(token_col) * topk_idx->nb[1];
    const size_t idx_bytes = static_cast<size_t>(k) * sizeof(int32_t);
    ggml_backend_tensor_get(topk_idx, result.token_ids.data(), idx_offset, idx_bytes);

    const size_t logits_col_offset = static_cast<size_t>(token_col) * logits->nb[1];
    for (uint32_t i = 0; i < k; ++i) {
        const int32_t token_id = result.token_ids[i];
        if (token_id < 0 || token_id >= logits->ne[0]) {
            throw std::runtime_error(
                "invalid top-k token index: " + std::to_string(token_id) +
                ", valid range: [0, " + std::to_string(logits->ne[0]) + ")");
        }
        float value = 0.0f;
        const size_t logit_offset = logits_col_offset + static_cast<size_t>(token_id) * sizeof(float);
        ggml_backend_tensor_get(logits, &value, logit_offset, sizeof(float));
        result.logits[i] = value;
    }

    return result;
}
