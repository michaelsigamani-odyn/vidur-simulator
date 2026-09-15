from math import ceil

from vidur.config import ReplicaConfig
from vidur.logger import init_logger
import os
import json

logger = init_logger(__name__)


class ParamCounter:
    def __init__(self, replica_config: ReplicaConfig) -> None:
        self._replica_config = replica_config
        self._model_config = self._replica_config.model_config
        self.config = self._model_config

        assert (
            self._model_config.num_q_heads % self._replica_config.tensor_parallel_size
            == 0
        )
        assert (
            self._model_config.num_layers % self._replica_config.num_pipeline_stages
            == 0
        )
        assert (
            self._model_config.embedding_dim % self._replica_config.tensor_parallel_size
            == 0
        )
        assert self._model_config.embedding_dim % self._model_config.num_q_heads == 0

        self._num_layers_per_pipeline_stage = (
            self._model_config.num_layers // self._replica_config.num_pipeline_stages
        )
        self._attention_head_dim = (
            self._model_config.embedding_dim // self._model_config.num_q_heads
        )
        self._q_heads_per_tensor_parallel_worker = (
            self._model_config.num_q_heads // self._replica_config.tensor_parallel_size
        )
        self._kv_heads_per_tensor_parallel_worker = ceil(
            self._model_config.num_kv_heads / self._replica_config.tensor_parallel_size
        )
        
        # TODO(tianhao909): support FP8 precision quantization
        # TODO(tianhao909):  FP8 
        if self._replica_config.pd_p2p_comm_dtype == "fp8":
            logger.debug(f"FP8 enabled, dtype={self._replica_config.pd_p2p_comm_dtype}")
            self.use_fp8 = True
        else:
            logger.debug(f"FP8 disabled, dtype={self._replica_config.pd_p2p_comm_dtype}")
            self.use_fp8 = False
        self.tp = self._replica_config.tensor_parallel_size 
        self.ep = self._replica_config.expert_model_parallel_size
        
        #  | Flag to track if debug info has been printed
        self._debug_printed = False
        
        if self._replica_config.model_name in ['deepseek-671B', 'qwen3-moe-235B', 'qwen3-next-80B']:
            self.model_config_postprocessing()
            # self._model_config
            
    def model_config_postprocessing(self, ):
        #  | Initialize configuration dictionary
        d = dict()
        if self._replica_config.model_name == 'deepseek-671B':
            #  | Use relative path to locate config file
            config_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "hf_configs", "deepseek_v3_config.json")
        elif self._replica_config.model_name == 'qwen3-moe-235B':
            # TODO(tianhao909): add corresponding JSON config file for Qwen3-MoE
            # TODO(tianhao909):  JSON 
            config_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "hf_configs", "qwen3_moe_config.json")
        elif self._replica_config.model_name == 'qwen3-next-80B':
            config_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "hf_configs", "qwen3-next-80B-A3B_config.json")
        logger.debug(f"config_path={config_path}")
        #  | Check if config file exists
        if not os.path.exists(config_path):
            logger.warning(f"Config file {config_path} not found, using default config")
            return
        # JSON | Load JSON config in read-only mode
        with open(config_path, "r") as f:
            d = json.load(f)

        #  | Model hidden size dimension
        self._model_config.hidden_size = d["hidden_size"]
        #  | Number of hidden layers
        self._model_config.num_hidden_layers = d["num_hidden_layers"]

        # （）
        # Determine if using hybrid attention (alternating full and linear attention)
        self._model_config.is_hybrid_linear = d.get("full_attention_interval") is not None
        if self._model_config.is_hybrid_linear:
            # ：N | Full attention layers: inserted every N layers
            self._model_config.num_full_attn_layers = (
                self._model_config.num_hidden_layers // d["full_attention_interval"]
            )
            # ： | Linear attention layers: total - full attention
            self._model_config.num_linear_attn_layers = (
                self._model_config.num_hidden_layers - self._model_config.num_full_attn_layers
            )
            #  | Linear attention convolution kernel dimension
            self._model_config.linear_conv_kernel_dim = d["linear_conv_kernel_dim"]
            #  | Linear attention key head dimension
            self._model_config.linear_key_head_dim = d["linear_key_head_dim"]
            #  | Linear attention number of key heads
            self._model_config.linear_num_key_heads = d["linear_num_key_heads"]
            #  | Linear attention value head dimension
            self._model_config.linear_value_head_dim = d["linear_value_head_dim"]
            #  | Linear attention number of value heads
            self._model_config.linear_num_value_heads = d["linear_num_value_heads"]

        self._model_config.attn_type = "MHA/GQA"  # Default attention: MHA or GQA / 
        if "kv_lora_rank" in d:  # If kv_lora_rank present, use MLA attention /  kv_lora_rank，MLA
            self._model_config.attn_type = "MLA"

        # Attention mechanism parameter setup / 
        if self._model_config.attn_type == "MHA/GQA":  # MHA/GQA type
            self._model_config.num_attention_heads = d["num_attention_heads"]  # Number of attention heads / 
            self._model_config.num_key_value_heads = d["num_key_value_heads"]  # KV heads for GQA / 
            if "head_dim" in d:  # If head_dim specified in config
                self._model_config.head_dim = d["head_dim"]
            else:
                self._model_config.head_dim = self._model_config.hidden_size // self._model_config.num_attention_heads  # Compute from hidden_size / heads
        elif self._model_config.attn_type == "MLA":  # MLA type
            self._model_config.q_lora_rank = d["q_lora_rank"]  # Query LoRA rank / LoRA
            self._model_config.qk_nope_head_dim = d["qk_nope_head_dim"]  # QK no-position head dim / QK
            self._model_config.qk_rope_head_dim = d["qk_rope_head_dim"]  # QK RoPE head dim / RoPEQK
            self._model_config.kv_lora_rank = d["kv_lora_rank"]  # KV LoRA rank / LoRA
            self._model_config.num_attention_heads = d["num_attention_heads"]  # Total attention heads / 
            self._model_config.v_head_dim = d["v_head_dim"]  # Value head dim / 
            self._model_config.qk_head_dim = self._model_config.qk_nope_head_dim + self._model_config.qk_rope_head_dim  # QK total head dim = nope + rope

        # FFN/MoE (Feed-Forward Network / Mixture of Experts) configuration
        # FFN/MoE（/）
        self._model_config.is_moe = True  # Default enable MoE / MoE
        if "num_routed_experts" in d:  # Routed expert count / 
            self._model_config.num_routed_experts = d["num_routed_experts"]
        elif "num_experts" in d:  # Fallback to num_experts field
            self._model_config.num_routed_experts = d["num_experts"]
        else:
            self._model_config.is_moe = False  # No MoE if no expert fields / MoE
            self._model_config.num_routed_experts = 1  # Single expert (standard FFN) / 

        if self._model_config.is_moe:  # If MoE enabled / MoE
            self._model_config.num_experts_per_tok = d["num_experts_per_tok"]  # Experts activated per token / token
            self._model_config.intermediate_size = d["moe_intermediate_size"]  # Per-expert intermediate size / 
            self._model_config.num_shared_experts = d.get("num_shared_experts", 0)  # Shared expert count / 
        else:  # Standard FFN (no MoE) / MoE
            self._model_config.num_experts_per_tok = 1  # Single "expert" / FFN
            self._model_config.intermediate_size = d["intermediate_size"]  # Standard FFN intermediate size / FFN
            self._model_config.num_shared_experts = 0  # No shared experts / 

    def get_num_parameters_per_layer(self) -> int:
        num_parameters = 0
        # weights for attention metrics Wq, Wk, Wv
        num_parameters += (
            self._model_config.embedding_dim
            * self._attention_head_dim
            * (
                self._q_heads_per_tensor_parallel_worker
                + 2 * self._kv_heads_per_tensor_parallel_worker
            )
        )
        # weights for attention metrics Wo
        num_parameters += (
            self._model_config.embedding_dim
            * self._attention_head_dim
            * self._q_heads_per_tensor_parallel_worker
        )
        # fc layer weights
        if self._model_config.use_gated_mlp:
            num_parameters += (
                3
                * self._model_config.embedding_dim
                * self._model_config.mlp_hidden_dim
                // self._replica_config.tensor_parallel_size
            )
        else:
            num_parameters += (
                2
                * self._model_config.embedding_dim
                * self._model_config.mlp_hidden_dim
                // self._replica_config.tensor_parallel_size
            )

        return num_parameters
    
    # Layer Dimension
        # First 3 layers are dense (no gate). Based on the above calculation,
        # each of the first 3 layers of DeepSeek V3 has parameter count:
        # Layer 
        #  3  dense， gate，，DeepSeek V3  3 ：
            # (MLAQLoRA48,760,320 + MLAKVLoRA20,906,496 +  MLAWO117,440,512 + （pre+post）attention layernorm14336（7168+7168）） + （44,040,192 * 9 （9  3  dense，8））
            # (48,760,320 + 20,906,496 + 117,440,512 + 14336) + (44,040,192 * 9) = 583,483,392
        # Last 58 layers are MoE sparse-activated experts. DeepSeek V3 per-layer params:
        #  58  MoE ，，DeepSeek V3  58 ：
            # (48,760,320 + 20,906,496 + 117,440,512 + 14336) + (44,040,192 * 257 + 1,835,264) = 11,507,286,272
            # (MLAQLoRA48,760,320 + MLAKVLoRA20,906,496 +  MLAWO117,440,512 + （pre+post）attention layernorm14336（7168+7168）） + （44,040,192 * 257 （256） +  Gate 1,835,264）
    def get_num_parameters_per_layer_by_layer_id(self, layer_id: int = 0) -> tuple:
        """
        Get parameter count per layer by layer_id.
        Returns tuple: (params_per_layer, prefill_params_per_layer, decode_params_per_layer)

         layer_id 
        : (params_per_layer, prefill_params_per_layer, decode_params_per_layer)
        """
        #  | Initialize variables
        params_per_layer_per_gpu = 0
        prefill_params_per_layer_per_gpu = 0
        decode_params_per_layer_per_gpu = 0
            
        if self._replica_config.model_name == 'deepseek-671B':
            #  | Only print debug info on first call
            if not self._debug_printed:
                logger.info("{s:{c}^{n}}".format(s="[ParamCounter] DeepSeek-671B Model Weights", n=60, c="-"))
                attn_params_bytes = self.get_attn_params_size(self._model_config, self.use_fp8)
                expert_params_bytes = self.get_expert_params_size(self._model_config, self.use_fp8)
                logger.info(f"[ParamCounter] One MLA params size (MB): {attn_params_bytes / 1024 / 1024:.2f}")
                logger.info(f"[ParamCounter] One expert params size (MB): {expert_params_bytes / 1024 / 1024:.2f}")
                logger.info(f"[ParamCounter] use_fp8={self.use_fp8}, tp={self.tp}, ep={self.ep}")
                self._debug_printed = True
                
            if layer_id >= 0 and layer_id <= 2:
                #  3  dense，81
                # First 3 layers are dense, each layer activates 8 routed experts + 1 shared expert
                params_per_layer_per_gpu = (self.get_mla_params_size(self._model_config, self.use_fp8)/self.tp + 
                                           self.get_expert_params_size(self._model_config, self.use_fp8) * (8 + 1) / self.tp)
                prefill_params_per_layer_per_gpu = params_per_layer_per_gpu 
                decode_params_per_layer_per_gpu = params_per_layer_per_gpu
                    
            elif layer_id >= 3 and layer_id <= 60:
                #  58  MoE 
                # Remaining 58 layers are MoE sparse activated experts
                mla_params = self.get_mla_params_size(self._model_config, self.use_fp8) / self.tp
                expert_params = self.get_expert_params_size(self._model_config, self.use_fp8)
                    
                params_per_layer_per_gpu = mla_params + expert_params * (256/self.ep + 1)
                prefill_params_per_layer_per_gpu = mla_params + expert_params * (256/self._replica_config.prefill_world_size + 1)
                decode_params_per_layer_per_gpu = mla_params + expert_params * (256/self._replica_config.decode_world_size + 1)
                    
        elif self._replica_config.model_name == 'qwen3-next-80B':
            #  | Only print debug info on first call
            if not self._debug_printed:
                logger.info("{s:{c}^{n}}".format(s="[ParamCounter] Qwen3-Next-80B Model Weights", n=60, c="-"))
                full_attn_params_bytes = self.get_attn_params_size(self._model_config, self.use_fp8)
                linear_attn_params_bytes = self.get_linear_attn_params_size(self._model_config, self.use_fp8)
                expert_params_bytes = self.get_expert_params_size(self._model_config, self.use_fp8)
                logger.info(f"[ParamCounter] One full attn params size (MB): {full_attn_params_bytes / 1024 / 1024:.2f}")
                logger.info(f"[ParamCounter] One linear attn params size (MB): {linear_attn_params_bytes / 1024 / 1024:.2f}")
                logger.info(f"[ParamCounter] One expert params size (MB): {expert_params_bytes / 1024 / 1024:.2f}")
                logger.info(f"[ParamCounter] use_fp8={self.use_fp8}, tp={self.tp}, ep={self.ep}")
                self._debug_printed = True
                    
            full_attn_params_bytes = self.get_attn_params_size(self._model_config, self.use_fp8)
            linear_attn_params_bytes = self.get_linear_attn_params_size(self._model_config, self.use_fp8)
            expert_params_bytes = self.get_expert_params_size(self._model_config, self.use_fp8)
                
            # :  | Base params: expert network part
            params_per_layer_per_gpu = expert_params_bytes * (
                self.config.num_shared_experts + self.config.num_routed_experts / self._replica_config.world_size
            )
            prefill_params_per_layer_per_gpu = expert_params_bytes * (
                self.config.num_shared_experts + self.config.num_routed_experts / self._replica_config.prefill_world_size
            )
            decode_params_per_layer_per_gpu = expert_params_bytes * (
                self.config.num_shared_experts + self.config.num_routed_experts / self._replica_config.decode_world_size
            )
                
            #  layer_id  | Add attention layer params based on layer_id
            if layer_id % 4 == 3:  # Full attention layers (e.g., layer 3, 7, 11...)
                params_per_layer_per_gpu += full_attn_params_bytes / self.tp
                prefill_params_per_layer_per_gpu += full_attn_params_bytes / self.tp
                decode_params_per_layer_per_gpu += full_attn_params_bytes / self.tp
            else:  # Linear attention layers (e.g., layer 0, 1, 2, 4, 5, 6...)
                params_per_layer_per_gpu += linear_attn_params_bytes / self.tp
                prefill_params_per_layer_per_gpu += linear_attn_params_bytes / self.tp
                decode_params_per_layer_per_gpu += linear_attn_params_bytes / self.tp
                    
        elif self._replica_config.model_name == 'qwen3-moe-235B':
            #  | Only print debug info on first call
            if not self._debug_printed:
                logger.info("{s:{c}^{n}}".format(s="[ParamCounter] Qwen3-MoE-235B Model Weights", n=60, c="-"))
                attn_params_bytes = self.get_mha_params_size(self._model_config, self.use_fp8)
                expert_params_bytes = self.get_expert_params_size(self._model_config, self.use_fp8)
                logger.info(f"[ParamCounter] One MHA params size (MB): {attn_params_bytes / 1024 / 1024:.2f}")
                logger.info(f"[ParamCounter] One expert params size (MB): {expert_params_bytes / 1024 / 1024:.2f}")
                logger.info(f"[ParamCounter] use_fp8={self.use_fp8}, tp={self.tp}, ep={self.ep}")
                self._debug_printed = True
                
            # Qwen3-MoE-235B: 128, 0, MHA/GQA, dense
            # Qwen3-MoE-235B: 128 routed experts, 0 shared experts, MHA/GQA attention, no dense layers
            mha_params = self.get_mha_params_size(self._model_config, self.use_fp8)
            expert_params = self.get_expert_params_size(self._model_config, self.use_fp8)
                
            params_per_layer_per_gpu = mha_params + expert_params * 128
            prefill_params_per_layer_per_gpu = mha_params/self.tp + expert_params * (128/self._replica_config.prefill_world_size)
            decode_params_per_layer_per_gpu = mha_params/self.tp + expert_params * (128/self._replica_config.decode_world_size)
                
        return params_per_layer_per_gpu, prefill_params_per_layer_per_gpu, decode_params_per_layer_per_gpu

    def get_num_parameters_per_device(self) -> int:
        # TODO(tianhao909): refactor per-layer param calculation with layer_id support
        # TODO(tianhao909):  get_num_parameters_per_device  layer_id 
        if self._replica_config.model_name in ['deepseek-671B', 'qwen3-moe-235B', 'qwen3-next-80B']:
            # Reference: see ExecutionTime._get_block_execution_time_by_layer_id
            # Need to get start/end layer_id for the current pipeline stage
            #  ExecutionTime._get_block_execution_time_by_layer_id 
            # pipeline stagelayer id
            # try:
            
            pipeline_stage_id = getattr(self, '_pipeline_stage_id', 0)
            start_layer = pipeline_stage_id * self._num_layers_per_pipeline_stage
            end_layer = start_layer + self._num_layers_per_pipeline_stage
            logger.debug(f"pipeline_stage_id={pipeline_stage_id} num_layers_per_pipeline_stage={self._num_layers_per_pipeline_stage} start_layer={start_layer} end_layer={end_layer}")
            
            params_per_gpu = 0
            prefill_params_per_gpu = 0  #  | Fixed variable name
            decode_params_per_gpu = 0   #  | Fixed variable name
            for layer_id in range(start_layer, end_layer):
                params_per_layer, prefill_params_per_layer, decode_params_per_layer = self.get_num_parameters_per_layer_by_layer_id(layer_id)
                params_per_gpu += params_per_layer
                prefill_params_per_gpu += prefill_params_per_layer
                decode_params_per_gpu += decode_params_per_layer
                
            # params_per_gpu B | Unit is Bytes
            params_per_gpu_gb = params_per_gpu / 1024 / 1024 / 1024  # Convert to GB / GB
            prefill_params_per_gpu_gb = prefill_params_per_gpu / 1024 / 1024 / 1024  # Convert to GB / GB
            decode_params_per_gpu_gb = decode_params_per_gpu / 1024 / 1024 / 1024  # Convert to GB / GB
            logger.info("{:<40} {:<10.2f}".format("Per GPU params size (GB):", params_per_gpu_gb))
            logger.info("{:<40} {:<10.2f}".format("Prefill Per GPU params size (GB):", prefill_params_per_gpu_gb))
            logger.info("{:<40} {:<10.2f}".format("Decode Per GPU params size (GB) :", decode_params_per_gpu_gb))
            logger.info(f"Prefill: tp={self.tp} dp={self._replica_config._num_prefill_replicas} ep={self._replica_config.prefill_world_size} prefill_params_per_gpu_gb={prefill_params_per_gpu_gb} (GB)")
            logger.info(f"Decode: tp={self.tp} dp={self._replica_config._num_decode_replicas} ep={self._replica_config.decode_world_size} decode_params_per_gpu_gb={decode_params_per_gpu_gb} (GB)")
            assert self._replica_config._num_prefill_replicas % 1 == 0 and self._replica_config._num_decode_replicas % 1 == 0, "Prefill and Decode replicas must be integer"
                
            # # GPU（）
            # params_per_gpu = attn_params_bytes + expert_params_bytes * (
            #     self._model_config.num_shared_experts
            #     + self._model_config.num_routed_experts / self.ep
            # )
            
            # params_per_gpu = params_per_gpu / 1024 / 1024 / 1024  # GB
            # params_per_gpu *= self._model_config.num_hidden_layers  # 
            # # KV（、）
            # self.kvcache_mem = (
            #     self.gpu.mem - params_per_gpu - 15 - 5
            # )  # 15GB for runtime, 5GB for encoder（15GB，5GB）
            # print("{:<40} {:<10.2f}".format("Per GPU params size (GB):", params_per_gpu))  # GPU（GB）
            
            # Return tuple: (total params, prefill params, decode params)
            # : (, prefill, decode)
            return params_per_gpu, prefill_params_per_gpu, decode_params_per_gpu
            
            
            # except AttributeError:
            #     # _pipeline_stage_id，
            #     num_parameters_per_layer = self.get_num_parameters_per_layer()
            #     return num_parameters_per_layer * self._num_layers_per_pipeline_stage
        else:
            num_parameters_per_layer = self.get_num_parameters_per_layer()
            return num_parameters_per_layer * self._num_layers_per_pipeline_stage

    def get_attn_params_size(self, config, use_fp8):
        if config.attn_type == "MHA/GQA":  # MHA or GQA attention type / MHAGQA
            return get_mha_params_size(self, config, use_fp8)
        elif config.attn_type == "MLA":  # MLA architecture / MLA
            return get_mla_params_size(self, config, use_fp8)


    # Reference: /InferSim/params/params.py
    #  /InferSim/params/params.py
    # def get_mha_params_size(config: ModelConfig, use_fp8: bool):
    def get_mha_params_size(self, config, use_fp8):
        wq = config.hidden_size * config.num_attention_heads * config.head_dim  # Q weight: hidden * heads * head_dim
        wk = config.hidden_size * config.num_key_value_heads * config.head_dim  # K weight: hidden * kv_heads * head_dim
        wv = config.hidden_size * config.num_key_value_heads * config.head_dim  # V weight: hidden * kv_heads * head_dim
        wo = config.hidden_size * config.num_attention_heads * config.head_dim  # Output weight: hidden * heads * head_dim
        if use_fp8:  # FP8 quantization / FP8
            return wq + wk + wv + wo  # Single precision storage / 
        return 2 * (wq + wk + wv + wo)  # Full precision (e.g. FP16, 2 bytes per param) / 

    # MLA (suited for DeepSeek) / MLA（ DeepSeek）
        # DeepSeek V3 parameter derivation references:
        # dpsk v3 ：
            # https://zhuanlan.zhihu.com/p/21455638257 
            # https://yangwenbo.com/articles/deepseek-v3-parameter-size.html
        # "hidden_size": 7168,
        # "num_key_value_heads": 128,
        # "v_head_dim": 128,
        # "kv_lora_rank": 512,

        # "num_attention_heads": 128,
        # "q_lora_rank": 1536,

        # "qk_nope_head_dim": 128,
        # "qk_rope_head_dim": 64,

        # "num_hidden_layers": 61,
    # def get_mla_params_size(config: ModelConfig, use_fp8: bool):
    def get_mla_params_size(self, config, use_fp8):
        # Per-layer MLA Q LoRA params:
        #  MLA  Q  LoRA ：
            # = 7168 * 1536 + 1536 + 1536 * 128 * (128 + 64) = 48,760,320
            # = wq_down + wq_up
            # = (config.hidden_size * config.q_lora_rank) + (config.q_lora_rank * config.num_attention_heads * (config.qk_nope_head_dim + config.qk_rope_head_dim))
            # = (config.hidden_size * config.q_lora_rank) + (config.q_lora_rank * config.num_attention_heads * (config.qk_head_dim))
        wq_down = config.hidden_size * config.q_lora_rank  # Q LoRA down-projection / QLoRA
        wq_up = config.q_lora_rank * config.num_attention_heads * config.qk_head_dim  # Q LoRA up-projection / QLoRA
        # Per-layer MLA KV LoRA params:
        #  MLA  KV  LoRA ：
            # = 7168 * (512 + 64) + 512 + 512 * 128 * (128 + 128) = 20,906,496
            # = wkv_down + 512 + wkv_up (TODO(tianhao909): clarify what the 512 constant represents)
            # = config.hidden_size *（config.kv_lora_rank + config.qk_rope_head_dim) + 512 + config.kv_lora_rank * config.num_attention_heads * (config.qk_nope_head_dim + config.qk_rope_head_dim)
            # = (config.hidden_size * config.kv_lora_rank) + (config.kv_lora_rank * config.num_key_value_heads * (config.qk_nope_head_dim + config.qk_rope_head_dim))
        wkv_down = config.hidden_size * config.kv_lora_rank  # KV LoRA down-projection / KVLoRA
        wkv_up = (  # KV LoRA up-projection / KVLoRA
            config.kv_lora_rank
            * config.num_attention_heads
            * (config.qk_nope_head_dim + config.v_head_dim)
        )
        # Per-layer MLA output (WO) params:
        #  MLA  WO 
            # 128 * 128 * 7168 = 117,440,512
            # config.num_attention_heads * config.v_head_dim * config.hidden_size
        wo = config.hidden_size * config.num_attention_heads * config.v_head_dim  # Output weight / 
        if use_fp8:  # FP8 quantization / FP8
            return wq_down + wq_up + wkv_down + wkv_up + wo  # Sum all params (single precision) / 
        # Unit: Bytes / :B
        return 2 * (wq_down + wq_up + wkv_down + wkv_up + wo)  # FP16: multiply by 2 / 2

        # Additionally: pre+post attention layernorm params = 7168*2 = 14,336
        # DeepSeek V3 MLA total across 61 layers:
        # ：pre+post attention layernorm  = 7168*2 = 14,336
        #  DeepSeek V3  MLA  61 ：
            # (48,760,320 + 20,906,496 + 117,440,512 + 14,336) * 61 = 11,414,421,504 (~11B)


    # def get_gdn_params_size(config: ModelConfig, use_fp8: bool):
    def get_gdn_params_size(self, config, use_fp8):
        wq = config.hidden_size * config.linear_num_key_heads * config.linear_key_head_dim  # Q linear attention weight
        wk = wq  # K weight same as Q
        wv = (  # V weight params
            config.hidden_size
            * config.linear_num_value_heads
            * config.linear_value_head_dim
        )
        wz = wv  # Z weight same as V
        wa = config.hidden_size * config.linear_num_value_heads  # A gate params
        wb = wa  # B gate same as A
        s = wq + wk + wv + wz + wa + wb  # Total primary weight params
        wconv = (  # Conv kernel weight part 1
            config.linear_num_key_heads
            * config.linear_key_head_dim
            * config.linear_conv_kernel_dim
        )
        wconv += (  # Conv kernel weight part 2
            config.linear_num_key_heads
            * config.linear_key_head_dim
            * config.linear_conv_kernel_dim
        )
        wconv += (  # Conv kernel weight part 3
            config.linear_num_value_heads
            * config.linear_value_head_dim
            * config.linear_conv_kernel_dim
        )
        if use_fp8:  # FP8 quantization
            return s + wconv  # Primary + conv params (single precision)
        return 2 * s + wconv  # Primary *2, conv stays single precision


    # def get_attn_params_size(config: ModelConfig, use_fp8: bool):
    def get_attn_params_size(self, config, use_fp8):
        if config.attn_type == "MHA/GQA":  # MHA or GQA attention type
            return self.get_mha_params_size(config, use_fp8)
        elif config.attn_type == "MLA":  # MLA architecture
            return self.get_mla_params_size(config, use_fp8)


    # def get_linear_attn_params_size(config: ModelConfig, use_fp8: bool):
    def get_linear_attn_params_size(self, config, use_fp8):
        return self.get_gdn_params_size(config, use_fp8)  # Get linear attention (GD-Nets style) params / 

    # MoE (suited for DeepSeek) / MoE（ DeepSeek）
        # "num_hidden_layers": 61,
        # "hidden_size": 7168,
        # "moe_intermediate_size": 2048,  // Routed expert MLP intermediate dim /  MLP 
        # "n_shared_experts": 1,          // Shared expert count / 
        # "n_routed_experts": 256,        // Routed expert count / 
        # "first_k_dense_replace": 3,     // First K layers use dense instead of MoE / denseMoE
        # "intermediate_size": 18432,     // First 3 layers (9*moe_intermediate_size) / 3
        
        # Per-expert params: / ：
            # 7168 * 2048 * 3 = 44,040,192
            # config.hidden_size * config.moe_intermediate_size * 3
        # Router gate params: /  Gate ：
            # 256 * 7168 + 256 = 1,835,264
        # First 3 dense layers (8 routed + 1 shared per layer): /  3  dense（ 8 ）：
            # 44,040,192 * 9 * 3 = 1,189,085,184
        # Last 58 sparse layers (dynamically activate 8 routed): /  58 （ 8 ）：
            # (44,040,192 * 257 + 1,835,264) * 58 = 656,569,547,264
        # DeepSeek V3 MoE total params: / DeepSeek V3 MoE ：
            # 1,189,085,184 + 656,569,547,264 = 657,758,632,448 (~657B)
        # Active params per forward (1 shared + 8 routed): / （1 + 8）：
            # 44,040,192 * 9 * 61 + 1,835,264 * 58 = 24,284,510,720 (~24B)
    # def get_expert_params_size(config: ModelConfig, use_fp8: bool):
    def get_expert_params_size(self, config, use_fp8):
        if self._replica_config.model_name in [ 'qwen3-moe-235B']:
            # config.intermediate_size = 122888
            # config.moe_intermediate_size = 1536
            config.intermediate_size = config.moe_intermediate_size
            w = 3 * config.hidden_size * config.intermediate_size  # MoE expert FFN params (W1, W2, W3) / MoE
        else:
            w = 3 * config.hidden_size * config.intermediate_size  # MoE expert FFN params (W1, W2, W3) / MoE
        if not use_fp8:  # Not using FP8 / FP8
            w *= 2  # Double for FP16 / 
        return w  # Return expert params total / 


    # def load_attn_weights_time(config: ModelConfig, use_fp8: bool, gpu: GPU):
    def load_attn_weights_time(self, config, use_fp8, gpu):
        size = self.get_attn_params_size(config, use_fp8)  # Get attention weights size (bytes) / 
        return size / 1024 / 1024 / 1024 / gpu.mem_bw  # Convert to GB / mem_bw = load time (s) / GBGPU


    # def load_moe_weights_time(config: ModelConfig, use_fp8: bool, gpu: GPU, num_gpus):
    def load_moe_weights_time(self, config, use_fp8, gpu, num_gpus):
        size = self.get_expert_params_size(config, use_fp8)  # Get single expert weights size / 
        size *= config.num_routed_experts / num_gpus  # Distribute across GPUs / GPU
        return size / 1024 / 1024 / 1024 / gpu.mem_bw  # Load time in seconds / （）
