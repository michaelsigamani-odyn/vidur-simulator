from vidur.config import ReplicaConfig
from vidur.entities.replica import Replica
from vidur.logger import init_logger
from vidur.utils.param_counter import ParamCounter

logger = init_logger(__name__)


class MemoryPlanner:
    def __init__(self, replica_config: ReplicaConfig, replica: Replica) -> None:
        self._param_counter = ParamCounter(replica_config)
        self._replica = replica
        self._replica_config = replica_config
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

    # refer to https://github.com/Odyn/InferSim/blob/main/kvcache/kvcache.py
    def get_mha_kvcache_size(self, config, use_fp8):
        """
        Calculate MHA/GQA KV cache size (bytes) for all layers.
         MHA/GQA KV Cache （）

        Args:
            config: Model config object / 
            use_fp8: Whether to use FP8 precision /  FP8 

        Returns:
            int: Total KV cache size in bytes / KV Cache （）
        """
        # 2 for K and V; layers * KV heads * head_dim
        # 2  K  V 
        kvcache_size = (
            2 * config.num_hidden_layers * config.num_key_value_heads * config.head_dim
        )
        if not use_fp8:
            # FP16/BF16 uses 2x bytes vs FP8 / FP16/BF16  FP8 
            kvcache_size *= 2
        return kvcache_size

    # refer to https://github.com/Odyn/InferSim/blob/main/kvcache/kvcache.py
    # TODO(tianhao909): verify how TP splits MLA head_dim
    # TODO(tianhao909):  TP  MLA  head_dim
    def get_mla_kvcache_size(self, config, use_fp8):
        """
        Calculate MLA KV cache size (bytes) for all layers.
         MLA KV Cache （）

        MLA uses kv_lora_rank + qk_rope_head_dim instead of full KV heads.
        MLA  kv_lora_rank + qk_rope_head_dim  KV 

        Args:
            config: Model config object / 
            use_fp8: Whether to use FP8 precision /  FP8 

        Returns:
            int: Total KV cache size in bytes / KV Cache （）
        """
        kvcache_size = config.num_hidden_layers * (
            config.kv_lora_rank + config.qk_rope_head_dim
        )
        if not use_fp8:
            # FP16/BF16 uses 2x bytes vs FP8 / FP16/BF16  FP8 
            kvcache_size *= 2
        return kvcache_size

    # TODO(tianhao909): re-verify KV cache calc for DeepSeek/Qwen3
    # TODO(tianhao909):  DeepSeek/Qwen3  KV Cache 
    # TODO(tianhao909): use per-request prefill/decode seq_len for PD separation.
    # Currently prefill_input_seq_len and decode_output_seq_len are hardcoded to 1,
    # which significantly underestimates KV cache memory per request.
    # In production, a single request may have thousands of tokens.
    # Fixing this requires passing actual seq_len from the request generator or
    # using config-level max_prefill_tokens / max_decode_tokens.
    # This is an architectural change that affects the entire memory planning pipeline.
    # TODO(tianhao909): PD  req  prefill_input_seq_len  decode_output_seq_len。
    #  seq_len=1  KV cache 。
    #  token。
    #  seq_len， max_prefill_tokens / max_decode_tokens。
    # 。
    # TODO(tianhao909): optimize from static to dynamic token allocation.
    # Currently allocates by max_request_tokens statically, could be optimized to dynamic allocation.
    # TODO(tianhao909):  max_request_tokens ，
    def _get_kv_cache_memory_per_layer_per_request(self) -> int:
        """Calculate KV cache memory per layer per request (bytes).
         KV Cache （）"""
        # Currently only DeepSeek-671B uses MLA KV cache
        #  DeepSeek-671B  MLA KV Cache
        if self._replica_config.model_name in ['deepseek-671B']:
            kvcache_size_all_layers_per_token = self.get_mla_kvcache_size(self._replica_config.model_config, self.use_fp8)
            kvcache_size_per_layer_per_token = kvcache_size_all_layers_per_token // self._replica_config.model_config.num_hidden_layers
            prefill_input_seq_len = 1
            decode_output_seq_len = 1
            kvcache_size_per_layer_per_req = kvcache_size_per_layer_per_token * (prefill_input_seq_len + decode_output_seq_len)
            return kvcache_size_per_layer_per_req
        elif self._replica_config.model_name in ['qwen3-moe-235B', 'qwen3-next-80B']:
            kvcache_size_all_layers_per_token = self.get_mha_kvcache_size(self._replica_config.model_config, self.use_fp8)
            kvcache_size_per_layer_per_token = kvcache_size_all_layers_per_token // self._replica_config.model_config.num_hidden_layers
            prefill_input_seq_len = 1
            decode_output_seq_len = 1
            kvcache_size_per_layer_per_req = kvcache_size_per_layer_per_token * (prefill_input_seq_len + decode_output_seq_len)
            return kvcache_size_per_layer_per_req
        
        else:
            # TP shard fallback / TP 
                # self._kv_heads_per_tensor_parallel_worker = ceil(
                #     self._model_config.num_kv_heads / self._replica_config.tensor_parallel_size
                # )
            
            return (
                2  # 2 bytes per float
                * 2  # one for key, one for value
                * self._replica.attention_head_dim
                * self._replica.kv_heads_per_tensor_parallel_worker
                * self._replica.max_request_tokens
            )

    # TODO(tianhao909): split get_num_parameters_per_device for P and D replicas
    # TODO(tianhao909):  P  D ， get_num_parameters_per_device
    def _get_parameter_memory_per_device(self) -> int:
        """Get model parameter memory per device (bytes or GB for new models).
        （， GB）"""
        # New models return params in GB instead of count
        #  GB 
        if self._replica_config.model_name in ['deepseek-671B', 'qwen3-moe-235B', 'qwen3-next-80B']:
            return self._param_counter.get_num_parameters_per_device()
        else:
            return 2 * self._param_counter.get_num_parameters_per_device()

    def _get_kv_cache_memory_per_device_per_request(self) -> int:
        return (
            self._get_kv_cache_memory_per_layer_per_request() * self._replica.num_layers
        )

    def get_max_batch_size(self) -> int:
        """
        Calculate maximum batch size based on GPU memory budget.
         GPU 

        Formula / :
        1. available_memory = total_GPU_memory * (1 - memory_margin_fraction)
        2. memory_for_kv_cache = available_memory - parameter_memory
        3. number_of_requests = memory_for_kv_cache // kv_cache_per_request

        For PD disaggregation /  PD :
        - Prefill cluster: fewer params (larger EP), more KV cache memory
          Prefill : (EP), KV cache 
        - Decode cluster: more params (smaller EP), less KV cache memory
          Decode : (EP), KV cache 

        Returns:
            int: Maximum concurrent requests / 
        """
        # ===== 1. Calculate available GPU memory / GPU =====
        available_memory_bytes = (
            self._replica.total_memory_gb
            * 1024**3
            * (1 - self._replica.memory_margin_fraction)
        )
        available_memory_gb = available_memory_bytes / (1024**3)
        
        if self._replica_config.model_name in ['deepseek-671B', 'qwen3-moe-235B', 'qwen3-next-80B']:
            # ===== 2. Get model parameter memory (unit: Bytes) =====
            # Returns triple: (total_param_mem, prefill_param_mem, decode_param_mem)
            total_param_mem, prefill_param_mem, decode_param_mem = self._get_parameter_memory_per_device()
            
            # ===== 3. Get per-request KV cache memory / KV cache =====
            kv_cache_per_request = self._get_kv_cache_memory_per_device_per_request()

            # ===== 4. Calculate KV cache available memory =====
            # Note: must use per-phase param memory, not total
            # : , PDEP
            prefill_kv_cache_memory = available_memory_bytes - prefill_param_mem  # Prefill KV cache available
            decode_kv_cache_memory = available_memory_bytes - decode_param_mem    # Decode KV cache available
            
            # ===== 5. Calculate max supported requests =====
            # If KV cache memory is negative, model params exceed GPU memory
            if prefill_kv_cache_memory > 0:
                prefill_num_requests = int(prefill_kv_cache_memory // kv_cache_per_request)
            else:
                prefill_num_requests = 0  # OOM, set to 0
                
            if decode_kv_cache_memory > 0:
                decode_num_requests = int(decode_kv_cache_memory // kv_cache_per_request)
            else:
                decode_num_requests = 0  # OOM, set to 0
            
            # ===== 6.  | Print detailed debug info =====
            logger.info("\n" + "="*80)
            logger.info("[MemoryPlanner] GPU Memory Allocation (GPU):")
            logger.info("="*80)
            logger.info(f"  Total GPU mem (GPU):          {self._replica.total_memory_gb:.2f} GB")
            logger.info(f"  Mem margin ():            {self._replica.memory_margin_fraction*100:.1f}%")
            logger.info(f"  Available mem ():            {available_memory_gb:.2f} GB")
            logger.info("-"*80)
            logger.info(f"  Total param mem ():          {total_param_mem / (1024**3):.2f} GB")
            logger.info(f"  Prefill param mem (Prefill): {prefill_param_mem / (1024**3):.2f} GB")
            logger.info(f"  Decode param mem (Decode):   {decode_param_mem / (1024**3):.2f} GB")
            logger.info("-"*80)
            logger.info(f"  Prefill KV cache avail (Prefill):  {prefill_kv_cache_memory / (1024**3):.2f} GB")
            logger.info(f"  Decode KV cache avail (Decode):   {decode_kv_cache_memory / (1024**3):.2f} GB")
            logger.info(f"  Per-req KV cache (KV cache):   {kv_cache_per_request / (1024**3):.6f} GB")
            logger.info("-"*80)
            logger.info(f"  Prefill max reqs (Prefill): {prefill_num_requests}")
            logger.info(f"  Decode max reqs (Decode):   {decode_num_requests}")
            logger.info("="*80 + "\n")
            
            # ===== 7. OOM check and error handling /  =====
            # Check Prefill cluster memory / Prefill
            if prefill_param_mem > available_memory_bytes:
                logger.error(f"Prefill cluster OOM (Prefill)!")
                logger.error(f"        Param mem needed (): {prefill_param_mem / (1024**3):.2f} GB")
                logger.error(f"        Available mem ():     {available_memory_gb:.2f} GB")
                logger.error(f"        Deficit ():     {(prefill_param_mem - available_memory_bytes) / (1024**3):.2f} GB")
                logger.error(f"[Suggestion] Increase TP/EP, use larger GPU, or enable FP8")
                
            # Check Decode cluster memory / Decode
            if decode_param_mem > available_memory_bytes:
                logger.error(f"Decode cluster OOM (Decode)!")
                logger.error(f"        Param mem needed (): {decode_param_mem / (1024**3):.2f} GB")
                logger.error(f"        Available mem ():     {available_memory_gb:.2f} GB")
                logger.error(f"        Deficit ():     {(decode_param_mem - available_memory_bytes) / (1024**3):.2f} GB")
                logger.error(f"[Suggestion] Increase TP/EP, use larger GPU, or enable FP8")
            
            # Assert: at least one request must fit / : 
            assert prefill_num_requests > 0, (
                f"Prefill cluster OOM! param_mem({prefill_param_mem/(1024**3):.2f}GB) > "
                f"available({available_memory_gb:.2f}GB), increase parallelism or use quantization"
            )
            assert decode_num_requests > 0, (
                f"Decode cluster OOM! param_mem({decode_param_mem/(1024**3):.2f}GB) > "
                f"available({available_memory_gb:.2f}GB), increase parallelism or use quantization"
            )
            
            # Return prefill max requests as system upper bound
            # In PD disaggregation, prefill and decode clusters are scheduled independently,
            # each with its own capacity. We return prefill's max requests here.
            # Note: decode_num_requests may be smaller, but since the two clusters
            # operate independently, each cluster uses its own limit for scheduling.
            # Prefill ()
            # PD，prefill  decode ，。
            # ：decode_num_requests ，，
            # 。
            return int(prefill_num_requests)
        else:
            parameter_memory_per_device  = self._get_parameter_memory_per_device()
            kv_cache_memory_per_device_per_request = (
                self._get_kv_cache_memory_per_device_per_request()
            )

            memory_for_kv_cache = available_memory_bytes - parameter_memory_per_device
            number_of_requests = (
                memory_for_kv_cache // kv_cache_memory_per_device_per_request
            )
                
            logger.debug(f"available_memory={available_memory_gb}(GB) parameter_memory_per_device={parameter_memory_per_device / (1024**3)}(GB) memory_for_kv_cache={memory_for_kv_cache / (1024**3)} GB kv_cache_memory_per_device_per_request={kv_cache_memory_per_device_per_request / (1024**3)}(GB) number_of_requests={number_of_requests}")

            assert (
                number_of_requests > 0
            ), "Not enough memory to store even a single request"
            
            return number_of_requests

    def get_max_request_slots(self) -> int:
        return self.get_max_batch_size() * self._replica.num_pipeline_stages

    def get_kv_cache_available_memory(self) -> int:
        """
        Get actual available memory for KV cache (bytes).
         KV cache （）

        Formula / :
        kv_cache_available = GPU_available_memory - model_param_memory

        Returns:
            int: Available memory for KV cache (bytes) /  KV cache 
        """
        # ===== 1. Calculate available GPU memory (bytes) =====
        available_memory_bytes = (
            self._replica.total_memory_gb
            * 1024**3
            * (1 - self._replica.memory_margin_fraction)
        )
        
        # ===== 2. Get model parameter memory =====
        # Previously used prefill_param_mem for all replicas incorrectly,
        # causing decode replica KV cache available memory calculation error.
        # Now select param memory based on replica type.
        #  replica  prefill_param_mem，
        #  decode replica  KV cache 。
        #  replica 。
        if self._replica_config.model_name in ['deepseek-671B', 'qwen3-moe-235B', 'qwen3-next-80B']:
            from vidur.entities.replica import ReplicaType
            _, prefill_param_mem, decode_param_mem = self._get_parameter_memory_per_device()
            
            if hasattr(self._replica, 'replica_type') and self._replica.replica_type == ReplicaType.DECODE:
                # Decode replica uses decode_param_mem
                # Decode  decode_param_mem
                param_memory = decode_param_mem
                logger.debug(f"get_kv_cache_available_memory: "
                      f"Decode replica uses decode_param_mem={decode_param_mem/(1024**3):.2f} GB "
                      f"(not prefill_param_mem={prefill_param_mem/(1024**3):.2f} GB)")
            else:
                # Prefill/Mixed replica uses prefill_param_mem
                # Prefill/Mixed  prefill_param_mem
                param_memory = prefill_param_mem
                logger.debug(f"get_kv_cache_available_memory: "
                      f"Prefill replica uses prefill_param_mem={prefill_param_mem/(1024**3):.2f} GB")
        else:
            param_memory = self._get_parameter_memory_per_device()
        
        # ===== 3. Calculate available KV cache memory =====
        kv_cache_available = available_memory_bytes - param_memory
        
        # Ensure non-negative / 
        if kv_cache_available < 0:
            logger.warning(f"KV cache available is negative! param_mem({param_memory/(1024**3):.2f}GB) > available({available_memory_bytes/(1024**3):.2f}GB)")
            assert kv_cache_available >= 0, f"kv_cache_available={kv_cache_available} must be >= 0"
            kv_cache_available = 0
        
        logger.info(f"[MemoryPlanner] Real KV cache available (KV cache): {kv_cache_available/(1024**3):.2f} GB")
        return int(kv_cache_available)
