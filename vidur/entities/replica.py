from math import ceil
from typing import Tuple

from vidur.config import BaseRequestGeneratorConfig, ReplicaConfig
from vidur.entities.base_entity import BaseEntity
from vidur.logger import init_logger

logger = init_logger(__name__)

# >
import bisect
from enum import IntEnum  
class ReplicaType(IntEnum):  # Define task type enumeration class, inheriting from IntEnum
    MIXED = 0 # Mixed, no distinction
    PREFILL = 1  # Prompt task (prefill stage)
    DECODE = 2  # Token task (generation stage)


# Replica represents a model entity, which is a Data Parallelism (DP) unit
# Replica，DP
class Replica(BaseEntity):
    def __init__(
        self,
        replica_config: ReplicaConfig,
        generator_config: BaseRequestGeneratorConfig,
    ) -> None:
        self._id = Replica.generate_id()

        self._replica_config = replica_config
        self._model_config = replica_config.model_config
        self._device_config = replica_config.device_config
        self._generator_config = generator_config

        assert (
            self._model_config.num_layers % self._replica_config.num_pipeline_stages
            == 0
        )
        assert (
            self._model_config.embedding_dim % self._replica_config.tensor_parallel_size
            == 0
        )
        
        # TODO(tianhao909): decouple pending_requests from replica
        # TODO(tianhao909):  pending_requests  replica 
        # self._pending_requests = []
        self.pending_requests = []
        self._pending_tasks = []
        # Scheduler metadata / 
        # self.sched_memory = self.model.size.total_size  # Memory usage from scheduler's perspective
        self.sched_memory = self._device_config.total_memory_gb
        self.sched_pending_tokens = 0  # Number of pending tokens from scheduler's perspective
        self.sched_tag = None  # Scheduler tag
        # Separate pending queue for prompt tasks (to prioritize prompts)
        self.pending_prompt_queue = []
        # Map requests->tasks on this instance
        self.request_tasks = {}
        self.replica_type = ReplicaType.MIXED
        
        # >
        self.pd_p2p_comm_bandwidth = self._replica_config.pd_p2p_comm_bandwidth
        self.pd_p2p_comm_dtype = self._replica_config.pd_p2p_comm_dtype
        self.pd_node_ratio = self._replica_config.pd_node_ratio
        self.nvlink_bandwidth = self._replica_config.nvlink_bandwidth
        self.rdma_bandwidth = self._replica_config.rdma_bandwidth

        # New variables: track KV cache memory usage
        # ：kvcache
        self._allocated_kv_cache_memory = 0  # Allocated KV cache memory (bytes) / kvcache
        self._max_kv_cache_memory = None  # Max KV cache capacity (bytes) / kvcache
        self._kv_cache_allocation_map = {}  # Track per-request KV cache allocation / kvcache
        
        
    @property
    def id(self) -> int:
        return self._id

    @property
    def num_layers(self) -> int:
        return self._model_config.num_layers

    @property
    def num_q_heads(self) -> int:
        return self._model_config.num_q_heads

    @property
    def num_kv_heads(self) -> int:
        return self._model_config.num_kv_heads

    @property
    def embedding_dim(self) -> int:
        return self._model_config.embedding_dim

    @property
    def mlp_hidden_dim(self) -> int:
        return self._model_config.mlp_hidden_dim

    @property
    def use_gated_mlp(self) -> int:
        return self._model_config.use_gated_mlp

    @property
    def vocab_size(self) -> int:
        return self._model_config.vocab_size

    @property
    def num_pipeline_stages(self) -> int:
        return self._replica_config.num_pipeline_stages

    @property
    def num_layers_per_pipeline_stage(self) -> int:
        return self._model_config.num_layers // self._replica_config.num_pipeline_stages

    @property
    def attention_head_dim(self) -> int:
        return self._model_config.embedding_dim // self._model_config.num_q_heads

    @property
    def q_heads_per_tensor_parallel_worker(self) -> int:
        return (
            self._model_config.num_q_heads // self._replica_config.tensor_parallel_size
        )

    @property
    def kv_heads_per_tensor_parallel_worker(self) -> int:
        return ceil(
            self._model_config.num_kv_heads / self._replica_config.tensor_parallel_size
        )

    @property
    def num_tensor_parallel_workers(self) -> int:
        return self._replica_config.tensor_parallel_size

    @property
    def total_memory_gb(self) -> int:
        return self._device_config.total_memory_gb

    @property
    def memory_margin_fraction(self) -> float:
        return self._replica_config.memory_margin_fraction

    @property
    def max_request_tokens(self) -> int:
        return self._generator_config.max_tokens

    @property
    def per_device_flops(self) -> float:
        return self._device_config.fp16_tflops * 2**40
    
    # > sw
    # @property
    # def pending_requests(self) -> list:
    #     return self._pending_requests

    @property
    def pending_tasks(self) -> list:
        return self._pending_tasks

    def get_kv_cache_per_token(self) -> int:
        """
        Calculate per-token KV Cache size (unit: Bytes).
        tokenKV Cache (: Bytes)
        
        Formula / : 2 * num_kv_heads * head_dim * num_layers * bytes_per_element
        
        Returns:
            int: Per-token KV Cache size (Bytes)
        """
        # Determine bytes per element / 
        dtype_to_bytes = {
            'float16': 2, 'bfloat16': 2,
            'float32': 4, 'float64': 8,
            'fp8': 1, 'int8': 1,
            'int16': 2, 'int32': 4, 'int64': 8
        }
        bytes_per_element = dtype_to_bytes.get(self.pd_p2p_comm_dtype, 2)
        
        # KV Cache size per token / KV Cache token
        kv_cache_per_token = (
            2                        # KV
            * self.num_kv_heads      # KV heads
            * self.attention_head_dim  # head
            * self.num_layers        # 
            * bytes_per_element      # 
        )
        return kv_cache_per_token

    def get_remaining_kv_cache_capacity(self, avg_tokens_per_request=None) -> Tuple[int, int]:
        """
        Calculate remaining KV cache memory capacity and how many requests it can serve.
        kvcache，request
        
        Args:
            avg_tokens_per_request: Avg tokens per request (default: max_request_tokens)
                token
        
        Returns:
            (remaining_kv_cache_bytes, remaining_request_capacity)
        """
        from vidur.scheduler.utils.memory_planner import MemoryPlanner
        memory_planner = MemoryPlanner(self._replica_config, self)

        # ===== 1. Init max KV cache capacity (computed on first call) =====
        # ===== 1. kvcache () =====
        if self._max_kv_cache_memory is None:
            # Get real KV cache available memory (bytes) from memory_planner
            #  memory_planner  KV cache  (bytes)
            # Correct calculation: available memory - model parameter memory
            # :  - 
            self._max_kv_cache_memory = memory_planner.get_kv_cache_available_memory()
            
            # Compute per-request KV cache for display
            #  KV cache 
            kv_cache_per_token = self.get_kv_cache_per_token()
            tokens_per_req = avg_tokens_per_request or self.max_request_tokens
            kv_cache_per_request = kv_cache_per_token * tokens_per_req
            max_requests = int(self._max_kv_cache_memory / kv_cache_per_request) if kv_cache_per_request > 0 else 0
            
            logger.info(f"[Replica] KV Cache Capacity Init (KV Cache):")
            logger.info(f"  Total GPU mem (GPU): {self.total_memory_gb:.2f} GB")
            logger.info(f"  Mem margin (): {self.memory_margin_fraction*100:.1f}%")
            logger.info(f"  Max KV cache capacity (KV cache): {self._max_kv_cache_memory/(1024**3):.2f} GB")
            logger.info(f"  KV cache per token (token KV cache): {kv_cache_per_token} bytes = {kv_cache_per_token/1024:.2f} KB")
            logger.info(f"  Avg tokens per req (token): {tokens_per_req}")
            logger.info(f"  KV cache per req (KV cache): {kv_cache_per_request/(1024**3):.4f} GB")
            logger.info(f"  Max servable reqs (): {max_requests}")

        # ===== 2. Compute remaining KV cache memory =====
        # ===== 2. kvcache =====
        remaining_kv_cache = self._max_kv_cache_memory - self._allocated_kv_cache_memory

        # ===== 3. Compute remaining request capacity =====
        # ===== 3.  =====
        # Unified calculation: per-token KV cache * avg tokens per request
        # 
        kv_cache_per_token = self.get_kv_cache_per_token()
        tokens_per_req = avg_tokens_per_request or self.max_request_tokens
        kv_cache_per_request = kv_cache_per_token * tokens_per_req
        
        if kv_cache_per_request > 0:
            remaining_request_capacity = int(remaining_kv_cache / kv_cache_per_request)
        else:
            remaining_request_capacity = 0

        # ===== 4. Print debug info =====
        # ===== 4.  =====
        logger.debug(f"Remaining KV cache: {remaining_kv_cache / (1024**3):.2f} GB ({remaining_kv_cache / (1024**2):.2f} MB)")
        logger.debug(f"Per-request KV cache: {kv_cache_per_request/(1024**3):.4f} GB ({tokens_per_req} tokens)")
        logger.debug(f"Remaining request capacity: {remaining_request_capacity}")

        return remaining_kv_cache, remaining_request_capacity

    def release_request_kv_cache_memory(self, request) -> None:
        """
        Release KV cache memory occupied by the specified request.
        requestkvcache
        """
        # Get KV cache size occupied by this request from allocation map
        # requestkvcache
        if request.id in self._kv_cache_allocation_map:
            kv_cache_size = self._kv_cache_allocation_map[request.id]
            assert kv_cache_size > 0, f"fth debug: request {request.id} kv cache size should be positive"
                   
            # Subtract this request's KV cache from allocated total
            # kvcacherequest
            self._allocated_kv_cache_memory = max(0, self._allocated_kv_cache_memory - kv_cache_size)
            
            # Remove this request from allocation map
            # 
            del self._kv_cache_allocation_map[request.id]

            logger.debug(f"Released KV cache for request {request.id}: {kv_cache_size / (1024**3):.2f} GB ({kv_cache_size / (1024**2):.2f} MB)")
        else:
            logger.warning(f"Request {request.id} not found in KV cache allocation map")

    # def allocate_request_kv_cache_memory(self, request, num_blocks):
    #     """
    #     requestkvcache，_allocation_map
    #     kvcache
    #     """
    #     # requestkvcache
    #     kv_cache_size = request.estimate_kv_cache_size(num_blocks, self)
    
    def allocate_request_kv_cache_memory(self, request, num_blocks, block_size) -> None:
        """
        Allocate KV cache memory for a request, tracking per-request allocation.
        requestkvcache，
        
        Previously num_blocks was passed directly as num_tokens,
        causing KV cache tracking to be underestimated by block_size times.
        Now correctly converts num_blocks * block_size to num_tokens.
         num_blocks  num_tokens ， block_size 。
        
        Args:
            request: Request object / 
            num_blocks: Number of allocated memory blocks / 
            block_size: Tokens per block / token
        """
        # Correct conversion: num_tokens = num_blocks * block_size
        # 
        num_tokens = num_blocks * block_size
        kv_cache_size = request.estimate_kv_cache_size(num_tokens, self)
        logger.debug(f"allocate_request_kv_cache_memory: "
                    f"req={request.id}, num_blocks={num_blocks}, block_size={block_size}, "
                    f"num_tokens={num_tokens}, kv_cache_size={kv_cache_size/(1024**2):.2f} MB")

        # Update allocation map / 
        if request.id not in self._kv_cache_allocation_map:
            self._kv_cache_allocation_map[request.id] = kv_cache_size
        else:
            # If already allocated, accumulate (for incremental allocation)
            # ，
            self._kv_cache_allocation_map[request.id] += kv_cache_size

        # Increase allocated KV cache / kvcache
        self._allocated_kv_cache_memory += kv_cache_size

        logger.debug(f"Allocated KV cache for request {request.id}: {kv_cache_size / (1024**3):.2f} GB, "
                    f"total allocated: {self._allocated_kv_cache_memory / (1024**3):.2f} GB")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "num_layers": self.num_layers,
            "num_q_heads": self.num_q_heads,
            "num_kv_heads": self.num_kv_heads,
            "embedding_dim": self.embedding_dim,
            "mlp_hidden_dim": self.mlp_hidden_dim,
            "use_gated_mlp": self.use_gated_mlp,
            "vocab_size": self.vocab_size,
            "num_pipeline_stages": self.num_pipeline_stages,
            "num_tensor_parallel_workers": self.num_tensor_parallel_workers,
        }


    def add_to_pool(self, task) -> None:
        """
        Add a Task to the request pool.
        Request pool is ordered by request arrival time.
        """
        
        # bisect.insort(): Uses binary search algorithm to insert element into sorted list, maintaining list's sorted state
        # self.pending_requests: Target list storing all pending requests
        # task.request: Request object to be inserted
        # key=lambda x: x.arrival_timestamp: Sort key function, sorting by request arrival time
        # lambda x: x.arrival_timestamp is an anonymous function that accepts a parameter x (request object) and returns its arrival_timestamp attribute
        # This ensures the pending_requests list is always sorted by request arrival time
        if task.request not in self.pending_requests:  # If request is not in current pool
            # bisect.insort(self.pending_requests, task.request,  # # Insert sort, insert by arrival time
            #               key=lambda x: x.arrival_timestamp)
            # arrived_at
            bisect.insort(self.pending_requests, task.request,  # # Insert sort, insert by arrival time
                          key=lambda x: x.arrived_at)
            self.request_tasks[task.request] = [task]  # Create task list for new request
        else:
            self.request_tasks[task.request].append(task)  # Otherwise append task