from dataclasses import dataclass, field

from vidur.config.base_fixed_config import BaseFixedConfig
from vidur.logger import init_logger
from vidur.types import DeviceSKUType

logger = init_logger(__name__)


@dataclass
class BaseDeviceSKUConfig(BaseFixedConfig):
    fp16_tflops: int
    total_memory_gb: int
    hourly_cost_usd: float


@dataclass
class A40DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 150
    total_memory_gb: int = 45
    hourly_cost_usd: float = 1.28

    @staticmethod
    def get_type():
        return DeviceSKUType.A40


@dataclass
class A100DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 312
    total_memory_gb: int = 80
    hourly_cost_usd: float = 2.21

    @staticmethod
    def get_type():
        return DeviceSKUType.A100

@dataclass
class H20DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 148
    fp8_tflops: int = 296
    total_memory_gb: int = 141
    hourly_cost_usd: float = 2.60

    @staticmethod
    def get_type():
        return DeviceSKUType.H20


@dataclass
class H100DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 1000
    total_memory_gb: int = 80
    hourly_cost_usd: float = 4.25

    @staticmethod
    def get_type():
        return DeviceSKUType.H100

@dataclass
class H800DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 989
    fp8_tflops: int = 1979
    total_memory_gb: int = 80
    hourly_cost_usd: float = 4.10

    @staticmethod
    def get_type():
        return DeviceSKUType.H800


@dataclass
class MI300XDeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 1307
    total_memory_gb: int = 192
    hourly_cost_usd: float = 5.95

    @staticmethod
    def get_type():
        return DeviceSKUType.MI300X


@dataclass
class RadeonProW7900DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 122
    total_memory_gb: int = 48
    hourly_cost_usd: float = 1.45

    @staticmethod
    def get_type():
        return DeviceSKUType.RADEON_PRO_W7900


@dataclass
class GB10DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 1000
    total_memory_gb: int = 128
    hourly_cost_usd: float = 4.25

    @staticmethod
    def get_type():
        return DeviceSKUType.GB10
