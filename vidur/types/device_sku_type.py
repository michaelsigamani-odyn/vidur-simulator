from vidur.types.base_int_enum import BaseIntEnum


class DeviceSKUType(BaseIntEnum):
    A40 = 1
    A100 = 2
    H100 = 3
    H800 = 4
    H20 = 5
    MI300X = 6
    RADEON_PRO_W7900 = 7
    GB10 = 8
    # NOTE: untested, for reference only (，)
    # H200 = 6
    # GB200 = 7
