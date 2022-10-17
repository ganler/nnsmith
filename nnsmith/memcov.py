import ctypes


class Memcov:
    def __init__(self, mcov_lib: ctypes.CDLL):
        self.get_hitbits = mcov_lib.mcov_get_hitbits
        self.set_hitbits = mcov_lib.mcov_set_hitbits
        self.reset_bitmap = mcov_lib.mcov_reset_bitmap
        self.get_bitmap_bytes = mcov_lib.mcov_get_bitmap_bytes

        self._char_array = ctypes.c_char * self.get_bitmap_bytes()
        self._mcov_copy_bitmap = mcov_lib.mcov_copy_bitmap
        self._mcov_set_bitmap = mcov_lib.mcov_set_bitmap

    def get_hitmap_buffer(self) -> bytes:
        hitmap_buffer = bytearray(self.get_bitmap_bytes())
        self._mcov_copy_bitmap(self._char_array.from_buffer(hitmap_buffer))
        return hitmap_buffer

    def set_hitmap_buffer(self, data: bytes):
        assert len(data) == self.get_bitmap_bytes()
        self._mcov_set_bitmap(self._char_array.from_buffer(data))
