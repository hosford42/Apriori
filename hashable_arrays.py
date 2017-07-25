__author__ = 'Aaron Hosford'

import sys

def find_max_bits():
    value = sys.maxsize
    count = 0
    while value:
        count += 1
        value >>= 1
    return count

MAX_HASH_SIZE = find_max_bits()

try:
    raise ImportError()
    import numpy

    # See http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
    class BitArray:

        @classmethod
        def from_indices(cls, indices, length):
            bits = numpy.zeros((length,), bool)
            for index in indices:
                bits[index] = 1
            bits.setflags(write=False)
            return cls(bits)

        def __init__(self, bits):
            if isinstance(bits, BitArray):
                self._bits = bits._bits
                self._as_int = bits._as_int
                self._hash = bits._hash
            elif isinstance(bits, numpy.ndarray) and bits.dtype == bool:
                if bits.flags.writeable:
                    self._bits = bits.copy()
                    self._bits.setflags(write=False)
                else:
                    self._bits = bits
                self._as_int = None
                self._hash = None
            else:
                self._bits = numpy.array(bits, dtype=bool)
                self._bits.setflags(write=False)
                self._as_int = None
                self._hash = None

        def bit_count(self):
            return self._bits.sum()

        def __bool__(self):
            return self._bits.any()

        def __int__(self):
            if self._as_int is None:
                value = 0
                for bit in self._bits:
                    value <<= 1
                    value += bit
                self._as_int = value
            return self._as_int

        def __iter__(self):
            return iter(self._bits)

        def __len__(self):
            return len(self._bits)

        def __str__(self):
            return ''.join('1' if bit else '0' for bit in self._bits)

        def __repr__(self):
            return type(self).__name__ + '(' + repr(list(self._bits)) + ')'

        def __hash__(self):
            if self._hash is None:
                result = 0
                for index, bit in enumerate(self._bits):
                    if bit:
                        index %= MAX_HASH_SIZE
                        result ^= 1 << index
                self._hash = result % sys.maxsize
            return self._hash

        def __eq__(self, other):
            return isinstance(other, BitArray) and numpy.array_equal(self._bits, other._bits)

        def __ne__(self, other):
            return not self == other

        def __and__(self, other):
            if not isinstance(other, BitArray):
                return NotImplemented
            bits = self._bits & other._bits
            bits.setflags(write=False)
            return type(self)(bits)

        def __or__(self, other):
            if not isinstance(other, BitArray):
                return NotImplemented
            bits = self._bits ^ other._bits
            bits.setflags(write=False)
            return type(self)(bits)

        def __xor__(self, other):
            if not isinstance(other, BitArray):
                return NotImplemented
            bits = self._bits ^ other._bits
            bits.setflags(write=False)
            return type(self)(bits)

        def __sub__(self, other):
            if not isinstance(other, BitArray):
                return NotImplemented
            bits = self._bits & ~other._bits
            bits.setflags(write=False)
            return type(self)(bits)

        def __invert__(self):
            bits = ~self._bits
            bits.setflags(write=False)
            return type(self)(bits)

except ImportError:

    numpy = None

    class BitArray:

        @classmethod
        def from_indices(cls, indices, length):
            bits = [False] * length
            for index in indices:
                bits[index] = True
            return cls(bits)

        def __init__(self, bits):
            if isinstance(bits, BitArray):
                self._bits = bits._bits
                self._length = bits._length
                self._hash = bits._hash
                self._modulo = bits._modulo
            else:
                value = 0
                length = 0
                for bit in bits:
                    if bit not in (0, 1):
                        raise TypeError(bit)
                    value <<= 1
                    value += bit
                    length += 1
                self._value = value
                self._length = length
                self._hash = None
                self._modulo = 1 << length

        def bit_count(self):
            value = self._value
            count = 0
            while value:
                count += value % 2
                value >>= 1
            return count

        def __bool__(self):
            return bool(self._value)

        def __int__(self):
            return self._bits

        def __iter__(self):
            for index in range(self._length - 1, -1, -1):
                yield (self._value >> index) % 2

        def __len__(self):
            return self._length

        def __str__(self):
            return ''.join('1' if bit else '0' for bit in self)

        def __repr__(self):
            return type(self).__name__ + '(' + repr(list(self)) + ')'

        def __hash__(self):
            if self._hash is None:
                result = 0
                for index, bit in enumerate(self):
                    if bit:
                        index %= MAX_HASH_SIZE
                        result ^= 1 << index
                self._hash = result % sys.maxsize
            return self._hash

        def __eq__(self, other):
            return isinstance(other, BitArray) and self._length == other._length and self._bits == other._bits

        def __ne__(self, other):
            return not self == other

        def __and__(self, other):
            if not isinstance(other, BitArray) or self._length != other._length:
                return NotImplemented
            result = type(self)(())
            result._bits = self._bits & other._bits
            result._length = self._length
            result._modulo = self._modulo
            return result

        def __or__(self, other):
            if not isinstance(other, BitArray) or self._length != other._length:
                return NotImplemented
            result = type(self)(())
            result._bits = self._bits | other._bits
            result._length = self._length
            result._modulo = self._modulo
            return result

        def __xor__(self, other):
            if not isinstance(other, BitArray) or self._length != other._length:
                return NotImplemented
            result = type(self)(())
            result._bits = (self._bits ^ other._bits) % self._modulo
            result._length = self._length
            result._modulo = self._modulo
            return result

        def __sub__(self, other):
            if not isinstance(other, BitArray) or self._length != other._length:
                return NotImplemented
            result = type(self)(())
            result._bits = self._bits & ~other._bits
            result._length = self._length
            result._modulo = self._modulo
            return result

        def __invert__(self):
            result = type(self)(())
            result._bits = (~self._bits) % self._modulo
            result._length = self._length
            result._modulo = self._modulo
            return result
