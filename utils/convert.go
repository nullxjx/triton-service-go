package utils

import (
	"bytes"
	"encoding/binary"
)

// SliceToBytes 模仿python的numpy.ndarray.tobytes()函数。
// 参考 https://numpy.org/doc/stable/reference/generated/numpy.ndarray.tobytes.html 。
func SliceToBytes[T int32 | uint32 | float32 | uint64 | bool | string | byte](array []T) []byte {
	var buf bytes.Buffer
	for i := range array {
		_ = binary.Write(&buf, binary.LittleEndian, array[i])
	}
	return buf.Bytes()
}

func Slice2DToBytes[T int32 | uint32 | float32 | uint64 | bool](array [][]T) []byte {
	var buf bytes.Buffer
	for i := range array {
		for j := range array[i] {
			_ = binary.Write(&buf, binary.LittleEndian, array[i][j])
		}
	}
	return buf.Bytes()
}

// BytesToSlice 是sliceToBytes的逆向函数。因为要判断类型T的字节步长比较麻烦，这个函数当强仅仅能支持同样是步长为4的类型。
func BytesToSlice[T uint32](bs []byte) []T {
	r := bytes.NewReader(bs)
	s := make([]T, len(bs)/4)
	for i := range s {
		_ = binary.Read(r, binary.LittleEndian, &s[i])

	}
	return s
}

func Pad(slice []int32, padValue int32, length int) []int32 {
	if len(slice) >= length {
		return slice
	}
	padded := make([]int32, length)
	copy(padded, slice)
	for i := len(slice); i < length; i++ {
		padded[i] = padValue
	}
	return padded
}
