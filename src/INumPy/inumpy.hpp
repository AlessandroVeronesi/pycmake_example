#ifndef __INUMPY_HPP__
#define __INUMPY_HPP__

#include <vector>

namespace INumPy {

namespace priv {

template <typename T, typename Tp>
int _PyArray_to_1Dvector_i(PyArrayObject* inArray, std::vector<T>& out_vec);

template <typename T, typename Tp>
int _PyArray_to_2Dvector_i(PyArrayObject* inArray, std::vector<std::vector<T> >& out_mat);

template <typename T, typename Tp>
int _PyArray_to_3Dvector_i(PyArrayObject* inArray, std::vector<std::vector<std::vector<T> > >& out_ten);

template <typename T, typename Tp>
int _PyArray_to_4Dvector_i(PyArrayObject* inArray, std::vector<std::vector<std::vector<std::vector<T> > > > & out_ten);

template <typename T, typename Tp>
int _PyArray_to_5Dvector_i(PyArrayObject* inArray, std::vector<std::vector<std::vector<std::vector<std::vector<T> > > > > & out_ten);

template <typename T, typename Tp>
int _1Dvector_to_PyArray_i(std::vector<Tp>& in_vec, PyArrayObject** outArray_ptr);

template <typename T, typename Tp>
int _2Dvector_to_PyArray_i(std::vector<std::vector<Tp> >& in_mat, PyArrayObject** outArray_ptr);

template <typename T, typename Tp>
int _3Dvector_to_PyArray_i(std::vector<std::vector<std::vector<Tp> > >& in_ten, PyArrayObject** outArray_ptr);

template <typename T, typename Tp>
int _4Dvector_to_PyArray_i(std::vector<std::vector<std::vector<std::vector<Tp> > > >& in_ten, PyArrayObject** outArray_ptr);

template <typename T, typename Tp>
int _5Dvector_to_PyArray_i(std::vector<std::vector<std::vector<std::vector<std::vector<Tp> > > > >& in_ten, PyArrayObject** outArray_ptr);

} // priv

// PyArray Parameters Check
int _isIntPyArray(PyArrayObject* testArray);
int _isFloatPyArray(PyArrayObject* testArray);

// PyArray to 1D C-array Conversion
template <typename T>
constexpr int _PyArray_to_1Dvector(PyArrayObject* inArray, std::vector<T>& out_vec);

template <typename T>
int _1Dvector_to_PyArray(std::vector<T>& in_vec, PyArrayObject** outArray_ptr);

// PyArray to 2D C-array Conversion
template <typename T>
constexpr int _PyArray_to_2Dvector(PyArrayObject* inArray, std::vector<std::vector<T> >& out_mat);

template <typename T>
int _2Dvector_to_PyArray(std::vector<std::vector<T> >& in_mat, PyArrayObject** outArray_ptr);

// PyArray to 3D C-array Conversion
template <typename T>
constexpr int _PyArray_to_3Dvector(PyArrayObject* inArray, std::vector<std::vector<std::vector<T> > >& out_ten);

template <typename T>
int _3Dvector_to_PyArray(std::vector<std::vector<std::vector<T> > >& in_ten, PyArrayObject** outArray_ptr);

// PyArray to 4D C-array Conversion
template <typename T>
constexpr int _PyArray_to_4Dvector(PyArrayObject* inArray, std::vector<std::vector<std::vector<std::vector<T> > > >& out_ten);

template <typename T>
int _4Dvector_to_PyArray(std::vector<std::vector<std::vector<std::vector<T> > > >& in_ten, PyArrayObject** outArray_ptr);

// PyArray to 5D C-array Conversion
template <typename T>
constexpr int _PyArray_to_5Dvector(PyArrayObject* inArray, std::vector<std::vector<std::vector<std::vector<std::vector<T> > > > >& out_ten);

template <typename T>
int _5Dvector_to_PyArray(std::vector<std::vector<std::vector<std::vector<std::vector<T> > > > >& in_ten, PyArrayObject** outArray_ptr);

} // INumPy

#include "tpp/inumpy.tpp"

#endif