#ifndef __INUMPY_TPP__
#define __INUMPY_TPP__

//* ======================= *//
// Internal: Check Array Type as Int
int INumPy::_isIntPyArray(PyArrayObject* testArray)
{
    if(
        (PyArray_TYPE(testArray) != NPY_INT64) &&
        (PyArray_TYPE(testArray) != NPY_INT32) &&
        (PyArray_TYPE(testArray) != NPY_INT16) &&
        (PyArray_TYPE(testArray) != NPY_INT8) 
        )
        return 0;

    return 1;
}

// Internal: Check Array Type as Float
int INumPy::_isFloatPyArray(PyArrayObject* testArray)
{
    if(
        (PyArray_TYPE(testArray) != NPY_FLOAT32) &&
        (PyArray_TYPE(testArray) != NPY_FLOAT64)
        )
        return 0;

    return 1;
}


//* ======================= *//
// Copy PyArray to C++ 1D Vector

template <typename T, typename Tp>
int INumPy::priv::_PyArray_to_1Dvector_i(PyArrayObject* inArray, std::vector<T>& out_vec)
{
    if(PyArray_NDIM(inArray) != 1) return 0;

    npy_intp* size = PyArray_SHAPE(inArray);
    Tp* base_addr = (Tp*) PyArray_DATA(inArray);
    npy_intp* stride = PyArray_STRIDES(inArray);
    size_t data_stride = (size_t) (stride[0]/sizeof(int));

    out_vec.clear();

    for(size_t i=0; i<(size_t)size[0]; i++) {
        out_vec.push_back(static_cast<T>(*(base_addr + (i*data_stride))));
    }

    return 1;
}

template <typename T>
constexpr int INumPy::_PyArray_to_1Dvector(PyArrayObject* inArray, std::vector<T> & out_mat) {
    return (std::is_integral<T>())? INumPy::priv::_PyArray_to_1Dvector_i<T, int>(inArray, out_mat) : INumPy::priv::_PyArray_to_1Dvector_i<T, float>(inArray, out_mat);
}

//* ======================= *//
// Copy C++ 1D Vector to PyArray
template <typename T, typename Tp>
int INumPy::priv::_1Dvector_to_PyArray_i(std::vector<Tp>& in_vec, PyArrayObject** outArray_ptr)
{
    int nd = 1;
    int size = in_vec.size();
    npy_intp dims[1];
    dims[0] = size;
    T *data;
    PyArrayObject *_PyTemp;

    // Copy C++ 1D Vector to the Heap
    data = (T*) malloc(size * sizeof(T));

    for(size_t i=0; i < size; i++)
        data[i] = static_cast<T>(in_vec[i]);

    // Create a PyArray wrapping data
    NPY_TYPES myNpyType;

    if constexpr(std::is_same<T, char>::value)
        myNpyType = NPY_INT8;
    else if constexpr(std::is_same<T, int>::value)
        myNpyType = NPY_INT32;
    else if constexpr(std::is_same<T, long int>::value)
        myNpyType = NPY_INT64;
    else if constexpr(std::is_same<T, float>::value)
        myNpyType = NPY_FLOAT32;
    else if constexpr(std::is_same<T, double>::value)
        myNpyType = NPY_FLOAT64;
    else {
        std::cerr << "ERROR: Not recognized data type" << std::endl;
        return 0;
    }

    _PyTemp = (PyArrayObject*) PyArray_SimpleNewFromData(nd, dims, myNpyType, data);
    if(!_PyTemp)
        return 0;

    // Pass PyArray pointer to the output
    *outArray_ptr = _PyTemp;

    return 1;
}


template <typename T>
int INumPy::_1Dvector_to_PyArray(std::vector<T>& in_vec, PyArrayObject** outArray_ptr) {
    if constexpr (std::is_floating_point<T>::value) {
        return INumPy::priv::_1Dvector_to_PyArray_i<float, T>(in_vec, outArray_ptr);
    }
    else {
        return INumPy::priv::_1Dvector_to_PyArray_i<T, T>(in_vec, outArray_ptr);
    }
}


//* ======================= *//
// Copy PyArray to C++ 2D Vector
template <typename T, typename Tp>
int INumPy::priv::_PyArray_to_2Dvector_i(PyArrayObject* inArray, std::vector<std::vector<T> >& out_mat)
{
    if(PyArray_NDIM(inArray) != 2) return 0;

    npy_intp* size = PyArray_SHAPE(inArray);

    Tp* base_addr = (Tp*) PyArray_DATA(inArray);

    npy_intp* stride = PyArray_STRIDES(inArray);
    size_t line_stride    = (size_t) (stride[0]/sizeof(int));
    size_t data_stride = (size_t) (stride[1]/sizeof(int));

    out_mat.clear();

    for(size_t j=0; j<(size_t)size[0]; j++) {
        std::vector<T> temp_vec;

        for(size_t i=0; i<(size_t)size[1]; i++) {
            temp_vec.push_back(static_cast<T>(*(base_addr + (j*line_stride + i*data_stride))));
        }

        out_mat.push_back(temp_vec);
    }

    return 1;
}

template <typename T>
constexpr int INumPy::_PyArray_to_2Dvector(PyArrayObject* inArray, std::vector<std::vector<T> >& out_mat) {
    return (std::is_integral<T>())? INumPy::priv::_PyArray_to_2Dvector_i<T, int>(inArray, out_mat) : INumPy::priv::_PyArray_to_2Dvector_i<T, float>(inArray, out_mat);
}

//* ======================= *//
// Copy C++ 2D Vector to PyArray
template <typename T, typename Tp>
int INumPy::priv::_2Dvector_to_PyArray_i(std::vector<std::vector<Tp> >& in_mat, PyArrayObject** outArray_ptr)
{
    int nd = 2;
    npy_intp dims[2];
    dims[0] = in_mat.size();
    dims[1] = in_mat[0].size();
    T *data;
    PyArrayObject *_PyTemp;

    // Copy C++ 2D Vector to the Heap
    data = (T*) malloc(in_mat.size() * in_mat[0].size() * sizeof(T));

    for(size_t i=0; i<in_mat.size(); i++)
        for(size_t j=0; j<in_mat[0].size(); j++)
            data[i*in_mat.size() + j] = (T) in_mat[i][j];

    // Create a PyArray wrapping data
    NPY_TYPES myNpyType;

    if constexpr(std::is_same<T, char>::value)
        myNpyType = NPY_INT8;
    else if constexpr(std::is_same<T, int>::value)
        myNpyType = NPY_INT32;
    else if constexpr(std::is_same<T, long int>::value)
        myNpyType = NPY_INT64;
    else if constexpr(std::is_same<T, float>::value)
        myNpyType = NPY_FLOAT32;
    else if constexpr(std::is_same<T, double>::value)
        myNpyType = NPY_FLOAT64;
    else {
        std::cerr << "ERROR: Not recognized data type" << std::endl;
        return 0;
    }

    _PyTemp = (PyArrayObject*) PyArray_SimpleNewFromData(nd, dims, myNpyType, data);
    if(!_PyTemp)
        return 0;

    // Pass PyArray pointer to the output
    *outArray_ptr = _PyTemp;

    return 1;
}


template <typename T>
int INumPy::_2Dvector_to_PyArray(std::vector<std::vector<T> >& in_mat, PyArrayObject** outArray_ptr) {
    if constexpr (std::is_floating_point<T>::value) {
        return INumPy::priv::_2Dvector_to_PyArray_i<float, T>(in_mat, outArray_ptr);
    }
    else {
        return INumPy::priv::_2Dvector_to_PyArray_i<T, T>(in_mat, outArray_ptr);
    }
}


//* ======================= *//
// Copy PyArray to C++ 3D Vector
template <typename T, typename Tp>
int INumPy::priv::_PyArray_to_3Dvector_i(PyArrayObject* inArray, std::vector<std::vector<std::vector<T> > >& out_ten)
{
    if(PyArray_NDIM(inArray) != 3) return 0;

    npy_intp* size = PyArray_SHAPE(inArray);
    Tp* base_addr = (Tp*) PyArray_DATA(inArray);
    npy_intp* stride = PyArray_STRIDES(inArray);
    size_t surface_stride    = (size_t) (stride[0]/sizeof(int));
    size_t line_stride    = (size_t) (stride[1]/sizeof(int));
    size_t data_stride = (size_t) (stride[2]/sizeof(int));

    out_ten.clear();
    for(size_t c=0; c<(size_t)size[0]; c++) {
        std::vector<std::vector<T> > temp_mat;

        for(size_t j=0; j<(size_t)size[1]; j++) {
            std::vector<T> temp_vec;

            for(size_t i=0; i<(size_t)size[2]; i++) {
                temp_vec.push_back(static_cast<T>(*(base_addr + (c*surface_stride +    j*line_stride + i*data_stride))));
            }
            temp_mat.push_back(temp_vec);
        }
        out_ten.push_back(temp_mat);
    }

    return 1;
}

template <typename T>
constexpr int INumPy::_PyArray_to_3Dvector(PyArrayObject* inArray, std::vector<std::vector<T> >& out_mat) {
    return (std::is_integral<T>())? INumPy::priv::_PyArray_to_3Dvector_i<T, int>(inArray, out_mat) : INumPy::priv::_PyArray_to_3Dvector_i<T, float>(inArray, out_mat);
}

//* ======================= *//
// Copy C++ 3D Vector to PyArray
template <typename T, typename Tp>
int INumPy::priv::_3Dvector_to_PyArray_i(std::vector<std::vector<std::vector<Tp> > >& in_ten, PyArrayObject** outArray_ptr)
{
    int nd = 3;
    size_t channels = in_ten.size();
    size_t line_number = in_ten[0].size();
    size_t line_stride = in_ten[0][0].size();
    npy_intp dims[3];
    dims[0] = channels;
    dims[1] = line_number;
    dims[2] = line_stride;
    T *data;
    PyArrayObject *_PyTemp;

    // Copy C++ 3D Vector to the Heap
    data = (T*) malloc(channels * line_stride * line_number * sizeof(T));

    for(size_t c=0; c<in_ten.size(); c++)
        for(size_t i=0; i<in_ten[0].size(); i++)
            for(size_t j=0; j<in_ten[0][0].size(); j++)
                data[c*(line_number*line_stride) + i*line_stride + j] = static_cast<T>(in_ten[c][i][j]);

    // Create a PyArray wrapping data
    NPY_TYPES myNpyType;

    if constexpr(std::is_same<T, char>::value)
        myNpyType = NPY_INT8;
    else if constexpr(std::is_same<T, int>::value)
        myNpyType = NPY_INT32;
    else if constexpr(std::is_same<T, long int>::value)
        myNpyType = NPY_INT64;
    else if constexpr(std::is_same<T, float>::value)
        myNpyType = NPY_FLOAT32;
    else if constexpr(std::is_same<T, double>::value)
        myNpyType = NPY_FLOAT64;
    else {
        std::cerr << "ERROR: Not recognized data type" << std::endl;
        return 0;
    }

    _PyTemp = (PyArrayObject*) PyArray_SimpleNewFromData(nd, dims, myNpyType, data);
    if(!_PyTemp)
        return 0;

    // Pass PyArray pointer to the output
    *outArray_ptr = _PyTemp;

    return 1;
}


template <typename T>
int INumPy::_3Dvector_to_PyArray(std::vector<std::vector<std::vector<T> > >& in_ten, PyArrayObject** outArray_ptr) {
    if constexpr (std::is_floating_point<T>::value) {
        return INumPy::priv::_3Dvector_to_PyArray_i<float, T>(in_ten, outArray_ptr);
    }
    else {
        return INumPy::priv::_3Dvector_to_PyArray_i<T, T>(in_ten, outArray_ptr);
    }
}


//* ======================= *//
// Copy PyArray to C++ 4D Vector
template <typename T, typename Tp>
int INumPy::priv::_PyArray_to_4Dvector_i(PyArrayObject* inArray, std::vector<std::vector<std::vector<std::vector<T> > > > & out_ten)
{
    if(PyArray_NDIM(inArray) != 4) return 0;

    npy_intp* size = PyArray_SHAPE(inArray);
    Tp* base_addr = (Tp*) PyArray_DATA(inArray);
    npy_intp* stride = PyArray_STRIDES(inArray);
    size_t channel_stride    = (size_t) (stride[0]/sizeof(int));
    size_t surface_stride    = (size_t) (stride[1]/sizeof(int));
    size_t line_stride    = (size_t) (stride[2]/sizeof(int));
    size_t data_stride = (size_t) (stride[3]/sizeof(int));

    out_ten.clear();
    for(size_t k=0; k<(size_t)size[0]; k++) {
        std::vector<std::vector<std::vector<T> > > temp_chan;

        for(size_t c=0; c<(size_t)size[1]; c++) {
            std::vector<std::vector<T> > temp_mat;
     
            for(size_t j=0; j<(size_t)size[2]; j++) {
                std::vector<T> temp_vec;
     
                for(size_t i=0; i<(size_t)size[3]; i++) {
                    temp_vec.push_back(static_cast<T>(*(base_addr + (k*channel_stride + c*surface_stride + j*line_stride + i*data_stride))));
                }
                temp_mat.push_back(temp_vec);
            }
            temp_chan.push_back(temp_mat);
        }
        out_ten.push_back(temp_chan);
    }

    return 1;
}

template <typename T>
constexpr int INumPy::_PyArray_to_4Dvector(PyArrayObject* inArray, std::vector<std::vector<std::vector<std::vector<T> > > >& out_ten) {
    return (std::is_integral<T>())? INumPy::priv::_PyArray_to_4Dvector_i<T, int>(inArray, out_ten) : INumPy::priv::_PyArray_to_4Dvector_i<T, float>(inArray, out_ten);
}

//* ======================= *//
// Copy C++ 4D Vector to PyArray
template <typename T, typename Tp>
int INumPy::priv::_4Dvector_to_PyArray_i(std::vector<std::vector<std::vector<std::vector<Tp> > > >& in_ten, PyArrayObject** outArray_ptr)
{
    int nd = 4;
    int channels_number = in_ten.size();
    int channels = in_ten[0].size();
    int line_number = in_ten[0][0].size();
    int line_stride = in_ten[0][0][0].size();
    npy_intp dims[4];
    dims[0] = channels_number;
    dims[1] = channels;
    dims[2] = line_number;
    dims[3] = line_stride;
    T *data;
    PyArrayObject *_PyTemp;

    // Copy C++ 4D Vector to the Heap
    data = (T*) malloc(channels_number * channels * line_stride * line_number * sizeof(T));

    for(size_t k=0; k<in_ten.size(); k++)
        for(size_t c=0; c<in_ten[0].size(); c++)
            for(size_t i=0; i<in_ten[0][0].size(); i++)
                for(size_t j=0; j<in_ten[0][0][0].size(); j++)
                    data[k*(channels*line_number*line_stride) + c*(line_number*line_stride) + i*line_stride + j] = static_cast<T>(in_ten[k][c][i][j]);


    // Create a PyArray wrapping data
    NPY_TYPES myNpyType;

    if constexpr(std::is_same<T, char>::value)
        myNpyType = NPY_INT8;
    else if constexpr(std::is_same<T, int>::value)
        myNpyType = NPY_INT32;
    else if constexpr(std::is_same<T, long int>::value)
        myNpyType = NPY_INT64;
    else if constexpr(std::is_same<T, float>::value)
        myNpyType = NPY_FLOAT32;
    else if constexpr(std::is_same<T, double>::value)
        myNpyType = NPY_FLOAT64;
    else {
        std::cerr << "ERROR: Not recognized data type" << std::endl;
        return 0;
    }

    _PyTemp = (PyArrayObject*) PyArray_SimpleNewFromData(nd, dims, myNpyType, data);
    if(!_PyTemp)
        return 0;

    // Pass PyArray pointer to the output
    *outArray_ptr = _PyTemp;

    return 1;
}


template <typename T>
int INumPy::_4Dvector_to_PyArray(std::vector<std::vector<std::vector<std::vector<T> > > >& in_ten, PyArrayObject** outArray_ptr) {
    if constexpr (std::is_floating_point<T>::value) {
        return INumPy::priv::_4Dvector_to_PyArray_i<float, T>(in_ten, outArray_ptr);
    }
    else {
        return INumPy::priv::_4Dvector_to_PyArray_i<T, T>(in_ten, outArray_ptr);
    }
}


//* ======================= *//
// Copy PyArray to C++ 5D Vector
template <typename T, typename Tp>
int INumPy::priv::_PyArray_to_5Dvector_i(PyArrayObject* inArray, std::vector<std::vector<std::vector<std::vector<std::vector<T> > > > > & out_ten)
{
    if(PyArray_NDIM(inArray) != 5) return 0;

    npy_intp* size = PyArray_SHAPE(inArray);
    Tp* base_addr = (Tp*) PyArray_DATA(inArray);
    npy_intp* stride = PyArray_STRIDES(inArray);
    size_t cube_stride    = (size_t) (stride[0]/sizeof(int));
    size_t channel_stride    = (size_t) (stride[1]/sizeof(int));
    size_t surface_stride    = (size_t) (stride[2]/sizeof(int));
    size_t line_stride    = (size_t) (stride[3]/sizeof(int));
    size_t data_stride = (size_t) (stride[4]/sizeof(int));

    out_ten.clear();
    for(size_t z=0; z<(size_t)size[0]; z++) {
        std::vector<std::vector<std::vector<std::vector<T> > > > temp_cube;

        for(size_t k=0; k<(size_t)size[1]; k++) {
            std::vector<std::vector<std::vector<T> > > temp_chan;
     
            for(size_t c=0; c<(size_t)size[2]; c++) {
                std::vector<std::vector<T> > temp_mat;
         
                for(size_t j=0; j<(size_t)size[3]; j++) {
                    std::vector<T> temp_vec;
         
                    for(size_t i=0; i<(size_t)size[4]; i++) {
                        temp_vec.push_back(static_cast<T>(*(base_addr + (z*cube_stride + k*channel_stride + c*surface_stride + j*line_stride + i*data_stride))));
                    }
                    temp_mat.push_back(temp_vec);
                }
                temp_chan.push_back(temp_mat);
            }
            temp_cube.push_back(temp_chan);
        }
        out_ten.push_back(temp_cube);
    }

    return 1;
}

template <typename T>
constexpr int INumPy::_PyArray_to_5Dvector(PyArrayObject* inArray, std::vector<std::vector<T> >& out_mat) {
    return (std::is_integral<T>())? INumPy::priv::_PyArray_to_5Dvector_i<T, int>(inArray, out_mat) : INumPy::priv::_PyArray_to_5Dvector_i<T, float>(inArray, out_mat);
}

//* ======================= *//
// Copy C++ 5D Vector to PyArray
template <typename T, typename Tp>
int INumPy::priv::_5Dvector_to_PyArray_i(std::vector<std::vector<std::vector<std::vector<std::vector<Tp> > > > >& in_ten, PyArrayObject** outArray_ptr)
{
    int nd = 5;
    int batch_size = in_ten.size();
    int channels_number = in_ten[0].size();
    int channels = in_ten[0][0].size();
    int line_number = in_ten[0][0][0].size();
    int line_stride = in_ten[0][0][0][0].size();
    npy_intp dims[5];
    dims[0] = batch_size;
    dims[1] = channels_number;
    dims[2] = channels;
    dims[3] = line_number;
    dims[4] = line_stride;
    T *data;
    PyArrayObject *_PyTemp;

    // Copy C++ 5D Vector to the Heap
    data = (T*) malloc(batch_size * channels_number * channels * line_stride * line_number * sizeof(T));

    for(size_t b=0; b<in_ten.size(); b++)
        for(size_t k=0; k<in_ten[0].size(); k++)
            for(size_t c=0; c<in_ten[0][0].size(); c++)
                for(size_t i=0; i<in_ten[0][0][0].size(); i++)
                    for(size_t j=0; j<in_ten[0][0][0][0].size(); j++)
                        data[b*(channels_number*channels*line_number*line_stride) + k*(channels*line_number*line_stride) + c*(line_number*line_stride) + i*line_stride + j] = static_cast<T>(in_ten[b][k][c][i][j]);


    // Create a PyArray wrapping data
    NPY_TYPES myNpyType;

    if constexpr(std::is_same<T, char>::value)
        myNpyType = NPY_INT8;
    else if constexpr(std::is_same<T, int>::value)
        myNpyType = NPY_INT32;
    else if constexpr(std::is_same<T, long int>::value)
        myNpyType = NPY_INT64;
    else if constexpr(std::is_same<T, float>::value)
        myNpyType = NPY_FLOAT32;
    else if constexpr(std::is_same<T, double>::value)
        myNpyType = NPY_FLOAT64;
    else {
        std::cerr << "ERROR: Not recognized data type" << std::endl;
        return 0;
    }

    _PyTemp = (PyArrayObject*) PyArray_SimpleNewFromData(nd, dims, myNpyType, data);
    if(!_PyTemp)
        return 0;

    // Pass PyArray pointer to the output
    *outArray_ptr = _PyTemp;

    return 1;
}


template <typename T>
int INumPy::_5Dvector_to_PyArray(std::vector<std::vector<std::vector<std::vector<std::vector<T> > > > >& in_ten, PyArrayObject** outArray_ptr) {
    if constexpr (std::is_floating_point<T>::value) {
        return INumPy::priv::_5Dvector_to_PyArray_i<float, T>(in_ten, outArray_ptr);
    }
    else {
        return INumPy::priv::_5Dvector_to_PyArray_i<T, T>(in_ten, outArray_ptr);
    }
}



#endif