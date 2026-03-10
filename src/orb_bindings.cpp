#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "orb.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

namespace py = pybind11;

struct OrbPipeline {
    orb::Orbor orbor;
    orb::OrbData data1{}, data2{};
    unsigned char* d_img = nullptr;
    unsigned char* d_desc1 = nullptr;
    unsigned char* d_desc2 = nullptr;
    int alloc_pitch = 0;
    int alloc_height = 0;
    int max_pts;

    OrbPipeline(int _max_pts, int noctaves, int edge_threshold,
                int fast_threshold, int patch_size, int wta_k)
        : max_pts(_max_pts)
    {
        orbor.initOrbData(data1, max_pts, true, true);
        orbor.initOrbData(data2, max_pts, true, true);
        orbor.init(noctaves, edge_threshold, wta_k, orb::HARRIS_SCORE,
                   patch_size, fast_threshold, -1, max_pts);
    }

    ~OrbPipeline() {
        if (d_img) cudaFree(d_img);
        if (d_desc1) cudaFree(d_desc1);
        if (d_desc2) cudaFree(d_desc2);
        orbor.freeOrbData(data1);
        orbor.freeOrbData(data2);
    }

    OrbPipeline(const OrbPipeline&) = delete;
    OrbPipeline& operator=(const OrbPipeline&) = delete;

    void ensureImageBuffer(int pitch, int h) {
        if (pitch > alloc_pitch || h > alloc_height) {
            if (d_img) cudaFree(d_img);
            alloc_pitch = std::max(pitch, alloc_pitch);
            alloc_height = std::max(h, alloc_height);
            CHECK(cudaMalloc((void**)&d_img,
                             static_cast<size_t>(alloc_pitch) * alloc_height));
        }
    }

    py::tuple detectAndMatch(
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> img1,
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> img2,
        float nndr)
    {
        auto buf1 = img1.request(), buf2 = img2.request();
        if (buf1.ndim != 2 || buf2.ndim != 2)
            throw std::runtime_error("Images must be 2D (height, width)");

        int h1 = buf1.shape[0], w1 = buf1.shape[1];
        int h2 = buf2.shape[0], w2 = buf2.shape[1];
        int3 whp1{w1, h1, iAlignUp(w1, 128)};
        int3 whp2{w2, h2, iAlignUp(w2, 128)};

        ensureImageBuffer(std::max(whp1.z, whp2.z), std::max(h1, h2));

        CHECK(cudaMemcpy2D(d_img, whp1.z, buf1.ptr, w1, w1, h1,
                           cudaMemcpyHostToDevice));
        data1.num_pts = 0;
        orbor.detectAndCompute(d_img, data1, whp1, (void**)&d_desc1, true);

        CHECK(cudaMemcpy2D(d_img, whp2.z, buf2.ptr, w2, w2, h2,
                           cudaMemcpyHostToDevice));
        data2.num_pts = 0;
        orbor.detectAndCompute(d_img, data2, whp2, (void**)&d_desc2, true);

        if (data1.num_pts > 0 && data2.num_pts > 0 && d_desc1 && d_desc2)
            orbor.match(data1, data2, d_desc1, d_desc2, nndr);

        int n1 = data1.num_pts;
        std::vector<int> src_idx, dst_idx;
        src_idx.reserve(n1);
        dst_idx.reserve(n1);
        for (int i = 0; i < n1; i++) {
            int m = data1.h_data[i].match;
            if (m >= 0) {
                src_idx.push_back(i);
                dst_idx.push_back(m);
            }
        }

        int ng = static_cast<int>(src_idx.size());
        py::array_t<float> pts0({ng, 2});
        py::array_t<float> pts1({ng, 2});
        float* p0 = pts0.mutable_data();
        float* p1 = pts1.mutable_data();
        for (int i = 0; i < ng; i++) {
            p0[2 * i]     = static_cast<float>(data1.h_data[src_idx[i]].x);
            p0[2 * i + 1] = static_cast<float>(data1.h_data[src_idx[i]].y);
            p1[2 * i]     = static_cast<float>(data2.h_data[dst_idx[i]].x);
            p1[2 * i + 1] = static_cast<float>(data2.h_data[dst_idx[i]].y);
        }
        return py::make_tuple(pts0, pts1);
    }
};


PYBIND11_MODULE(_cuda_orb, m) {
    m.doc() = "CUDA-accelerated ORB feature detection and matching";

    m.def("init_device", [](int device_id) {
        return initDevice(device_id);
    }, py::arg("device_id") = 0);

    py::class_<OrbPipeline>(m, "OrbPipeline")
        .def(py::init<int, int, int, int, int, int>(),
             py::arg("max_pts") = 10000,
             py::arg("noctaves") = 5,
             py::arg("edge_threshold") = 31,
             py::arg("fast_threshold") = 20,
             py::arg("patch_size") = 31,
             py::arg("wta_k") = 2)
        .def("detect_and_match", &OrbPipeline::detectAndMatch,
             py::arg("image1"), py::arg("image2"), py::arg("nndr") = 0.75f);
}
