#include "orb.h"
#include "orbd.h"
#include <cmath>


#ifndef MIN
#  define MIN(a, b)  ((a) > (b) ? (b) : (a))
#endif


namespace orb
{

	Orbor::Orbor()
	{
	}


	Orbor::~Orbor()
	{
		if (omem)
		{
			CHECK(cudaFree(omem));
		}
		if (vmem)
		{
			CHECK(cudaFree(vmem));
		}
		if (bmem)
		{
			CHECK(cudaFree(bmem));
		}
		if (d_detect_buf)
		{
			CHECK(cudaFree(d_detect_buf));
		}
	}


	void Orbor::init(int _noctaves, int _edge_threshold, int _wta_k, ScoreType _score_type,
		int _patch_size, int _fast_threshold, int _retain_topn, int _max_pts)
	{
		noctaves = _noctaves;
		edge_threshold = _edge_threshold;
		wta_k = _wta_k;
		score_type = _score_type;
		patch_size = _patch_size;
		fast_threshold = _fast_threshold;
		retain_topn = _retain_topn;
		max_pts = _max_pts;

		getPointCounter((void**)&d_point_counter_addr);
		setFastThresholdLUT(fast_threshold);
		setUmax(patch_size);
		setPattern(patch_size, wta_k);
		setGaussianKernel();
		if (score_type == HARRIS_SCORE)
		{
			setScaleSqSq();
		}
	}


	void Orbor::detectAndCompute(unsigned char* image, OrbData& result, int3 whp0, void** desc_addr, const bool compute_desc)
	{
		// Update parameters
		const bool reused = (whp0.x == width) && (whp0.y == height);
		if ((reused && !omem) || !reused)
		{
			this->updateParam(whp0);
		}
		else if (reused && omem)
		{
			CHECK(cudaMemset(omem, 0, obytes));
			CHECK(cudaMemset(vmem, 0, vbytes));
		}

		// Detect keypoints (returns sorted, truncated to max_pts)
		this->detect(image, result);

		// Compute descriptors
		if (compute_desc && result.num_pts > 0)
		{
			// Compute orientation
			hComputeAngle(omem, result, oszp.data(), max_octave, patch_size);

			// Blurring (out-of-place for determinism)
			hGassianBlur(omem, bmem, oszp.data(), max_octave);

			// Compute descriptors
			unsigned char* desc = (unsigned char*)(*desc_addr);
			if (desc)
			{
				CHECK(cudaFree(desc));
			}
			CHECK(cudaMalloc(desc_addr, result.num_pts * 32 * sizeof(unsigned char)));
			desc = (unsigned char*)(*desc_addr);
			hDescribe(omem, result, desc, wta_k, max_octave);
		}

		// Copy point data to host
		if (result.h_data != NULL && result.num_pts > 0)
		{
			int* h_ptr = &result.h_data[0].x;
			int* d_ptr = &result.d_data[0].x;
			CHECK(cudaMemcpy2D(h_ptr, sizeof(OrbPoint), d_ptr, sizeof(OrbPoint), 3 * sizeof(int) + 2 * sizeof(float), result.num_pts, cudaMemcpyDeviceToHost));
		}
	}


	void Orbor::match(OrbData& result1, OrbData& result2, unsigned char* desc1, unsigned char* desc2, float nndr_ratio)
	{
		hMatch(result1, result2, desc1, desc2, nndr_ratio);
		if (result1.h_data)
		{
			int* h_ptr = &result1.h_data[0].match;
			int* d_ptr = &result1.d_data[0].match;
			CHECK(cudaMemcpy2D(h_ptr, sizeof(OrbPoint), d_ptr, sizeof(OrbPoint), 2 * sizeof(int), result1.num_pts, cudaMemcpyDeviceToHost));
		}
	}


	void Orbor::initOrbData(OrbData& data, const int max_pts, const bool host, const bool dev)
	{
		data.num_pts = 0;
		//data.max_pts = max_pts;
		const size_t size = sizeof(OrbPoint) * max_pts;
		data.h_data = host ? (OrbPoint*)calloc(1, size) : NULL;
		data.d_data = NULL;
		if (dev)
		{
			CHECK(cudaMalloc((void**)&data.d_data, size));
		}
	}


	void Orbor::freeOrbData(OrbData& data)
	{
		if (data.d_data != NULL)
		{
			CHECK(cudaFree(data.d_data));
		}
		if (data.h_data != NULL)
		{
			free(data.h_data);
		}
		data.num_pts = 0;
		//data.max_pts = 0;
	}


	void Orbor::updateParam(int3 whp0)
	{
		// Compute truly octave layers
		max_octave = MIN(noctaves, (int)log2f(MIN(whp0.x, whp0.y) / 80) + 1);

		// Compute size
		oszp.resize(5 * max_octave + 1);
		int* osizes = oszp.data();
		int* widths = osizes + max_octave;
		int* heights = widths + max_octave;
		int* pitchs = heights + max_octave;
		int* offsets = pitchs + max_octave;

		width = whp0.x;
		height = whp0.y;
		widths[0] = width;
		heights[0] = height;
		pitchs[0] = iAlignUp(width, 128);
		osizes[0] = height * pitchs[0];
		offsets[0] = 0;
		offsets[1] = offsets[0] + osizes[0];

		for (int i = 0, j = 1, k = 2; j < max_octave; i++, j++, k++)
		{
			widths[j] = widths[i] >> 1;
			heights[j] = heights[i] >> 1;
			pitchs[j] = iAlignUp(widths[j], 128);
			osizes[j] = heights[j] * pitchs[j];
			offsets[k] = offsets[j] + osizes[j];
		}
		obytes = offsets[max_octave] * sizeof(unsigned char);
		// Margin for safe out-of-bounds reads in angleIC/gDescribe at last octave border
		int margin_pitch = max_octave > 1 ? pitchs[max_octave - 1] : pitchs[0];
		obytes += (patch_size / 2 + 1) * margin_pitch;
		vbytes = osizes[0] * (sizeof(float) + sizeof(int));

		// Clear old memory and Allocate new memory
		if (omem)
		{
			CHECK(cudaFree(omem));
		}
		if (vmem)
		{
			CHECK(cudaFree(vmem));
		}
		if (bmem)
		{
			CHECK(cudaFree(bmem));
		}
		CHECK(cudaMalloc((void**)&omem, obytes));
		CHECK(cudaMalloc((void**)&vmem, vbytes));
		bbytes = osizes[0] * sizeof(unsigned char);
		CHECK(cudaMalloc((void**)&bmem, bbytes));

		// Size detection buffer to image area — guarantees capturing all keypoints
		int new_detect_size = widths[0] * heights[0] / 8;
		if (new_detect_size < max_pts * 2)
			new_detect_size = max_pts * 2;
		if (new_detect_size != detect_buf_size)
		{
			detect_buf_size = new_detect_size;
			if (d_detect_buf)
			{
				CHECK(cudaFree(d_detect_buf));
			}
			CHECK(cudaMalloc((void**)&d_detect_buf, sizeof(OrbPoint) * detect_buf_size));
			setMaxNumPoints(detect_buf_size);
		}

		makeOffsets(pitchs, max_octave);
	}


	void Orbor::detect(unsigned char* image, OrbData& result)
	{
		CHECK(cudaMemset(d_point_counter_addr, 0, sizeof(unsigned int)));

		// Detect into over-allocated buffer to capture all keypoints
		OrbData detect_tmp{};
		detect_tmp.d_data = d_detect_buf;
		const bool use_harris = score_type == HARRIS_SCORE;
		hFastDectectWithNMS(image, omem, vmem, detect_tmp, oszp.data(), max_octave, fast_threshold, edge_threshold, use_harris);

		unsigned int actual_count;
		CHECK(cudaMemcpy(&actual_count, d_point_counter_addr, sizeof(unsigned int), cudaMemcpyDeviceToHost));
		actual_count = MIN(actual_count, (unsigned int)detect_buf_size);

		// Sort all detected keypoints, then keep only top max_pts
		hSortKeypoints(d_detect_buf, (int)actual_count);
		result.num_pts = MIN((int)actual_count, max_pts);
		if (result.num_pts > 0)
		{
			CHECK(cudaMemcpy(result.d_data, d_detect_buf,
				sizeof(OrbPoint) * result.num_pts, cudaMemcpyDeviceToDevice));
		}

		// Sync GPU counter so downstream kernels (angleIC, gDescribe) respect truncated count
		unsigned int synced = (unsigned int)result.num_pts;
		CHECK(cudaMemcpy(d_point_counter_addr, &synced, sizeof(unsigned int), cudaMemcpyHostToDevice));
	}

}