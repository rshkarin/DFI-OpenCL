/*
 * main.cpp
 *
 *  Created on: Jun 6, 2012
 *      Author: mathii
 */

/* #define DEVICE_TYPE CL_DEVICE_TYPE_CPU
 *
TOTAL TIME (Sinogram shifting + Zeropadding) time:0.00135

TOTAL TIME (Applying 1-D FFT to the each strip of the sinogram and shifting it) time:0.30660

TOTAL TIME (Make fftshift) time:0.00222

TOTAL TIME (Data Interpolation) time:0.03274

TOTAL TIME (Applying 2-D FFT to the interpolated spectrum) time:0.77266

TOTAL TIME (Applying 2-D fftshidt to the restored image) time:0.00903
*/


/* #define DEVICE_TYPE CL_DEVICE_TYPE_GPU
 *
TOTAL TIME (Sinogram shifting + Zeropadding) time:0.00175

TOTAL TIME (Applying 1-D FFT to the each strip of the sinogram and shifting it) time:0.26844

TOTAL TIME (Make fftshift) time:0.00101

TOTAL TIME (Data Interpolation) time:0.00459

TOTAL TIME (Applying 2-D FFT to the interpolated spectrum) time:0.63729

TOTAL TIME (Applying 2-D fftshidt to the restored image) time:0.00369
 */

#include <cstdio>
#include <stdlib.h>
#include <sys/stat.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <clFFT.h>
#include <CL/cl.h>
#include <tiffio.h>
#include <glib.h>

const char* opencl_map_error(int error);
char *load_program_source(const char *filename);

const char* opencl_error_msgs[] = {
  "CL_SUCCESS",
  "CL_DEVICE_NOT_FOUND",
  "CL_DEVICE_NOT_AVAILABLE",
  "CL_COMPILER_NOT_AVAILABLE",
  "CL_MEM_OBJECT_ALLOCATION_FAILURE",
  "CL_OUT_OF_RESOURCES",
  "CL_OUT_OF_HOST_MEMORY",
  "CL_PROFILING_INFO_NOT_AVAILABLE",
  "CL_MEM_COPY_OVERLAP",
  "CL_IMAGE_FORMAT_MISMATCH",
  "CL_IMAGE_FORMAT_NOT_SUPPORTED",
  "CL_BUILD_PROGRAM_FAILURE",
  "CL_MAP_FAILURE",
  "CL_MISALIGNED_SUB_BUFFER_OFFSET",
  "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST",

  /* next IDs start at 30! */
  "CL_INVALID_VALUE",
  "CL_INVALID_DEVICE_TYPE",
  "CL_INVALID_PLATFORM",
  "CL_INVALID_DEVICE",
  "CL_INVALID_CONTEXT",
  "CL_INVALID_QUEUE_PROPERTIES",
  "CL_INVALID_COMMAND_QUEUE",
  "CL_INVALID_HOST_PTR",
  "CL_INVALID_MEM_OBJECT",
  "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
  "CL_INVALID_IMAGE_SIZE",
  "CL_INVALID_SAMPLER",
  "CL_INVALID_BINARY",
  "CL_INVALID_BUILD_OPTIONS",
  "CL_INVALID_PROGRAM",
  "CL_INVALID_PROGRAM_EXECUTABLE",
  "CL_INVALID_KERNEL_NAME",
  "CL_INVALID_KERNEL_DEFINITION",
  "CL_INVALID_KERNEL",
  "CL_INVALID_ARG_INDEX",
  "CL_INVALID_ARG_VALUE",
  "CL_INVALID_ARG_SIZE",
  "CL_INVALID_KERNEL_ARGS",
  "CL_INVALID_WORK_DIMENSION",
  "CL_INVALID_WORK_GROUP_SIZE",
  "CL_INVALID_WORK_ITEM_SIZE",
  "CL_INVALID_GLOBAL_OFFSET",
  "CL_INVALID_EVENT_WAIT_LIST",
  "CL_INVALID_EVENT",
  "CL_INVALID_OPERATION",
  "CL_INVALID_GL_OBJECT",
  "CL_INVALID_BUFFER_SIZE",
  "CL_INVALID_MIP_LEVEL",
  "CL_INVALID_GLOBAL_WORK_SIZE"
};

#define CL_CHECK_ERROR(FUNC) \
{ \
cl_int err = FUNC; \
if (err != CL_SUCCESS) { \
fprintf(stderr, "Error %d executing %s on %d: (%s)\n",\
err, __FILE__, __LINE__, opencl_map_error(err)); \
abort(); \
}; \
}

#define DEVICE_TYPE CL_DEVICE_TYPE_GPU
#define PROGRAM_SRC "src/DFI.cl"

const cl_uint wavefronts_per_SIMD = 14;
const size_t local_work_size = 64;

cl_kernel get_kernel(char *kernel_name, cl_context *context, cl_device_id *device);

void dfi_process_sinogram(const char* tiff_input, const char* tiff_output, int center_rotation);
clFFT_Complex *tiff_read_complex(const char *filename,
								 int center_pos,
								 int *bits_per_sample,
								 int *samples_per_pixel,
								 int *width,
								 int *height);
int tiff_decode_complex(TIFF *tif, clFFT_Complex *buffer, int offset);
void tiff_write_complex(const char *filename,
						clFFT_Complex *data,
						int width,
						int height);
int main() {
  dfi_process_sinogram("resources/sino-egg.tif","resources/sino-out.tif", 461);
  return 0;
}

void dfi_process_sinogram(const char* tiff_input, const char* tiff_output, int center_rotation)
{
	if(!tiff_input) {
		printf("The filename of input is not valid. (pointer tiff_input = %p)", tiff_input);
	    return;
	}

	if(!tiff_output) {
	  printf("The filename of output is not valid. (pointer tiff_output = %p)", tiff_output);
	  return;
	}

	/////////////////////
	/* Input Data Part */
	/////////////////////

	/* Input a slice properties */
	int bits_per_sample;
	int samples_per_pixel;
	int theta_size;
	int slice_size;

	/* Read the slice */
	clFFT_Complex *data_tiff = tiff_read_complex(tiff_input,
	  	  	  	  	  	 	 	 	 	 	 	 center_rotation,
												 &bits_per_sample,
												 &samples_per_pixel,
												 &slice_size,
												 &theta_size);

	//tiff_write_complex("resources/initial-sino.tif", data_tiff, slice_size, theta_size);

	/*
	 * OpenCL
	 */
	printf("Hey!1\n");
	cl_int status = CL_SUCCESS;
	cl_platform_id platform;

	printf("Hey!1.2\n");
	CL_CHECK_ERROR(clGetPlatformIDs(1, &platform, NULL));

	printf("Hey!2\n");
	cl_device_id device[100];    // Compute device
	cl_context context;     // Compute context

	printf("Hey!3\n");
	CL_CHECK_ERROR(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 100, device, NULL));

	device[0] = device[2];
	device[1] = device[3];
	device[2] = device[4];

	//printf("Hey!3.1\n");
	context = clCreateContext(NULL, 1, device, NULL, NULL, &status);
	CL_CHECK_ERROR(status);

	/*
	 * Device
	 */
	printf("Hey!3.2\n");
	cl_int device_max_cu = 0;
	CL_CHECK_ERROR(clGetDeviceInfo(device[0], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &device_max_cu, NULL));
	size_t wg_count = device_max_cu * wavefronts_per_SIMD;
    size_t global_work_size = wg_count * local_work_size;

    printf("Hey!3.3\n");

    /*
     * Queues, Kernels
     */
    cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;
    cl_command_queue command_queue = clCreateCommandQueue(context, device[0], properties, &status);
    CL_CHECK_ERROR(status);

    printf("Hey! 3.4");
    cl_kernel kernel_linear_interp = get_kernel("linear_interp", &context, &device[0]);
    cl_kernel kernel_zero_ifftshift = get_kernel("zero_ifftshift", &context, &device[0]);
    cl_kernel kernel_fftshift = get_kernel("fftshift", &context, &device[0]);
    cl_kernel kernel_2dshift = get_kernel("shift2d", &context, &device[0]);
    cl_kernel kernel_crop_data = get_kernel("crop_data", &context, &device[0]);

	////////////////////////
	/* OpenCL - DFI Part */
	////////////////////////

	/* Reconstruction properties */
	int oversampling_ratio = 2;
	int dx = 1; /* zoom times */
	//int size_s = slice_size * oversampling_ratio;

	int min_theta = 0;
	int max_theta = theta_size - 1;

	int power = ceil(log2(slice_size));
	if (pow(2, power) == slice_size) {
		power++;
	}

	int size_zeropad_s = pow(2, power); /* get length of FFT operations */
	int size_s = size_zeropad_s;

	float d_omega_s = 2 * M_PI / (size_zeropad_s * dx); //normalized ratio [0; 2PI]

	GTimer *total_t = g_timer_new();
	g_timer_start(total_t);

	GTimer *global_timer = g_timer_new();
	g_timer_start(global_timer);

	/////////////////////////////////////
	/* Sinogram shifting + Zeropadding */
	/////////////////////////////////////
	long data_size = slice_size * theta_size * sizeof(clFFT_Complex);
	long zeropad_data_size = theta_size * size_zeropad_s * sizeof(clFFT_Complex);

	/* Buffers */
	cl_mem original_data_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, data_size, NULL, &status);
    clEnqueueWriteBuffer(command_queue,
    			     	 original_data_buffer,
    			     	 CL_FALSE,
    		             0,
    		             data_size,
    		             data_tiff,
    		             0,
    		             NULL,
    		             NULL);

    cl_mem zeropad_ifftshift_data_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, zeropad_data_size, NULL, &status);
    float *zero_out = (float *)g_malloc0(zeropad_data_size);
    clEnqueueWriteBuffer(command_queue,
    					 zeropad_ifftshift_data_buffer,
    			     	 CL_FALSE,
    		             0,
    		             zeropad_data_size,
    		             zero_out,
    		             0,
    		             NULL,
    		             NULL);

    /* Set arguments */
    clSetKernelArg(kernel_zero_ifftshift, 0, sizeof(void *), (void *)&original_data_buffer);
    clSetKernelArg(kernel_zero_ifftshift, 1, sizeof(theta_size), &theta_size);
    clSetKernelArg(kernel_zero_ifftshift, 2, sizeof(slice_size), &slice_size);

    clSetKernelArg(kernel_zero_ifftshift, 3, sizeof(void *), (void *)&zeropad_ifftshift_data_buffer);
    clSetKernelArg(kernel_zero_ifftshift, 4, sizeof(theta_size), &theta_size);
    clSetKernelArg(kernel_zero_ifftshift, 5, sizeof(size_zeropad_s), &size_zeropad_s);

    /* Run kernel */
    status = clEnqueueNDRangeKernel(command_queue,
    					kernel_zero_ifftshift,
    					1, // work dimensional 1D, 2D, 3D
    					NULL, // offset
    					&global_work_size, // total number of WI
    					&local_work_size, // number of WI in WG
    					0, // number events in wait list
    					NULL,  // event wait list
    					NULL); // event

    // Copy result from device to host
    /*
    clFFT_Complex *fur_kernel_sino = (clFFT_Complex *)clEnqueueMapBuffer(command_queue,
    									     	 	 	 	 	 	 	 zeropad_ifftshift_data_buffer,
    									     	 	 	 	 	 	 	 CL_TRUE,
    									     	 	 	 	 	 	 	 CL_MAP_READ,
    									     	 	 	 	 	 	 	 0,
    									     	 	 	 	 	 	 	 zeropad_data_size,
                                                                 	     0, NULL, NULL, NULL );

    clFinish(command_queue);
    tiff_write_complex("resources/zeropad-sino.tif", fur_kernel_sino, size_zeropad_s, theta_size);
	*/

    g_timer_stop(total_t);
    printf("\nTOTAL TIME (Sinogram shifting + Zeropadding) time:%3.5f\n", g_timer_elapsed(total_t, NULL));

    g_timer_reset(total_t);
    g_timer_start(total_t);
    ////////////////////////////////////////////////////////////////////////
    /* Applying 1-D FFT to the each strip of the sinogram and shifting it */
    ////////////////////////////////////////////////////////////////////////

    cl_mem zeropadded_1dfft_data_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, zeropad_data_size, NULL, &status);

    /* Setup clAmdFft */
    clFFT_Dim3 sino_fft;
    sino_fft.x = size_zeropad_s;
    sino_fft.y = 1;
    sino_fft.z = 1;
    cl_int error_code;

    /* Create FFT plan */
    clFFT_Plan plan_1dfft_sinogram = clFFT_CreatePlan(context,
    												  sino_fft,
    												  clFFT_1D,
    												  clFFT_InterleavedComplexFormat,
    												  &error_code);
    if (error_code != CL_SUCCESS) {
    	printf("\nERROR: clFFT_CreatePlan - plan_1dfft_sinogram\n");
    }

    /* Execute FFT */
    error_code = clFFT_ExecuteInterleaved(command_queue,
    						 	 	      plan_1dfft_sinogram,
    						 	 	      theta_size,
    						 	 	      clFFT_Forward,
    						 	 	      zeropad_ifftshift_data_buffer,
    						 	 	      zeropadded_1dfft_data_buffer,
    						 	 	      0,
    						 	 	      NULL,
    						 	 	      NULL);

    if (error_code != CL_SUCCESS) {
    	printf("\nERROR: clFFT_ExecuteInterleaved\n");
    }

    clFinish(command_queue);

    // Free FFT plan
    clFFT_DestroyPlan(plan_1dfft_sinogram);

    // Copy result from device to host
    /*
    clFFT_Complex *fourier_kernel_sinogram = (clFFT_Complex *)malloc(zeropad_data_size);
    clEnqueueReadBuffer(command_queue, zeropadded_1dfft_data_buffer, CL_TRUE, 0, zeropad_data_size, fourier_kernel_sinogram, 0, NULL, NULL);
    clFinish(command_queue);

	tiff_write_complex("resources/1dfft-sino.tif", fourier_kernel_sinogram, size_zeropad_s, theta_size);
    */
    g_timer_stop(total_t);
    printf("\nTOTAL TIME (Applying 1-D FFT to the each strip of the sinogram and shifting it) time:%3.5f\n", g_timer_elapsed(total_t, NULL));

    g_timer_reset(total_t);
    g_timer_start(total_t);
	///////////////////
	/* Make fftshift */
	///////////////////

    /* Buffers */
    cl_mem zeropad_fftshift_data_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, zeropad_data_size, NULL, &status);

    /* Set arguments */
    clSetKernelArg(kernel_fftshift, 0, sizeof(void *), (void *)&zeropadded_1dfft_data_buffer);
    clSetKernelArg(kernel_fftshift, 1, sizeof(theta_size), &theta_size);
    clSetKernelArg(kernel_fftshift, 2, sizeof(size_zeropad_s), &size_zeropad_s);

    clSetKernelArg(kernel_fftshift, 3, sizeof(void *), (void *)&zeropad_fftshift_data_buffer);
    clSetKernelArg(kernel_fftshift, 4, sizeof(theta_size), &theta_size);
    clSetKernelArg(kernel_fftshift, 5, sizeof(size_zeropad_s), &size_zeropad_s);

    /* Run kernel */
    status = clEnqueueNDRangeKernel(command_queue,
    					kernel_fftshift,
    					1, // work dimensional 1D, 2D, 3D
    					NULL, // offset
    					&global_work_size, // total number of WI
    					&local_work_size, // number of WI in WG
    					0, // number events in wait list
    					NULL,  // event wait list
    					NULL); // event

    /* Copy result from device to host */
    /*
    clFFT_Complex *fur_kernel_fftshift_sino = (clFFT_Complex *)clEnqueueMapBuffer(command_queue,
										      	  zeropad_fftshift_data_buffer,
										      	  CL_TRUE,
      									   	      CL_MAP_READ,0,zeropad_data_size,
      									   	      0, NULL, NULL, NULL );

    clFinish(command_queue);
    tiff_write_complex("resources/fftshift-sino.tif", fur_kernel_fftshift_sino, size_zeropad_s, theta_size);
    */

    g_timer_stop(total_t);
    printf("\nTOTAL TIME (Make fftshift) time:%3.5f\n", g_timer_elapsed(total_t, NULL));

    g_timer_reset(total_t);
    g_timer_start(total_t);
	////////////////////////
	/* Data Interpolation */
    ////////////////////////

	/* Performing Interpolation */
    cl_long data_length = size_s * size_s;
    cl_int in_rows = theta_size;
    cl_int in_cols = size_zeropad_s;

    cl_float norm_ratio = d_omega_s/dx;

    cl_float in_rows_first_val = min_theta;
    cl_float in_rows_last_val = max_theta;

    cl_float in_cols_first_val = (-in_cols/2)*norm_ratio;
    cl_float in_cols_last_val = (in_cols/2-1)*norm_ratio;

    cl_int interp_rows = size_s;
    cl_int interp_cols = interp_rows;

    cl_int iparams[5];
    iparams[0] = in_rows;
    iparams[1] = in_cols;
    iparams[2] = dx;
    iparams[3] = interp_rows;
    iparams[4] = interp_cols;

    cl_float fparams[5];
    fparams[0] = in_rows_first_val;
    fparams[1] = in_rows_last_val;
    fparams[2] = in_cols_first_val;
    fparams[3] = in_cols_last_val;
    fparams[4] = norm_ratio;

    /* Buffers */
    cl_mem i_buffer = clCreateBuffer(context,
                                     CL_MEM_READ_ONLY,
                                     sizeof(cl_int) * 5,
                                     NULL,
                                     NULL);

    clEnqueueWriteBuffer(command_queue,
                         i_buffer,
                         CL_FALSE,
                         0,
                         sizeof(cl_int) * 5,
                         iparams,
                         0,
                         NULL,
                         NULL);

    cl_mem f_buffer = clCreateBuffer(context,
                                     CL_MEM_READ_ONLY,
                                     sizeof(cl_float) * 5,
                                     NULL,
                                     NULL);

    clEnqueueWriteBuffer(command_queue,
                         f_buffer,
                         CL_FALSE,
                         0,
                         sizeof(cl_float) * 5,
                         fparams,
                         0,
                         NULL,
                         NULL);

    cl_mem output_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, data_length * sizeof(clFFT_Complex), NULL, NULL);

    /* Set arguments */
    clSetKernelArg(kernel_linear_interp, 0, sizeof(void *), (void *)&i_buffer);
    clSetKernelArg(kernel_linear_interp, 1, sizeof(void *), (void *)&f_buffer);
    clSetKernelArg(kernel_linear_interp, 2, sizeof(void *), (void *)&zeropad_fftshift_data_buffer);
    clSetKernelArg(kernel_linear_interp, 3, sizeof(void *), (void *)&output_buffer);
    clSetKernelArg(kernel_linear_interp, 4, sizeof(data_length), &data_length);

    /* Run kernel */
    status = clEnqueueNDRangeKernel(command_queue,
    					kernel_linear_interp,
    					1, // work dimensional 1D, 2D, 3D
    					NULL, // offset
    					&global_work_size, // total number of WI
    					&local_work_size, // nomber of WI in WG
    					0, // num events in wait list
    					NULL,  // event wait list
    					NULL); // event

    // Copy result from device to host
    /*
    clFFT_Complex *interpolated_spectrum = (clFFT_Complex *)clEnqueueMapBuffer(command_queue,
                                                                  	  	   output_buffer,
                                                                  	  	   CL_TRUE,
                                                                  	  	   CL_MAP_READ,
                                                                 	  	   0,
                                                                  	  	   data_length * sizeof(clFFT_Complex),
                                                                  	  	   0, NULL, NULL, NULL );

    clFinish(command_queue);
    tiff_write_complex("resources/interpolated-sino.tif", interpolated_spectrum, size_s, size_s);
    */
    g_timer_stop(total_t);
    printf("\nTOTAL TIME (Data Interpolation) time:%3.5f\n", g_timer_elapsed(total_t, NULL));

    g_timer_reset(total_t);
    g_timer_start(total_t);
    ///////////////////////////////////////////////////
    /* Applying 2-D FFT to the interpolated spectrum */
    ///////////////////////////////////////////////////

    /* Setup 2D IFFT */
    clFFT_Dim3 sino_2dfft;
    sino_2dfft.x = size_s;
    sino_2dfft.y = size_s;
    sino_2dfft.z= 1;

    /* Create 2D IFFT plan */
    clFFT_Plan plan_2difft = clFFT_CreatePlan(context,
    										  sino_2dfft,
    										  clFFT_2D,
    										  clFFT_InterleavedComplexFormat,
    										  &error_code);

    if (error_code != CL_SUCCESS) {
    	printf("\nERROR: clFFT_CreatePlan - plan_2difft\n");
    }

    /* Execute 2D IFFT */
    cl_mem reconstructed_image_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, data_length * sizeof(clFFT_Complex), NULL, NULL);
    error_code = clFFT_ExecuteInterleaved(command_queue,
    									  plan_2difft,
    									  1,
    									  clFFT_Inverse,
    									  output_buffer,
    									  reconstructed_image_buffer,
    									  0,
    									  NULL,
    									  NULL);

    if (error_code != CL_SUCCESS) {
    	printf("\nERROR: clFFT_ExecuteInterleaved - 2D IFFT\n");
    }

    clFinish(command_queue);

    // Free FFT plan
    clFFT_DestroyPlan(plan_2difft);

    // Copy result from device to host
    /*
    clFFT_Complex *ifft2d_interpolated_spectrum = (clFFT_Complex *)malloc(data_length * sizeof(clFFT_Complex));
    clEnqueueReadBuffer(command_queue,
    					reconstructed_image_buffer,
    					CL_TRUE,
    					0,
    					data_length * sizeof(clFFT_Complex),
    					ifft2d_interpolated_spectrum,
    					0,
    					NULL,
    					NULL);
    tiff_write_complex("resources/ifft2d_interpolated_spectrum.tif", ifft2d_interpolated_spectrum, size_s, size_s);
    clFinish(command_queue);
    */

    g_timer_stop(total_t);
    printf("\nTOTAL TIME (Applying 2-D FFT to the interpolated spectrum) time:%3.5f\n", g_timer_elapsed(total_t, NULL));

    g_timer_reset(total_t);
    g_timer_start(total_t);
    /////////////////////////////////////////////////
    /* Applying 2-D fftshidt to the restored image */
    /////////////////////////////////////////////////

    /* Buffers */
    cl_mem two_dim_fftshifted_data_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, data_length * sizeof(clFFT_Complex), NULL, &status);

    /* Set arguments */
    cl_int inverse_flag = 0;

    clSetKernelArg(kernel_2dshift, 0, sizeof(void *), (void *)&reconstructed_image_buffer);
    clSetKernelArg(kernel_2dshift, 1, sizeof(void *), (void *)&two_dim_fftshifted_data_buffer);
    clSetKernelArg(kernel_2dshift, 2, sizeof(interp_rows), &interp_rows);
    clSetKernelArg(kernel_2dshift, 3, sizeof(interp_cols), &interp_cols);
    clSetKernelArg(kernel_2dshift, 4, sizeof(inverse_flag), &inverse_flag);

    /* Run kernel */
    status = clEnqueueNDRangeKernel(command_queue,
    					kernel_2dshift,
        				1, // work dimensional 1D, 2D, 3D
        				NULL, // offset
        				&global_work_size, // total number of WI
        				&local_work_size, // number of WI in WG
        				0, // number events in wait list
        				NULL,  // event wait list
        				NULL); // event

    /* Copy result from device to host */
    /*
    clFFT_Complex *two_dim_fftshifted_data = (clFFT_Complex *)clEnqueueMapBuffer(command_queue,
										     	 two_dim_fftshifted_data_buffer,
										     	 CL_TRUE,
										     	 CL_MAP_READ,
										     	 0,
										     	 data_length * sizeof(clFFT_Complex),
										     	 0, NULL, NULL, NULL );


    clFinish(command_queue);
    */

    g_timer_stop(total_t);
    printf("\nTOTAL TIME (Applying 2-D fftshidt to the restored image) time:%3.5f\n", g_timer_elapsed(total_t, NULL));


    g_timer_reset(total_t);
    ////////////////
    /* Crop data  */
    ///////////////
    float lt_offset = 0, rb_offset = 0;

    int dif_sides = interp_cols - slice_size;
    if (dif_sides%2) {
       lt_offset = floor(dif_sides / 2.0);
       rb_offset = ceil(dif_sides / 2.0);
    }
    else {
       lt_offset = rb_offset = dif_sides / 2.0;
    }

    /* Buffers */
    long cropped_data_length = slice_size * slice_size * sizeof(clFFT_Complex);
    cl_mem cropped_restored_image_data_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, cropped_data_length, NULL, &status);

    /* Set arguments */
    clSetKernelArg(kernel_crop_data, 0, sizeof(void *), (void *)&two_dim_fftshifted_data_buffer);
    clSetKernelArg(kernel_crop_data, 1, sizeof(void *), (void *)&cropped_restored_image_data_buffer);
    clSetKernelArg(kernel_crop_data, 2, sizeof(slice_size), &slice_size);
    clSetKernelArg(kernel_crop_data, 3, sizeof(interp_cols), &interp_cols);
    clSetKernelArg(kernel_crop_data, 4, sizeof(lt_offset), &lt_offset);
    clSetKernelArg(kernel_crop_data, 5, sizeof(rb_offset), &rb_offset);

    /* Run kernel */
    status = clEnqueueNDRangeKernel(command_queue,
    								kernel_crop_data,
    								1, // work dimensional 1D, 2D, 3D
    								NULL, // offset
    								&global_work_size, // total number of WI
    								&local_work_size, // number of WI in WG
    								0, // number events in wait list
    								NULL,  // event wait list
    								NULL); // event

    g_timer_stop(total_t);
    printf("\nTOTAL TIME (Cropping data) time:%3.5f\n", g_timer_elapsed(total_t, NULL));

    g_timer_stop(global_timer);
    printf("\nTOTAL TIME (Total reconstruction) time:%3.5f\n", g_timer_elapsed(global_timer, NULL));

    // Copy result from device to host
    clFFT_Complex *cropped_restored_image = (clFFT_Complex *)clEnqueueMapBuffer(command_queue,
    																			cropped_restored_image_data_buffer,
    																			CL_TRUE,
    																			CL_MAP_READ,
    																			0,
    																			cropped_data_length,
    																			0, NULL, NULL, NULL );

    clFinish(command_queue);


    /* Write the restored slice */
    tiff_write_complex(tiff_output, cropped_restored_image, slice_size, slice_size);
}

int tiff_decode_complex(TIFF *tif, clFFT_Complex *buffer, int offset)
{
  const int strip_size = TIFFStripSize(tif);
  const int n_strips = TIFFNumberOfStrips(tif);

  printf("\nstrip_size:%d ; n_strips:%d\n", strip_size, n_strips);

  int result = 0;

  int width, height;

  TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
  TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);

  float *temp_buffer = (float *)malloc(strip_size);

  for (int strip = 0; strip < n_strips; strip++) {
	  result = TIFFReadEncodedStrip(tif, strip, temp_buffer, strip_size);

	  if (result == -1)
		  return 0;

	  int pixels_per_strip = strip_size / sizeof(float);
	  int rows = pixels_per_strip / width;

	  for(int i = 0; i < rows; ++i) {
		  for(int j = offset; j < width; ++j) {
	          buffer[strip * (pixels_per_strip - rows * offset) + i * (width-offset) + j - offset].real = temp_buffer[i * width + j];
	          buffer[strip * (pixels_per_strip - rows * offset) + i * (width-offset) + j - offset].imag = 0.0;
	      }
	  }
  }

  free(temp_buffer);

  return 1;
}

clFFT_Complex *tiff_read_complex(const char *filename,
		                         int center_pos,
                                 int *bits_per_sample,
                                 int *samples_per_pixel,
                                 int *width,
                                 int *height)
{
  TIFF *image;

  // Create the TIFF file
  if((image = TIFFOpen(filename, "r")) == NULL){
    printf("Could not open %s for reading\n", filename);
    return NULL;
  }

  // Get TIFF Attributes
  TIFFGetField(image, TIFFTAG_BITSPERSAMPLE, bits_per_sample);
  TIFFGetField(image, TIFFTAG_SAMPLESPERPIXEL, samples_per_pixel);
  TIFFGetField(image, TIFFTAG_IMAGEWIDTH, width);
  TIFFGetField(image, TIFFTAG_IMAGELENGTH, height);

  if (*samples_per_pixel > 1) {
    printf("%s has %i samples per pixel (%i bps)",
           filename, *samples_per_pixel, *bits_per_sample);
  }

  int offset = 0;
  if (center_pos != -1) {
	  offset = *width - (*width - center_pos) * 2;
	  *width = (*width - center_pos) * 2;
  }

  clFFT_Complex *buffer =
  (clFFT_Complex *)malloc((*height) * (*width) * sizeof(clFFT_Complex));

  if (!tiff_decode_complex(image, buffer, offset)) {
    goto error_close;
  }

  TIFFClose(image);
  return buffer;

error_close:
  TIFFClose(image);
  return NULL;
}

void tiff_write_complex(const char *filename,
						clFFT_Complex *data,
						int width,
						int height)
{
  TIFF *image;

  // Open the TIFF file
  if((image = TIFFOpen(filename, "w")) == NULL){
    printf("Could not open output.tif for writing\n");
    return;
  }

  // Set TIFF Attributes
  TIFFSetField(image, TIFFTAG_IMAGEWIDTH, width);
  TIFFSetField(image, TIFFTAG_IMAGELENGTH, height);
  TIFFSetField(image, TIFFTAG_BITSPERSAMPLE, 32);
  TIFFSetField(image, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
  TIFFSetField(image, TIFFTAG_SAMPLESPERPIXEL, 1);
  TIFFSetField(image, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
  TIFFSetField(image, TIFFTAG_ROWSPERSTRIP, 1);

  // Scanline size
  unsigned int scanlineSize = TIFFScanlineSize(image);

  for (int y = 0; y < height; y++) {
    float *vec_data = (float *)malloc(scanlineSize);

    for (int x = 0; x < width; x++) {
      vec_data[x] = fabs(data[x + y * width].real);
    }
    TIFFWriteScanline(image, vec_data, y, 0);

    free(vec_data);
  }

  TIFFClose(image);
}

const char* opencl_map_error(int error)
{
  if (error >= -14)
    return opencl_error_msgs[-error];
  if (error <= -30)
    return opencl_error_msgs[-error-15];
  return NULL;
}

char *load_program_source(const char *filename)
{
  struct stat statbuf;
  FILE        *fh;
  char        *source;

  fh = fopen(filename, "r");
  if (fh == 0)
    return 0;

  stat(filename, &statbuf);
  source = (char *) malloc(statbuf.st_size + 1);
  fread(source, statbuf.st_size, 1, fh);
  source[statbuf.st_size] = '\0';

  return source;
}

cl_kernel get_kernel(char *kernel_name, cl_context *context, cl_device_id *device)
{
  cl_int status = CL_SUCCESS;

  const char* program_source = load_program_source(PROGRAM_SRC);
  if(program_source == NULL) {
    fprintf(stderr, "Programm can not be created. File was not found.");
    abort();
  }

  cl_program program = clCreateProgramWithSource(*context, 1,
                                                 &program_source, NULL,
                                                 &status);
  CL_CHECK_ERROR(status);

  status = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  /* Print build log */
  char buf[0x10000];
  clGetProgramBuildInfo(program,
                        *device,
                        CL_PROGRAM_BUILD_LOG,
                        0x10000,
                        buf,
                        NULL);

  if(status != CL_SUCCESS) {
    printf("\n-------BUILD LOG:\n %s \n-------\n", buf);
    fprintf(stderr, "Programm can not be build. (%s)", opencl_map_error(status));
    abort();
  }

  return clCreateKernel(program, kernel_name, &status);
}
