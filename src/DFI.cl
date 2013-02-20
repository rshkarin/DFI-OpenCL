#define PI 3.1415f

typedef struct {
	float real;
	float imag;
} clFFT_Complex;


__kernel void clboost() {}

__kernel void linear_interp(__global int *iparams,
							__global float *fparams,
							__global clFFT_Complex *input,
                            __global clFFT_Complex *output,
                            long data_length)
{
 	uint global_size = get_global_size(0);
	uint global_id = get_global_id(0);
	uint local_id = get_local_id(0);
	
	int in_rows = iparams[0];
    int in_cols = iparams[1];
    int dx = iparams[2];
    int interp_rows = iparams[3];
    int interp_cols = iparams[4];
    
  __local float in_rows_first_val;
  __local float in_rows_last_val;
  __local float in_cols_first_val;
  __local float in_cols_last_val;
  __local float norm_ratio;
  
  if(local_id == 0) {
    in_rows_first_val = fparams[0];
    in_rows_last_val = fparams[1];
    in_cols_first_val = fparams[2];
    in_cols_last_val = fparams[3];
    norm_ratio = fparams[4];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

	long all_count = data_length / global_size;
 	long priv_count = (data_length % global_size) > global_id ? 1 : 0;
 	
 	uint row, col;
	for(uint i = 0; i < (all_count + priv_count); ++i, global_id += global_size) {
		row = (uint)(global_id / interp_cols);
    	col = global_id - row * interp_cols;
    	
		//range values along interp_cols and interp_rows
    	float omega_y = (interp_rows/2.0f * norm_ratio) - ((row*interp_cols + col)*norm_ratio/interp_cols);
    	float omega_x = (((row*interp_cols + col)%interp_cols)*norm_ratio) - (interp_rows/2.0f * norm_ratio);
      
    	//get angle in radians 
    	float atan_val = atan2(omega_y, omega_x);
      
    	//transfrom from radiangs to degree a - n * floor(a / n);
   		float a = (atan_val * (in_rows_last_val/PI));
    	float b = in_rows_last_val * floor(a/in_rows_last_val);
    	float theta_i = a - b;
    
    	//get angle on source image and check the border values
    	float s_val = 1 + ((theta_i - in_rows_first_val)/(in_rows_last_val - in_rows_first_val)) * (in_rows - 1);
    	if (s_val < 1 || s_val > in_rows) {
     		s_val = 1;
    	}
   
    	//get hypotenuse, if hypotenuse > PI then go next iteration
    	float sqrt_val = sqrt(pow(omega_x, 2) + pow(omega_y, 2));
    	if (sqrt_val <= PI) {
	    	//set the sign of hypotenuse
	    	float omega_si = sqrt_val * sign(atan_val);
	      
    		//get radius on source image and check the border values
    		float t_val = 1 + ((omega_si - ((-((float)in_cols)/2)*norm_ratio/dx))/((((((float)in_cols)/2) - 1)*norm_ratio/dx) - ((-((float)in_cols)/2)*norm_ratio/dx))) * (in_cols - 1);
   			if (t_val < 1 || t_val > in_cols) {
      			t_val = 1;
    		}
	      
	    	//calculate index on source image
	    	long index = (floor(t_val) + floor(s_val - 1) * in_cols);
	    	
	    	//checking border values on indices
	    	if (s_val == in_cols) {
	      		s_val = s_val + 1;
	      		index = index - in_rows;
	    	}
	    	float s = (s_val - floor(s_val));
	
	    	if (t_val == in_rows) {
	      		t_val = t_val + 1;
	      		index = index - 1;
	    	}
	    	float t = (t_val - floor(t_val));
	    	
	    	//get indices of interpolation      
	    	int cur_idx = index - 1;
	    	int cur_next_idx = cur_idx + 1;
	    	int cur_max_idx = cur_idx + (in_cols + 1);
	      
	    	//get values at indices
	    	float input_re_val = input[cur_idx].real;
	    	float input_im_val = input[cur_idx].imag;
	      
	    	float input_next_re_val = input[cur_next_idx].real;
	    	float input_next_im_val = input[cur_next_idx].imag;
	      
	    	float input_max_re_val = input[cur_max_idx].real;
	    	float input_max_im_val = input[cur_max_idx].imag;
	      
	    	//interpolate real and imaginary values 
	    	output[global_id].real = (input_re_val * (1 - t) + input_next_re_val * (t)) * (1 - s) + 
	      					  	     (input_re_val * (1 - t) + input_max_re_val * (t)) * (s);
	      
	    	output[global_id].imag = (input_im_val * (1 - t) + input_next_im_val * (t)) * (1 - s) + 
	      						     (input_im_val * (1 - t) + input_max_im_val * (t)) * (s); 
	     }
	     else {
	     	output[global_id].real = 0.0f;
	     	output[global_id].imag = 0.0f;
	     }				  	   						  						 
	}

}

__kernel void zero_ifftshift(__global clFFT_Complex *input, int in_rows, int in_cols, 
							 __global clFFT_Complex *output, int out_rows, int out_cols)
{
	uint elemID = get_global_id(0);
    uint global_size = get_global_size(0);
 
    long data_length = in_rows * in_cols;
  	
  	long all_count = data_length / global_size;
 	long priv_count = (data_length % global_size) > elemID ? 1 : 0;
 	
	for(uint i = 0; i < (all_count + priv_count); ++i, elemID += global_size) {
		int in_half_size = floor(in_cols/2.0f);
		int cur_row = ceil((float)(elemID + 1) / ((in_cols & 1)?(in_half_size + 1):(in_half_size)));
		long index = elemID + in_half_size * cur_row;
		
	    if ((index < in_rows * in_cols) && (index >= 0)) {
	    	unsigned long first_index = (cur_row - 1) * in_cols;
	        unsigned long last_index = (cur_row * in_cols) - 1;
	       
	        unsigned long first_zero_index = (cur_row - 1) * out_cols;
	        unsigned long last_zero_index = (cur_row * out_cols) - 1;
	 
	        int inner_index = index - first_index;
	               
	        output[first_zero_index + inner_index - in_half_size].real = input[index].real;
	        output[first_zero_index + inner_index - in_half_size].imag = input[index].imag; 
	        
	        if (((index != last_index) && (in_cols & 1)) || !(in_cols & 1)) {
	        	output[last_zero_index - (in_half_size - 1) + (inner_index - in_half_size)].real = input[first_index + inner_index - in_half_size].real;
	        	output[last_zero_index - (in_half_size - 1) + (inner_index - in_half_size)].imag = input[first_index + inner_index - in_half_size].imag;
	        }
	    }
	}
}

__kernel void fftshift(__global clFFT_Complex *input, int in_rows, int in_cols, 
				       __global clFFT_Complex *output, int out_rows, int out_cols)
{
    uint elemID = get_global_id(0);
    uint global_size = get_global_size(0);
    
    long data_length = in_cols * in_rows;
  	
  	long all_count = data_length / global_size;
 	long priv_count = (data_length % global_size) > elemID ? 1 : 0;
 	
	for(uint i = 0; i < (all_count + priv_count); ++i, elemID += global_size) {
       
	    int in_half_size = floor(in_cols/2.0f);
	    int cur_row = floor((float)(elemID) / ((in_cols & 1)?(in_half_size + 1):(in_half_size)));
	 
	    unsigned long index = elemID + in_half_size * cur_row;
	 
	    if ((index < in_rows * in_cols) && (index >= 0)) {
	    	unsigned long first_index = cur_row * in_cols;
	        unsigned long last_index = ((cur_row + 1) * in_cols) - 1;
	       
	        unsigned long first_zero_index = cur_row * out_cols;
	        unsigned long last_zero_index = ((cur_row + 1) * out_cols) - 1;
	 
	        int inner_index = index - first_index;
	       
	        output[last_zero_index - ((in_cols & 1)?(in_half_size):(in_half_size - 1)) + inner_index].real = input[index].real;
	        output[last_zero_index - ((in_cols & 1)?(in_half_size):(in_half_size - 1)) + inner_index].imag = input[index].imag;
	        
	        if (((inner_index != in_half_size) && (in_cols & 1)) || !(in_cols & 1)) {
	        	output[first_zero_index + inner_index].real = input[first_index + inner_index + ((in_cols & 1)?(in_half_size + 1):(in_half_size))].real;
	        	output[first_zero_index + inner_index].imag = input[first_index + inner_index + ((in_cols & 1)?(in_half_size + 1):(in_half_size))].imag;
	        }      
	    }
     }
  
}

__kernel void shift2d(__global clFFT_Complex *input, __global clFFT_Complex *output, int rows, int cols, int isInverse)
{
	uint index = get_global_id(0);
    uint global_size = get_global_size(0);
 
    long data_length = rows * cols;
  	
  	long all_count = data_length / global_size;
 	long priv_count = (data_length % global_size) > index ? 1 : 0;
 	
	for(uint i = 0; i < (all_count + priv_count); ++i, index += global_size) {
	    if ((index < cols * rows) && (index >= 0)) {
	    	int cur_row = (int)(index/cols);
	        int cur_col = index - cur_row * cols;
	 
	        int p_rows = (isInverse)?floor(rows/2.):ceil(rows/2.);
	        int p_cols = (isInverse)?floor(cols/2.):ceil(cols/2.);
	 
	        int p_rows_offset = (rows & 1)?(rows - p_rows):p_rows;
	        int p_cols_offset = (cols & 1)?(cols - p_cols):p_cols;
	 
	        int x_out = (cur_col > (p_cols - 1))?(cur_col - p_cols):(p_cols_offset + cur_col);
	        int y_out = (cur_row > (p_rows - 1))?(cur_row - p_rows):(p_rows_offset + cur_row);
	     
	        output[x_out + y_out * cols].real = input[index].real;
	        output[x_out + y_out * cols].imag = input[index].imag;
	    }
    }
}

__kernel void crop_data(__global clFFT_Complex *input, __global clFFT_Complex *output, 
                        int crop_side_length, int original_side_length, float lt_offset, float rb_offset)
{
    uint index = get_global_id(0);
    uint global_size = get_global_size(0);
 
    long data_length = crop_side_length * crop_side_length;
  	
  	long all_count = data_length / global_size;
 	long priv_count = (data_length % global_size) > index ? 1 : 0;
 	
	for(uint i = 0; i < (all_count + priv_count); ++i, index += global_size) {
	    if ((index < crop_side_length * crop_side_length) && (index >= 0)) {
    		long in_index = original_side_length * lt_offset + 
                    lt_offset + 
                    (ceil(index/(float)crop_side_length))*(crop_side_length + lt_offset + rb_offset) + 
                    index%crop_side_length;

    		output[index].real = input[in_index].real;
    		output[index].imag = input[in_index].imag;
	    }
    }
}