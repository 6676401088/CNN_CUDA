// CUDA runtime
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "utils.h"

#define CUDA_CALL(x) {if((x) != cudaSuccess){ \
          printf("CUDA error at %s:%d\n",__FILE__,__LINE__); \
          printf(" %s\n", cudaGetErrorString(cudaGetLastError())); \
          exit(EXIT_FAILURE);}}

#define LINE_BUFFER_SIZE 100  //Line buffer size for read write 

/*******************************************************************************
 * Input   : char array containing the filename containing all weights, number of weights
 * Output  : array matrix filled with weights for each feature map
 * Procedure: read all weights from file and strore in array
 ******************************************************************************/
void read_weight1(const char filename[], int size, float matrix[]) {
  FILE* finput;
    
  finput = fopen(filename , "rb" );
  if (finput==NULL) {fputs ("File error",stderr); exit (13);}
  
  fread(matrix, sizeof(float), size, finput);
  fclose(finput);
}

/************************************************************************************
 * Function: void read_bias(char filename[], int length, float vector[])
 * Input   : char array containing the filename and location for reading, number of bias values this
                is the same as the number of output featuremaps, pointer for output
                * Output  : vector filled with bias weights for each feature map
                * Procedure: read bias weights from file normalize to uchar range and strore on correct possition
                ************************************************************************************/
void read_bias1(const char filename[], int length, float vector[]) {
  int i;
  FILE* finput;
  
  finput = fopen(filename , "rb" );
  if (finput==NULL) {fputs ("File error",stderr); exit (13);}
  
  fread(vector, sizeof(float), length, finput);
  for(i=0; i<length; i++){
    vector[i]=256*vector[i];
  }
  fclose(finput);
}

void read_image_pgm(unsigned char image[], char filename[], int imageWidth, int imageHeight)
{   
  int grayMax;
  int PGM_HEADER_LINES=3;
  FILE* input;

  int headerLines = 1;
  int scannedLines= 0;
  long int counter =0;

  //read header strings
  char *lineBuffer = (char *) malloc(LINE_BUFFER_SIZE+1);
  char *split;
  char *format = (char *) malloc(LINE_BUFFER_SIZE+1);
  char P5[]="P5";
  char comments[LINE_BUFFER_SIZE+1];

  //open the input PGM file
  input=fopen(filename, "rb");

  //read the input PGM file header
  while(scannedLines < headerLines){
    fgets(lineBuffer, LINE_BUFFER_SIZE, input);
    //if not comments
    if(lineBuffer[0] != '#'){
      scannedLines += 1;
      //read the format
      if(scannedLines==1){
        split=strtok(lineBuffer, " \n");
        strcpy(format,split);
        if(strcmp(format,P5) == 0){
          //printf("FORMAT: %s\n",format);
          headerLines=PGM_HEADER_LINES;
        }
        else
        {
          printf("Only PGM P5 format is support. \n");
        }
      }
      //read width and height
      if (scannedLines==2)
      {
        split=strtok(lineBuffer, " \n");
        if(imageWidth == atoi(split)){ //check if width matches description
          //printf("WIDTH: %d, ", imageWidth);
        }
        else{
          printf("input frame has wrong width should be WIDTH: %d, ", imageWidth);
          exit(4);
        }
        split = strtok (NULL, " \n");
        if(imageHeight == atoi(split)){ //check if heigth matches description
          //printf("HEIGHT: %d\n", imageHeight);
        }
        else{
          printf("input frame has wrong height should be HEIGHT: %d, ", imageHeight);
          exit(4);
        }
      }
      // read maximum gray value
      if (scannedLines==3)
      {
        split=strtok(lineBuffer, " \n");
        grayMax = atoi(split);
        //printf("GRAYMAX: %d\n", grayMax);
      }
    }
    else
    {
      strcpy(comments,lineBuffer);
      //printf("comments: %s", comments);
    }
  }

  counter = fread(image, sizeof(unsigned char), imageWidth * imageHeight, input);
  //printf("pixels read: %d\n",counter);
        
  //close the input pgm file and free line buffer
  fclose(input);
  free(lineBuffer);
  free(format);
}

void write_image_pgm(unsigned char image[], const char filename[], int imageWidth, int imageHeight){
    FILE * output;
    output = fopen(filename, "wb");
    fprintf(output, "P5\n");
    fprintf(output, "%d %d\n255\n",imageWidth, imageHeight);
    fwrite(image, sizeof(unsigned char), imageWidth * imageHeight, output);
    fclose(output);
}


/******************************************
 * Device function declaration
 *****************************************/
__global__
void layer1_init_bias(float* d_y, float* d_bias, int r);
__global__
void layer1_feature_maps(float* d_y, unsigned char* d_in_layer, float* d_weight, int r);
__global__
void layer1_sigmoid(float* d_y, unsigned char* d_out_layer, int r);


void cuda_convolution_layer1(unsigned char in_layer[], unsigned char out_layer[],
			     const float bias[], const float weight[]) {

  unsigned int i;

  /*********************************
   * allocate device memory on GPU
   *********************************/

  unsigned int size_y = 6*358*638;
  unsigned int mem_size_y = sizeof(float) * size_y;
  float *d_y;

  unsigned int size_bias = 6;
  unsigned int mem_size_bias = sizeof(float) * size_bias;
  float *d_bias;

  unsigned int size_weight = 6*36;
  unsigned int mem_size_weight = sizeof(float) * size_weight;
  float *d_weight;

  unsigned int size_in_layer = 720*1280;
  unsigned int mem_size_in_layer = sizeof(unsigned char) * size_in_layer;
  unsigned char *d_in_layer;

  unsigned int size_out_layer = 6*358*638;
  unsigned int mem_size_out_layer = sizeof(unsigned char) * size_out_layer;
  unsigned char *d_out_layer;

  // Setup execution parameters 
  dim3 threads;
  dim3 grid;


  cudaEvent_t start1, end1, start2, end2;
  float time1, time2;
  cudaEventCreate(&start1);
  cudaEventCreate(&end1);
  cudaEventRecord(start1,0);

  CUDA_CALL(cudaMalloc((void **) &d_y, mem_size_y)); 
  CUDA_CALL(cudaMalloc((void **) &d_in_layer, mem_size_in_layer)); 
  CUDA_CALL(cudaMalloc((void **) &d_bias, mem_size_bias)); 
  CUDA_CALL(cudaMalloc((void **) &d_weight, mem_size_weight)); 
  CUDA_CALL(cudaMalloc((void **) &d_out_layer, mem_size_out_layer)); 

  /*********************************************
   * copy data from host (CPU) to device (GPU)
   ********************************************/
  CUDA_CALL(cudaMemcpy(d_in_layer, in_layer, mem_size_in_layer, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_bias, bias, mem_size_bias, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_weight, weight, mem_size_weight, cudaMemcpyHostToDevice));

  CUDA_CALL(cudaDeviceSynchronize());

  threads.x = 16;
  threads.y = 16;
  threads.z = 1;

  grid.x = (638/16)+1;
  grid.y = (358/16)+1;
  grid.z = 1;

  cudaEventCreate(&start2);
  cudaEventCreate(&end2);
  cudaEventRecord(start2,0);

  for(i = 0; i < 6; i++){
    layer1_init_bias<<<grid,threads>>>(d_y, d_bias, i);
  }

  /* Just in case, put a sync here */
  CUDA_CALL(cudaDeviceSynchronize());


  threads.x = 8;
  threads.y = 8;
  threads.z = 1;

  grid.x = (638/8)+1;
  grid.y = (358/8)+1;
  grid.z = 1;
  /* Loop over each feature map */
  for(i = 0; i < 6; i++){
    layer1_feature_maps<<<grid,threads>>>(d_y, d_in_layer, d_weight, i);
  }

  CUDA_CALL(cudaDeviceSynchronize());


  threads.x = 16;
  threads.y = 16;
  threads.z = 1;
  grid.x = (638/16)+1;
  grid.y = (358/16)+1;
  grid.z = 1;
  for(i = 0; i < 6; i++){
    layer1_sigmoid<<<grid,threads>>>(d_y, d_out_layer, i);
  }

  cudaEventRecord(end2,0);
  cudaEventSynchronize(end2);
  cudaEventElapsedTime (&time2, start2, end2);


  /* Read back the output from device (GPU) to host (CPU) */
  CUDA_CALL(cudaMemcpy(out_layer, d_out_layer, mem_size_out_layer, cudaMemcpyDeviceToHost) );


  CUDA_CALL(cudaDeviceSynchronize());

  /* release device memory */
  CUDA_CALL(cudaFree(d_y))
  CUDA_CALL(cudaFree(d_in_layer));
  CUDA_CALL(cudaFree(d_bias));
  CUDA_CALL(cudaFree(d_weight));
  CUDA_CALL(cudaFree(d_out_layer));
  cudaEventRecord(end1,0);
  cudaEventSynchronize(end1);
  cudaEventElapsedTime (&time1, start1, end1);
  printf("Total time by layer 1 : %.9f\t sec\n",time2/1000);
  printf("Total time (comm+comput) : %.9f\t sec\n",time1/1000);
  cudaEventDestroy( start1 );
  cudaEventDestroy( end1 );
  cudaEventDestroy( start2 );
  cudaEventDestroy( end2 );
}

__global__
void layer1_init_bias(float* d_y, float* d_bias, int r){
  // Thread index 
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;
  /* row and column index */
  int m, n;
  /* get the index for each thread */
  m = by*16+ty;
  n = bx*16+tx;
  /* mask out threads that are out of bound */
  if(m<358 && n<638){
    d_y[r*358*638+m*638+n]=d_bias[r];
  }

}


__global__
void layer1_feature_maps(float* d_y, unsigned char* d_in_layer, float* d_weight, int r){
  // Thread index 
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;
  /* row and column index */
  int m,n;
  /* base address of the d_in_layer */
  int d_in_layer_base;
  /* convolution index */
  int l, k;
  /* local register for convolution */
  float accu;
  /* pack 4 unsigned char into each load */
  uchar4 uchar4_tmp;


  __shared__ float s_weight[6*6];


  __shared__ unsigned int s_in_layer[(8*2+8)*(8*2+5)];

  /* row and column index of each thread */
  m = by*8+ty;
  n = bx*8+tx;

  d_in_layer_base = by*8*2*1280+bx*8*2;

  if(tx<6 && ty<6){
    s_weight[ty*6+tx] = d_weight[r*36+ty*6+tx];
  }


  if(tx<6){
    uchar4_tmp = ((uchar4*)d_in_layer)[ (d_in_layer_base + ty*1280 + tx*4) / 4];
    s_in_layer[ty*24+tx*4] = uchar4_tmp.x;
    s_in_layer[ty*24+tx*4+1] = uchar4_tmp.y;
    s_in_layer[ty*24+tx*4+2] = uchar4_tmp.z;
    s_in_layer[ty*24+tx*4+3] = uchar4_tmp.w;
    uchar4_tmp = ((uchar4*)d_in_layer)[ (d_in_layer_base + (8+ty)*1280 + tx*4) / 4];
    s_in_layer[(8+ty)*24+tx*4] = uchar4_tmp.x;
    s_in_layer[(8+ty)*24+tx*4+1] = uchar4_tmp.y;
    s_in_layer[(8+ty)*24+tx*4+2] = uchar4_tmp.z;
    s_in_layer[(8+ty)*24+tx*4+3] = uchar4_tmp.w;
    if(ty<5){
      uchar4_tmp = ((uchar4*)d_in_layer)[ (d_in_layer_base + (16+ty)*1280 + tx*4) / 4];
      s_in_layer[(16+ty)*24+tx*4] = uchar4_tmp.x;
      s_in_layer[(16+ty)*24+tx*4+1] = uchar4_tmp.y;
      s_in_layer[(16+ty)*24+tx*4+2] = uchar4_tmp.z;
      s_in_layer[(16+ty)*24+tx*4+3] = uchar4_tmp.w;
    }
  }
  __syncthreads();

  /* mask out threads that are out of bound */
  if(m<358 && n<638){
    /* load the bias from global mem to register */
    accu = d_y[r*358*638+m*638+n];
    /* the actual convolution */
    for(k=0; k<6; k++){
      for(l=0; l<6; l++){
	accu += s_in_layer[(ty*2+k)*24+(tx*2+l)] * s_weight[k*6+l];
      }
    }
    /* write back from register to global mem */
    d_y[r*358*638+m*638+n] = accu;
  }

}


__global__
void layer1_sigmoid(float* d_y, unsigned char* d_out_layer, int r){
  // Thread index 
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;
  /* row and column index of each thread */
  int m, n;
  /* get the index for each thread */
  m = by*16+ty;
  n = bx*16+tx;
  /* mask out threads that are out of bound */
  if(m<358 && n<638){
    d_out_layer[r*358*638+m*638+n]=(unsigned char)(255.999f/(1.0+expf(-1.0*d_y[r*358*638+m*638+n]/256.0)));
  }

}


///////////////////// CPU code

/*******************************************************************************
 * @file
 * File:   conv_layer4.c
 * @author Maurice Peemen <m.c.j.peemen@tue.nl>
 * @author Dongrui She <d.she@tue.nl>
 *
 * @copyright  Eindhoven University of Technology - Electronic Systems Group
 * @date       2013
 * @section    DISCLAIMER
 *             THIS SOFTWARE IS PROVIDED "AS IS" AND WITHOUT ANY EXPRESS OR
 *             IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
 *             WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *             PURPOSE.
 *
 * @section DESCRIPTION
 *
 ******************************************************************************/


/************************************************************************************
 * Input   : input image, pointer to output result, coefficients bias and weights
 * Output  : neuron outputs of the feature maps represented as an image
 * Procedure: perform feed forward computation through the feature extraction layers
     *******************************************************************************/
void run_convolution_layer1(unsigned char in_layer[], unsigned char out_layer[],
                            const float bias[], const float weight[]) {
  int k,l,m,n,r;
  static float y[6*358*638];

  //init values of feature maps at bias value
  for(r=0; r<6; r++){
    for(m=0; m<358*638; m++){
      y[r*358*638+m]=bias[r];
    }
  }  

  //loop over output feature maps
  for(r=0; r<6; r++){
    //convolve weight kernel with input image
    for(m=0; m<358; m++){//shift input window over image
      for(n=0; n<638; n++){
        //multiply input window with kernel
  for(k=0; k<6; k++){
    for(l=0; l<6; l++){
        y[r*358*638+m*638+n] += in_layer[(m*2+k)*1280+n*2+l] * weight[r*36+k*6+l];
          }
  }
      }
    }
  }
  
  //sigmoid activation function
  for(r=0; r<6*358*638; r++){
    out_layer[r]=(unsigned char)(255.999f/(1+expf(-y[r]/256)));
  }

}

/********************************************************************************
 * Input   : input image, pointer to output result, coefficients bias and weights
 * Output  : the neuron outputs computed from the input pattern
 * Procedure: perform feed forward computation through the neural network
 ********************************************************************************/
void run_convolution_layer2(unsigned char in_layer[], unsigned char out_layer[],
                            const float bias[], const float weight[]) {
  int k,l,m,n,q,r,qindex;
  static float y[16*177*317];
  //feature maps are sparse connected therefore connection scheme is used
  const int qq[60]={0,1,2, 1,2,3, 2,3,4, 3,4,5, 0,4,5, 0,1,5,
                    0,1,2,3, 1,2,3,4, 2,3,4,5, 0,3,4,5, 0,1,4,5, 0,1,2,5,
                    0,1,3,4, 1,2,4,5, 0,2,3,5, 0,1,2,3,4,5};

  //init values of feature map at bias value
  for(r=0; r<16; r++){
    for(q=0; q<177*317; q++){
      y[r*177*317+q]=bias[r];
    }
  }  
            
  //loops over output feature maps with 3 input feature maps
  for(r=0; r<6; r++){
    for(q=0; q<3; q++){//connect with all connected 3 input feature maps
      qindex=qq[r*3+q];//lookup connection address
      //convolve weight kernel with input image
      for(m=0; m<177; m++){//shift input window over image
        for(n=0; n<317; n++){
          //multiply input window with kernel
          for(k=0; k<6; k++){
            for(l=0; l<6; l++){
              y[r*177*317+m*317+n] += in_layer[qindex*358*638+(m*2+k)*638+n*2+l]
                * weight[(r*3+q)*36+k*6+l];
            }
          }
        }
      }         
    }
  }
    
  for(r=0; r<9; r++){//loop over output feature maps with 4 input maps
    for(q=0; q<4; q++){//connect with all connected 4 input feature maps
      qindex=qq[r*4+q+18];//lookup feature map adress
        
      //convolve weight kernel with input image
      for(m=0; m<177; m++){//shift input window over image
        for(n=0; n<317; n++){
          //multiply input window with kernel
          for(k=0; k<6; k++){
            for(l=0; l<6; l++){
              y[(r+6)*177*317+m*317+n]
                += in_layer[qindex*358*638+(m*2+k)*638+n*2+l]
                * weight[(r*4+q+18)*36+k*6+l];
            }
          }
        }
      }
    }
  }
  
  //compute last feature map connected with all 6 input feature maps
  for(q=0; q<6; q++){//connect with all input feature maps
    qindex=qq[54+q];//lookup the 6 adresses
    
    //convolve weight kernel with input image
    for(m=0; m<177; m++){//shift input window over image
      for(n=0; n<317; n++){
        //multiply input window with kernel
        for(k=0; k<6; k++){
          for(l=0; l<6; l++){
            y[15*177*317+m*317+n]
              += in_layer[qindex*358*638+(m*2+k)*638+n*2+l]
              * weight[(54+q)*36+k*6+l];
          }
        }
      }
    }
  }

  for(r=0; r<16*177*317; r++){ //sigmoid activation
    out_layer[r]=255.999f/(1+expf(-y[r]/256));
  }
}

/************************************************************************************
 * Input   : input image, pointer to output result, coefficients bias and weights
 * Output  : the neuron outputs computed from the input pattern
 * Procedure: perform feed forward computation through the neural network
 ************************************************************************************/
void run_convolution_layer3(unsigned char in_layer[], unsigned char out_layer[],
                            const float bias[], const float weight[]) {
  int k,l,m,n,q,r;
  static float y[80*173*313];

  //init values of feature maps at bias value
  for(r=0; r<80; r++){
    for(q=0; q<173*313; q++){
      y[r*173*313+q]=bias[r];
    }  
  }

  for(q=0; q<8; q++){//connect with first 8 feature maps
    for(r=0; r<40; r++){//loops over first 40 output feature maps
      //convolve weight kernel with input image
      for(n=0; n<313; n++){//shift input window over image
        for(m=0; m<173; m++){      
          //multiply input window with kernel
          for(l=0; l<5; l++){//only 5x5 convolution
            for(k=0; k<5; k++){//there is no subsampling in this layer    
              y[r*173*313+m*313+n]
                += in_layer[q*177*317+(m+k)*317+n+l] * weight[(r*8+q)*25+k*5+l];
            }
          }
        }
      }
    }           
  }
  
  for(q=8; q<16; q++){//connect with last 8 feature maps
    for(r=40; r<80; r++){ //loops over remaining 40 output feature maps 
      //convolve weight kernel with input image
      for(n=0; n<313; n++){//shift input window over image
        for(m=0; m<173; m++){
          //multiply input window with kernel
          for(l=0; l<5; l++){//only 5x5 convolution 
            for(k=0; k<5; k++){     
              y[r*173*313+m*313+n] += in_layer[q*177*317+(m+k)*317+n+l]
                * weight[(r*8+q-8)*25+k*5+l];
            }
          }
        }       
      }
    }           
  }
  
  for(r=0; r<80*173*313; r++){//sigmoid activation function
    out_layer[r]=255.999f/(1+expf(-y[r]/256));
  }
}

/************************************************************************************
 * Input   : input image, coefficients bias and weights, vote array for detected signs
 * Output  : voting histogram for the signs
 * Procedure: perform feed forward computation through the neural network layer
              threshold with neuron output to detect signs at pixel positions
************************************************************************************/
int run_convolution_layer4(unsigned char in_layer[], const float bias[],
                            const float weight[], unsigned int detect[]) {
  int m,n,q,r;
  int detections=0;
  int posx, posy;
  float y;
  int set=0;

  float max;

  //convolve weight kernel with input image
  for(m=0; m<173; m++){//shift input window over image
    for(n=0; n<313; n++){
      //init values of feature map at bias value
      y = bias[0];
      for(q=0; q<80; q++){
        y += in_layer[q*173*313+m*313+n] * weight[q];
      }
      // no sigmoid required sigmoid threshold 0.6 => potential should be
      // inverse -ln(0.6^-1 -1)= 0.405 x 256 = 103.799
      //if (y >= 103.799f){ // if sign detected figure out which sign
    if (y >= 0.0f){ // if sign detected figure out which sign
        max=0;
        for(r=1; r<8; r++){// check other 7 maps for the stronges sign
          y = bias[r];
          for(q=0; q<80; q++){
            y += in_layer[q*173*313+m*313+n] * weight[r*80+q];
          }
          //if (y>=103.799f && y>max){
      if (y>=0.0f && y>max){
        max=y;
            posx=n*4;
      posy=m*4;
      detect[detections*4]=posx;
      detect[detections*4+1]=posy;
      detect[detections*4+2]=r;
      detect[detections*4+3]=100.0f/(1+expf(-y/256));
      set=1;
          }
        }
        if (set==1){//this means that a sign is found
          detections=detections+1;
        set=0;
        }
      }
    }           
  }
  return detections;
}

void annotate_img(unsigned char img[], unsigned int detectarray[], int detections)
{
  int i,x,y,posx,posy; 
  
  for(i=0; i<detections; i++){
    posx=detectarray[i*4];
  posy=detectarray[i*4+1];
    for(x=0; x<32; x++){
    img[posy*1280+posx+x]=255;
    img[(posy+31)*1280+posx+x]=255;
  }
    for(y=1; y<31; y++){
      img[(posy+y)*1280+posx]=255;
    img[(posy+y)*1280+posx+31]=255;
    } 
  }
}

int main(void) {
  //int i;
  //,j,k;
  const int max_speed[8]={0, 30, 50, 60, 70, 80, 90, 100};
  char imagename[32]; 
  static unsigned char in_image[720*1280];//for input image
  static unsigned char in_image_annotate[720*1280];//for output image
  //feature map results due to unroling+2 otherwise writes outside array
  static unsigned char net_layer1[6*358*638];
  static unsigned char net_layer2[16*177*317];
  static unsigned char net_layer3[80*173*313];

  static unsigned char cuda_net_layer1[6*358*638];

  static float bias1[6];  //memory for network coefficients
  static float weight1[6*36];
  static float bias2[16];
  static float weight2[(6*3+9*4+6)*36];
  static float bias3[80];
  static float weight3[25*8*80];
  static float bias4[8];
  static float weight4[80*8]; 
  
  static unsigned int detectarray[3*10];
  int detections;

  clock_t starttime, endtime; //vars for measure computation time

  /* check the error */

  read_bias1("data/bias01.bin", 6, bias1);
  read_weight1("data/weight01.bin", 6*36, weight1);

  read_bias1("data/bias02.bin", 16, bias2);
  read_weight1("data/weight02.bin", 2160, weight2);

  read_bias1("data/bias03.bin", 80, bias3);
  read_weight1("data/weight03.bin", 25*8*80, weight3);

  read_bias1("data/bias04.bin", 8, bias4);
  read_weight1("data/weight04.bin", 80*8, weight4);

  //compute input name
  sprintf(imagename,"data/test%06d.pgm",46);

  //read image from file
  read_image_pgm(in_image, imagename, 1280, 720);
  // duplicate image for annotation
  memcpy(in_image_annotate, in_image, sizeof(unsigned char)*1280*720);  

  //start timer
  starttime=clock();
        
  //perform feed forward operation thourgh the network
  run_convolution_layer1(in_image, net_layer1, bias1, weight1);
  run_convolution_layer2(net_layer1, net_layer2, bias2, weight2);
  run_convolution_layer3(net_layer2, net_layer3, bias3, weight3);
  detections=run_convolution_layer4(net_layer3, bias4, weight4, detectarray);      

  //stop timer
  endtime=clock();
  printf("CPU Elapsed time is %f s\n", 1.0*(endtime-starttime)/CLOCKS_PER_SEC);
  
  printf("number of detections = %d\n",detections);
 // for(i=0; i<detections; i++){
 //   printf("detection nr %d = %d km/h, box pos= x %d, y %d, confidence = %d\n",i,max_speed[detectarray[i*4+2]], detectarray[i*4],detectarray[i*4+1],detectarray[i*4+3]);
 // }

  annotate_img(in_image_annotate, detectarray, detections);
  
  write_image_pgm(in_image_annotate, "output.pgm", 1280, 720);  

  /****************************
   * CPU is done. GPU starts.
   ***************************/

  printf("\n ######## GPU running... ######## \n");
  //start timer
  starttime=clock();

  cuda_convolution_layer1(in_image, cuda_net_layer1, bias1, weight1);

  //stop timer
  endtime=clock();
  printf("GPU Elapsed time is %f s\n", 1.0*(endtime-starttime)/CLOCKS_PER_SEC);

  return 0;
}