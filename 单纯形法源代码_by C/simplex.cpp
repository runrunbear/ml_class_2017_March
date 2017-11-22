#include <math.h>
#include <string.h>
#include <queue>
#include "nrutil.h"
#include "nrutil.c"
#include "amotry.c"
#include "amoeba.c"
#define NDIM 2

float funk(float x[]){
	//要求的最小值函数写在这里。比如这里我要求
	//y=(x1-3)^2 + (x2-2)^2
	//的最小值

	return pow(x[1]-3,2)+pow(x[2]-2,2);
}

int main(){
	// 单纯形法拟合九个参数，NDIM在宏定义中定义为9
	float a[NDIM+2][NDIM+1];
	float y[NDIM+2];
	float lamda=1.0;
	float **aa,ftol=0.01f;
	int i;
	memset(a,0,sizeof(a));
	// 初始化p[][]
	for(i=2;i<=NDIM+1;i++){
		a[i][i-1]=lamda;
	}
	// 预计算y[]
	for(i=1;i<=NDIM+1;i++)
		y[i]=funk(a[i]);

	aa=(float **)malloc((unsigned) (NDIM+2)*sizeof(float*));
	for(i=0;i<NDIM+2;i++) aa[i]=a[i];
	int *nfunk=new int();
	amoeba(aa,y,NDIM,ftol,*funk,nfunk);
	for(i=1;i<=NDIM;i++)
		printf("X%d=%f ",i,a[1][i]);
	printf("\n");
	
	delete nfunk;
	return 0;
}
