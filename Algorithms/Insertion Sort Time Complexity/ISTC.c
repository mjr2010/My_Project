#include<stdio.h>
#include<conio.h>
#include<time.h>
void main()
{	 int i,j,n,key,a[10];
 float t;
 clock_t t1,t2;
 clrscr();
 t1=clock();
  printf("\nEnter the number of elements ");
  scanf("%d",&n);
  printf("\nEnter the numbers= ");
  for(i=0;i<n;i++)
  {	scanf("%d",&a[i]);		}
  for(i=1;i<n;i++)
  {	key=a[i];
   	j=i-1;
  	 while(j>=0 && a[j]>key)
   	{	a[j+1]=a[j];
    		j=j-1;
    		a[j+1]=key;	}
  }
  printf("\nSorted numbers are ");
  for(i=0;i<n;i++)
  {	printf("\t%d",a[i]);		}
 t2=clock()-t1;
 t=(float)t2/CLOCKS_PER_SEC;
 printf("\nTime Complexity= %f",t);
 getch();
}
