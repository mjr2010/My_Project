#include<stdio.h>
#include<conio.h>
#include<math.h>
#include<graphics.h>
#include<dos.h>
void main()
{
int gd=DETECT,gm;
int xc,yc,x,y,p;
float r;
initgraph(&gd,&gm,"C:\\turboc3\\BGI");
printf("Enter the radius and center");
scanf("%f %d %d",&r,&xc,&yc);
x=0;
y=r;
p=(5/4)-r;
do
{
if(p<0)
{
//putpixel(xc+x+1,yc+y,7);
x=x+1;
p=p+(2*x)+1;
}
else
{
//putpixel(xc+x+1,yc+y+1,7);
x=x+1;
y=y-1;
p=p+(2*x)-(2*y)+1;
}

putpixel(xc+y,yc+x,7);
putpixel(xc+x,yc+y,7);
putpixel(xc+x,yc-y,7);
putpixel(xc+y,yc-x,7);
putpixel(xc-y,yc-x,7);
putpixel(xc-x,yc-y,7);
putpixel(xc-x,yc+y,7);
putpixel(xc-y,yc+x,7);
delay (100);
}
while(x<y);
delay(200);
closegraph();
getch();
}
