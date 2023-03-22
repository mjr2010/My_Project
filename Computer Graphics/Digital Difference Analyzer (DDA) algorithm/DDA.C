#include<stdio.h>
#include<conio.h>
#include<graphics.h>
#include<dos.h>
int main()
{
float x,y,x1,y1,x2,y2,dx,dy,step;
int i,gd=DETECT,gm;
initgraph(&gd,&gm,"C:\\turboc3\\BGI");
printf("enter first coordinates\n");
scanf("%f%f",&x1,&y1);
printf("enter second coordinat\n");
scanf("%f%f",&x2,&y2);
dx=x2-x1;
dy=y2-y1;
if(dx>dy)
{
step=dx;
}
else
{
step=dy;
}
dx=dx/step;
dy=dy/step;
x=x1;
y=y1;
for(i=1;i<=step;i++)
{
putpixel(x,y,7);
x=x+dx;
y=y+dy;
delay(50);
}
closegraph();
return 0;
}
