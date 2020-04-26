#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct _CTM{
    char    filename[16];
    int     tmp;
    float   t_begin;
    float   t_duration;
    char    phone[16];
}CTM;

void err_fopen(char *fname);
void usage();

int main(int argc, char **argv)
{
    FILE *flab=NULL, *fctm=NULL;
    CTM ctm;
    char filename_tmp[16]="null_lab";

    if(argc!=2)
        usage();

    fctm = fopen(argv[1], "rb");
    if(!fctm)
        err_fopen(argv[1]);

    for(; fscanf(fctm, "%s%d%f%f%s", ctm.filename, &ctm.tmp, &ctm.t_begin, &ctm.t_duration, ctm.phone)==5;){

        if(!strcmp(ctm.filename, filename_tmp)){
            fprintf(flab, "%.3f\t%.3f\t%s\n", ctm.t_begin, ctm.t_begin+ctm.t_duration, ctm.phone);
        }
        else{
            if(!flab){
                flab = fopen(ctm.filename, "wb");
                fprintf(flab, "%.3f\t%.3f\t%s\n", ctm.t_begin, ctm.t_begin+ctm.t_duration, ctm.phone);
            }
            else{
                fclose(flab);
                flab = fopen(ctm.filename, "wb");
                fprintf(flab, "%.3f\t%.3f\t%s\n", ctm.t_begin, ctm.t_begin+ctm.t_duration, ctm.phone);
            }
        }
        strcpy(filename_tmp, ctm.filename);
    }

    fclose(flab);
    fclose(fctm);
    return 0;
}

void usage(){
    printf("Convert CTM format to label file\n");
    printf("Usage: ctm2lab <ctm-file>\n");
    printf("e.g. ctm2lab test.ctm\n\n");
    exit(1);
}

void err_fopen(char *fname){
    printf("%s is not found!!\n\n", fname);
    exit(1);
}

