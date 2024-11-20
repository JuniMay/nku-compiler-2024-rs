const float f=2.2*99;
float norm=8.3;
int testnorm(int l){
        return l;
    }
int testconst(int k){
        return k;
    }
int main()
{
    const int i=7;
    int j=77;
    int ans=testnorm(i)+testconst(j);
    float fk=f+norm;
    return 0;
}