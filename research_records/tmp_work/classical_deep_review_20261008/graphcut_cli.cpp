// Standalone image-grid adapter to the separately licensed Boykov-Kolmogorov
// maxflow v3.01 library. No MATLAB or modifications to the project pipeline.
#include <cstdint>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "graph.h"

std::vector<double> read_array(std::ifstream &f, size_t n) {
    std::vector<double> a(n);
    f.read(reinterpret_cast<char*>(a.data()), n*sizeof(double));
    if (!f) throw std::runtime_error("Incomplete graph input");
    for (double v:a) if (!std::isfinite(v)) throw std::runtime_error("Nonfinite capacity");
    return a;
}
int main(int argc,char**argv) {
    try {
        if(argc!=3) throw std::runtime_error("Usage: graphcut_cli input.bin output.bin");
        std::ifstream f(argv[1],std::ios::binary);
        uint32_t h=0,w=0;
        f.read(reinterpret_cast<char*>(&h),4); f.read(reinterpret_cast<char*>(&w),4);
        if(!f || h<2 || w<2) throw std::runtime_error("Invalid image shape");
        size_t n=size_t(h)*w;
        auto src=read_array(f,n),snk=read_array(f,n);
        auto down=read_array(f,size_t(h-1)*w),right=read_array(f,size_t(h)*(w-1));
        size_t edges=0;
        for(double v:down) {if(v<0) throw std::runtime_error("Negative edge");edges+=(v>0);}
        for(double v:right) {if(v<0) throw std::runtime_error("Negative edge");edges+=(v>0);}
        typedef Graph<double,double,double> G;
        G g(static_cast<int>(n),static_cast<int>(edges));
        g.add_node(static_cast<int>(n));
        for(size_t i=0;i<n;i++) g.add_tweights(static_cast<int>(i),src[i],snk[i]);
        for(uint32_t y=0;y+1<h;y++) for(uint32_t x=0;x<w;x++) {
            double v=down[size_t(y)*w+x]; int i=y*w+x;
            if(v>0) g.add_edge(i,i+w,v,v);
        }
        for(uint32_t y=0;y<h;y++) for(uint32_t x=0;x+1<w;x++) {
            double v=right[size_t(y)*(w-1)+x]; int i=y*w+x;
            if(v>0) g.add_edge(i,i+1,v,v);
        }
        double flow=g.maxflow();
        std::vector<uint8_t> result(n);
        for(size_t i=0;i<n;i++) result[i]=(g.what_segment(static_cast<int>(i))==G::SINK);
        std::ofstream o(argv[2],std::ios::binary);
        o.write(reinterpret_cast<char*>(result.data()),n);
        if(!o) throw std::runtime_error("Cannot write result");
        std::cout.precision(17); std::cout<<flow<<"\n";
        return 0;
    } catch(const std::exception &e) {std::cerr<<e.what()<<"\n"; return 1;}
}
