// Grid adapter with increasing-capacity flow reuse, following the update
// procedure in Nicholas Howe's imgcutmulti.cpp. See Howe_LICENSE.txt for the
// copyright notice, conditions, and disclaimer retained with this derivative.
// Boykov-Kolmogorov maxflow v3.01 is separately licensed; see Maxflow_README.txt.
#include <cstdint>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "graph.h"
std::vector<double> read_array(std::ifstream &f,size_t n) {
    std::vector<double> a(n);f.read(reinterpret_cast<char*>(a.data()),n*sizeof(double));
    if(!f)throw std::runtime_error("incomplete array");
    for(double v:a)if(!std::isfinite(v))throw std::runtime_error("nonfinite capacity");
    return a;
}
struct Edge {int u,v;double a;};
int main(int argc,char **argv) {
    try {
        if(argc!=3)throw std::runtime_error("sweep_cli input output");
        std::ifstream f(argv[1],std::ios::binary);uint32_t h=0,w=0,k=0;
        f.read(reinterpret_cast<char*>(&h),4);f.read(reinterpret_cast<char*>(&w),4);f.read(reinterpret_cast<char*>(&k),4);
        if(!f||h<2||w<2||k<1)throw std::runtime_error("bad header");
        auto cs=read_array(f,k);size_t n=size_t(h)*w;
        for(size_t i=0;i<k;i++)if(cs[i]<0||(i&&cs[i]<cs[i-1]))throw std::runtime_error("weights must increase");
        auto src=read_array(f,n),snk=read_array(f,n),down=read_array(f,size_t(h-1)*w),right=read_array(f,size_t(h)*(w-1));
        std::vector<Edge> edges;edges.reserve(size_t(h-1)*w+size_t(h)*(w-1));
        for(uint32_t y=0;y+1<h;y++)for(uint32_t x=0;x<w;x++){double a=down[size_t(y)*w+x];if(a<0)throw std::runtime_error("negative pair cost");if(a>0)edges.push_back({int(y*w+x),int((y+1)*w+x),a});}
        for(uint32_t y=0;y<h;y++)for(uint32_t x=0;x+1<w;x++){double a=right[size_t(y)*(w-1)+x];if(a<0)throw std::runtime_error("negative pair cost");if(a>0)edges.push_back({int(y*w+x),int(y*w+x+1),a});}
        typedef Graph<double,double,double> G;G g(static_cast<int>(n),static_cast<int>(edges.size()));g.add_node(n);
        for(size_t i=0;i<n;i++)g.add_tweights(i,src[i],snk[i]);
        for(const auto&e:edges)g.add_edge(e.u,e.v,cs[0]*e.a,cs[0]*e.a);
        std::ofstream out(argv[2],std::ios::binary);std::vector<uint8_t> b(n);
        for(size_t i=0;i<k;i++) {
            if(i&&!edges.empty()) {
                auto a=g.get_first_arc();
                for(size_t j=0;j<edges.size();j++) {
                    auto a2=g.get_next_arc(a);const auto&e=edges[j];
                    if(g.get_rcap(a)==0||g.get_rcap(a2)==0){g.mark_node(e.u);g.mark_node(e.v);}
                    double delta=(cs[i]-cs[i-1])*e.a;
                    g.set_rcap(a,g.get_rcap(a)+delta);g.set_rcap(a2,g.get_rcap(a2)+delta);
                    if(j+1<edges.size())a=g.get_next_arc(a2);
                }
            }
            double flow=g.maxflow(i>0);
            for(size_t j=0;j<n;j++)b[j]=(g.what_segment(j)==G::SINK);
            out.write(reinterpret_cast<char*>(&flow),8);out.write(reinterpret_cast<char*>(b.data()),n);
            if(!out)throw std::runtime_error("cannot write");
            std::cout<<i+1<<"/"<<k<<" c="<<cs[i]<<"\n"<<std::flush;
        }
        return 0;
    }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
}
