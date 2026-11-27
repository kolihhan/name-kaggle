// Exact full-vocabulary teacher-forced token scorer; llama.cpp b12774.
// Input: one line per job: prefix_length target_length prefix_ids... target_ids...
// Output: ordinal, mean log probability, first-token log probability, scored count.
// All raw text/IDs stay on the private ephemeral runner.
#include "llama.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>
static double logp(const float* x, int n, int t) {
    if (!x || t<0 || t>=n) throw std::runtime_error("invalid logits or target");
    float m=*std::max_element(x,x+n);
    double z=0;
    for(int i=0;i<n;++i) z+=std::exp(double(x[i]-m));
    double p=double(x[t])-m-std::log(z);
    if(!std::isfinite(p)||p>1e-7) throw std::runtime_error("nonfinite log probability");
    return p;
}
int main(int argc,char**argv) {
    if(argc!=3) return 2;
    try {
        llama_backend_init();
        auto mp=llama_model_default_params();mp.n_gpu_layers=0;
        auto *model=llama_model_load_from_file(argv[1],mp);
        if(!model) throw std::runtime_error("model load");
        auto cp=llama_context_default_params();cp.n_ctx=2048;cp.n_batch=1024;cp.n_ubatch=256;
        cp.n_threads=4;cp.n_threads_batch=4;cp.n_seq_max=1;
        auto *ctx=llama_init_from_model(model,cp);
        if(!ctx) throw std::runtime_error("context load");
        int nv=llama_vocab_n_tokens(llama_model_get_vocab(model));
        std::ifstream in(argv[2]); if(!in) throw std::runtime_error("input open");
        std::string line; int ord=0;
        std::cout<<std::setprecision(12);
        while(std::getline(in,line)) {
            std::istringstream ss(line);int p,n;
            if(!(ss>>p>>n)||p<1||n<1||p+n-1>1024) throw std::runtime_error("bad lengths");
            std::vector<llama_token> ids(p+n);
            for(auto &t:ids) if(!(ss>>t)||t<0||t>=nv) throw std::runtime_error("bad token");
            int junk;if(ss>>junk) throw std::runtime_error("trailing token");
            llama_memory_clear(llama_get_memory(ctx),true);
            auto batch=llama_batch_init(p+n-1,0,1);batch.n_tokens=p+n-1;
            for(int i=0;i<batch.n_tokens;++i){
                batch.token[i]=ids[i];batch.pos[i]=i;batch.n_seq_id[i]=1;
                batch.seq_id[i][0]=0;batch.logits[i]=(i>=p-1);
            }
            if(llama_decode(ctx,batch)!=0) throw std::runtime_error("decode failure");
            double sum=0,first=0;
            for(int i=0;i<n;++i){
                double v=logp(llama_get_logits_ith(ctx,p-1+i),nv,ids[p+i]);
                if(i==0)first=v;sum+=v;
            }
            llama_batch_free(batch);
            std::cout<<ord++<<" "<<sum/n<<" "<<first<<" "<<n<<std::endl;
        }
        llama_free(ctx);llama_model_free(model);llama_backend_free();
        if(ord==0)throw std::runtime_error("empty jobs");
        return 0;
    }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}
}
