import numpy as np

def phased_memmap_like(rng, footprint_pages:int, phase_seconds:int=60, seq_jump_pages:int=128, stride_k_pages:int=4):
    page=0; t=0; phase=0; switch=phase_seconds
    
    # Calc Zipf dist
    ranks=np.arange(1, footprint_pages+1, dtype=np.float64)
    pmf=1.0/(ranks**1.1); pmf/=pmf.sum(); cdf=np.cumsum(pmf)
    def rz():
        u=rng.random(); return int(np.searchsorted(cdf,u))

    while True:
        if t>0 and t%switch==0: phase=(phase+1)%3
        # Phase1 : Sequential access
        if phase==0: page=(page+seq_jump_pages)%footprint_pages; write=False
        # Phase2 : Sequential access
        elif phase==1: page=(page+stride_k_pages)%footprint_pages; write=True
        # Phase3 : Random(Zipf) access
        else: page=rz()%footprint_pages; write=False

        # write Y/N is not used for now,,,
        # Assume we are in read only env
        yield (page, write); t+=1
