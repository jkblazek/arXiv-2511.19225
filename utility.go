package main

import ("time"; . "math"; "os"; "bufio"; "fmt"; "strings"; "strconv")

/*	Timing routines based on wall clock */
var tictime time.Time
func tic(){
    tictime=time.Now()
}
func toc() float64 {
    now:=time.Now()
    elapsed:=now.Sub(tictime)
    return elapsed.Seconds()
}

/*  Fido's favorite middle-square Weyl sequence random numbers.  See
    Bernard Widynski, Middle-Square Weyl Sequence RNG, arXiv:1704.00358v7
    Ron Kneusel, Random Numbers and Computers, Springer, 13-14 (2018)
*/
type rstate struct {
    x,w,s uint64
	f bool; y float64 // Gaussian
}
func (p *rstate)rint32() uint32 {
    p.x*=p.x; p.w+=p.s
    p.x+=p.w; p.x=(p.x>>32)|(p.x<<32)
    return uint32(p.x)
}
func (p *rstate)rint64() uint64 {
    r:=uint64(p.rint32())<<32
    return r|uint64(p.rint32())
}
func ipoly8(x uint64) uint64{
    x2:=x*x
    return (((x2*x2+
        0x4cca10b00bc5da27)*x2+
        0x872a4954ca3a3537)*x+
        0x2bf0e3ddcc4fb44f)*x+
        0xb5ad4eceda1ce2a9
}
func (p *rstate)rseed(x uint64){
	p.x=x; p.w=0
	p.s=ipoly8(x)|1
}
func (p *rstate)rfloat64() float64 {
	r:=float64(p.rint64()>>1)
	return float64(r/float64(uint64(1)<<63))
}
func (p *rstate)rnormal() float64 {
	if p.f {
		p.f=false
		return p.y
	}
	var v1,v2,rr float64
	for {
		v1=2*p.rfloat64()-1
		v2=2*p.rfloat64()-1
		rr=v1*v1+v2*v2
		if rr!=0&&rr<1 { break }
	}
	dy:=Sqrt(-2*Log(rr)/rr)
	p.y=v2*dy
	p.f=true
	return v1*dy
}
func (p *rstate)rweibull(lambda,beta float64) float64 {
	return lambda*Pow(-Log(1-p.rfloat64()),1/beta)
}
func (p *rstate)qrand(qs []float64) { // qs sums to 1
    A:=0.0
    for l:=range qs {
        Z:=p.rnormal()
        qs[l]=Z; A+=Z*Z
    }
    for l:=range qs {
        qs[l]*=qs[l]/A
    }
}
func (p *rstate)rdice(d uint32)uint32 {
    q:=0x7fffffff/d
    for {
        r:=(p.rint32()&0x7fffffff)/q
        if r<d { return r }
    }
}
var gs=rstate{0,0,0xb5ad4eceda1ce2a9,false,0}

/*  Read configuration */
var cfgmap=make(map[string]string)
func cfgread(s string){
	fd,err:=os.Open(s)
	if err!=nil {
		fmt.Printf("Error opening file %s.\n%v\n",fd,err)
		os.Exit(1)
	}
	defer fd.Close()
	fp:=bufio.NewScanner(fd)
	for fp.Scan() {
		v:=strings.SplitN(fp.Text(),"=",2)
		if len(v)>1 {
			cfgmap[v[0]]=v[1]
		} else {
			cfgmap[v[0]]="true"
		}
	}
}
func cfgfloat(s string,r float64)float64 {
    if p,ok:=cfgmap[s]; ok {
        if v,err:=strconv.ParseFloat(p,64); err==nil {
            r=v
        }
    }
	return r
}
func cfgfloatslc(s string,r []float64)[]float64 {
    if p,ok:=cfgmap[s]; ok {
		r=r[:0]
		a:=strings.Split(p,",")
		for _,p:=range a {
	        if v,err:=strconv.ParseFloat(p,64); err==nil {
	            r=append(r,v)
	        }
		}
    }
	return r
}
func cfgint(s string,r int)int {
    if p,ok:=cfgmap[s]; ok {
        if v,err:=strconv.ParseInt(p,0,0); err==nil {
            r=int(v)
        }
    }
	return r
}
func cfgstring(s string,r string)string {
    if p,ok:=cfgmap[s]; ok {
		r=p
    }
	return r
}
