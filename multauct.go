/*  multauct -- Progressive Second Price Auctions
    Written 2025 by Eric Olson */

package main

import ("os"; . "fmt"; "bufio"; "sort"; "strconv"; "math")

type buyspec struct {
	i,key int // buyer index and secondary sort key
	qbar,pbar float64 // valuation curve and budget
}
func (b *buyspec)theta(z float64)float64 {
	if z>=b.qbar { return 0.5*b.qbar*b.pbar }
	return (1.0-0.5*z/b.qbar)*z*b.pbar
}
func (b *buyspec)dtheta(z float64)float64 {
	r:=(1-z/b.qbar)*b.pbar
	if r<0 { return 0 }
	return r
}
func (b *buyspec)dthetainv(r float64)float64 {
	z:=(1-r/b.pbar)*b.qbar
	if z<0 { return 0 }
	return z
}

type selspec struct {
	j int // auction index
	Q,P float64 // quantity available, reserve price
}

type bidspec struct {
	i,j int // buyer index, auction index
	q,p float64 // quantity and marginal price
	a float64 // allocations
}

var (buyseed=2137; bidseed=3915; mtype="mauct:30")
var (nbuy=10; nsel=3; nbid=20; bcount=0; epsilon=5.0)
var (buyers []buyspec; sellers []selspec; bids,bidsold []bidspec)
var (Qs=[]float64{1000.0}; Ps=[]float64{12.0})
var stdout *bufio.Writer

func mkbuyers(){
	buyers=make([]buyspec,nbuy+nsel); bcount=len(buyers)
	var bs rstate
	bs.rseed(uint64(buyseed))
	for k:=0;k<7;k++ { bs.rint32() } // spin for luck
	for k:=0;k<nsel;k++ { // sellers
		buyers[k].i=k
		buyers[k].key=k
		buyers[k].qbar=0
		buyers[k].pbar=0
	}	
	for k:=nsel;k<len(buyers);k++ { // buyers
		buyers[k].i=k
		buyers[k].qbar=bs.rfloat64()*50+50
		buyers[k].pbar=bs.rfloat64()*10+10
	}
	for k:=nsel;k<len(buyers)-1;k++ { // buyback lowest priority
		l:=k+int(bs.rdice(uint32(len(buyers)-k)))
		buyers[l].key,buyers[k].key=buyers[k].key,buyers[l].key
	}
}
func prbuyers(fp *bufio.Writer){
	Fprintf(fp,"\n\n# %s %s %s\n","i","qbar","pbar")
	for i:=nsel;i<len(buyers);i++ {
		Fprintf(fp,"%d %g %g\n",
			buyers[i].i,buyers[i].qbar,buyers[i].pbar)
	}
	fp.Flush()
}

func mksellers(){
	sellers=make([]selspec,nsel)
	for j:=0;j<nsel;j++ { // sellers
		sellers[j].j=j
		sellers[j].Q=Qs[j%len(Qs)]
		sellers[j].P=Ps[j%len(Ps)]
	}	
}
func prsellers(fp *bufio.Writer){
	Fprintf(fp,"\n\n# %s %s %s\n","j","Q","P")
	for j:=0;j<len(sellers);j++ {
		Fprintf(fp,"%d %g %g\n",
			sellers[j].j,sellers[j].Q,sellers[j].P)
	}
	fp.Flush()
}

func prefix(s,t string)bool {
	n:=len(s)
	if len(t)>=n&&s==t[0:n] { return true }
	return false
}
func mkgraph(mtype string){
	bids=make([]bidspec,nsel,nsel+nbid)
	for k:=0;k<nsel;k++ { // reserve prices
		bids[k].i=k; bids[k].j=k
	}
	if prefix("mauct:",mtype){ // nsel auctions xp percent mixed
		xp,err:=strconv.ParseFloat(mtype[6:],64)
		if err!=nil { xp=30.0 }
		di:=int(xp*float64(nbuy)/float64(nsel)/100.0+0.5)
		Printf("# %d auctions %g percent or %d overlap.\n",
			nsel,xp,di)
		for j:=0;j<nsel;j++ {
			imin:=j*nbuy/nsel-di
			imax:=(j+1)*nbuy/nsel+di
			if imin<0 { imin=0 }
			if imax>nbuy { imax=nbuy }
			for i:=imin;i<imax;i++ {
				bids=append(bids,bidspec{i+nsel,j,0.0,0.0,0.0})
			}
		}
		Printf("# Created a total of %d bidslots!\n",len(bids))
	} else {
		bids=bids[0:nsel+nbid]
		for k:=nsel;k<len(bids);k++ {
			i:=nsel+(k-nsel)%nbuy; bids[k].i=i
			bids[k].j=(k-nsel)%nsel
		}
	}
	bidsold=make([]bidspec,len(bids))
}

func mkbids(){
	var bs,ps rstate
	bs.rseed(uint64(bidseed))
	for k:=0;k<7;k++ { ps.rint32() } // spin for luck
	for k:=0;k<nsel;k++ { // reserve prices
		bids[k].p=sellers[k].P
		bids[k].q=sellers[k].Q
	}
	for i:=nsel;i<len(buyers);i++ {
		n:=0
		for k:=nsel;k<len(bids);k++ {
			if bids[k].i==i { n++ }
		}
		if n>0 {
			p:=buyers[i].pbar*bs.rfloat64()
			q:=buyers[i].dtheta(p)/float64(n)
			for k:=nsel;k<len(bids);k++ {
				if bids[k].i==i {
					bids[k].q=q
					bids[k].p=p
				}
			}
		}
	}
}

func pralloc(fp *bufio.Writer){
	Fprintf(fp,"\n\n# %s %s %s %s %s\n","i","j","q","p","a")
	for k:=0;k<len(bids);k++ {
		Fprintf(fp,"%d %d %g %g %g\n",
		bids[k].i,bids[k].j,bids[k].q,bids[k].p,bids[k].a)
	}
	fp.Flush()
}
func getalloc(i int){ // allocations missing i
	S:=make([]float64,len(sellers))
	for j:=range S {
		S[j]=sellers[j].Q
	}
	bidorder:=func(i,j int) bool {
		if bids[i].p==bids[j].p {
			return buyers[bids[i].i].key<buyers[bids[j].i].key
			//return buyers[bids[i].i].key>buyers[bids[j].i].key
		}
		return bids[i].p>bids[j].p
	}
	sort.Slice(bids,bidorder)
	for k:=range bids {
		j:=bids[k].j
		if bids[k].i==i {
			bids[k].a=0
		} else {
			bids[k].a=min(bids[k].q,S[j])
		}
		S[j]-=bids[k].a
	}
	for j:=range S {
		if S[j]!=0.0 {
			Printf("Surplus of %g in auction %d minus buyer %d!\n",
				S[j],j,i)
			os.Exit(1)
		}
	}
}

func ebest(i int)[]float64 {
	if i<nsel {
		Printf("Seller buyback %d has no ebest reply!\n",i)
		os.Exit(1)
	}
	getalloc(i)
	Ji:=make([]bool,len(sellers))
	for k:=0;k<len(bids);k++ {
		if bids[k].i==i { Ji[bids[k].j]=true }
	}
	qj:=make([]float64,len(sellers))
	qtot:=epsilon/buyers[i].pbar
	for k:=len(bids)-1;k>=0;k-- {
		if !Ji[bids[k].j] { continue }
		dq:=buyers[i].dthetainv(bids[k].p)-qtot // how much more?
		if dq<=0 { // don't want any more
			break
		} else if dq<bids[k].a { // satisfied and exit
			qtot+=dq
			qj[bids[k].j]+=dq
			break
		}
		qtot+=bids[k].a // win bid and continue
		qj[bids[k].j]+=bids[k].a
	} // took everything
	return qj
}
func makebid(i int,qj []float64) {
	q:=0.0
	for j:=range(qj) { q+=qj[j] }
	p:=buyers[i].dtheta(q)
	for k:=0;k<len(bids);k++ {
		if bids[k].i==i {
			buyers[i].key=bcount
			bids[k].q=qj[bids[k].j]
			bids[k].p=p
		}
	}
}

func sumpiai(i int)float64 { // sum ai*pi of opposing bids
	r:=0.0
	for k:=0;k<len(bids);k++ {
		if bids[k].i==i { continue }
		r+=bids[k].p*bids[k].a
	}
	return r
}

func sumai(i int)float64 { // sum ai of buyer i
	r:=0.0
	for k:=0;k<len(bids);k++ {
		if bids[k].i==i { r+=bids[k].a }
	}
	return r
}
func ci(i int)float64 {
	getalloc(i); c0:=sumpiai(i)
	getalloc(-1); c:=sumpiai(i)
    return c0-c
}
func ui(i int)float64 {
	getalloc(i); c0:=sumpiai(i)
	getalloc(-1); c:=sumpiai(i); qtot:=sumai(i)
    return buyers[i].theta(qtot)-c0+c
}
func prmarket(fp *bufio.Writer){
	Fprintf(fp,"\n\n# %s %s %s %s %s %s %s\n",
		"i","ai","S[qi]","pi","vi","ci","gi")
	tval:=0.0; trev:=0.0
	for i:=0;i<len(buyers);i++ {
		c:=0.0
		if i<nsel { getalloc(-1)
		} else { c=ci(i) }
		a:=0.0; p:=0.0; q:=0.0; g:=make([]int,0,nsel)
		for k:=range(bids) {
			if bids[k].i==i {
				a+=bids[k].a; g=append(g,bids[k].j)
				p=bids[k].p
				q+=bids[k].q
			}
		}
		v:=0.0
		if i>=nsel {
			v=buyers[i].theta(a)
		}
		Fprintf(fp,"%d %.6g %.6g %.6g %.6g %.6g %v\n",
			buyers[i].i,a,q,p,v,c,g)
		if i>=nsel {
			tval+=v; trev+=c
		}
	}
	Fprintf(fp,"# After %d bids\n# Total value %g\n"+
		"# Total revenue %g\n",bcount,tval,trev)
	fp.Flush()
}

func sumpiaim(i int)[]float64 { // per-market sum ai*pi
	rj:=make([]float64,nsel)
	for k:=0;k<len(bids);k++ {
		if bids[k].i==i { continue }
		rj[bids[k].j]+=bids[k].p*bids[k].a
	}
	return rj
}
func getcim(i int)[]float64 {
	getalloc(i); c0m:=sumpiaim(i)
	getalloc(-1); cm:=sumpiaim(i)
	for j:=0;j<len(c0m);j++ { c0m[j]-=cm[j] }
    return c0m
}
func getrevenue()[]float64 { // poor code
	rj:=make([]float64,nsel)
	for i:=nsel;i<len(buyers);i++ {
		cj:=getcim(i)
		for j:=0;j<len(rj);j++ { rj[j]+=cj[j] }
	}
	return rj
}
func marketstat()([]float64,[]float64,[]float64,[]float64){
	rtot:=getrevenue() // side effect getalloc(-1)
	qres:=make([]float64,nsel)
	for k:=0;k<len(bids);k++ { // how much bought back
		if bids[k].i<nsel { qres[bids[k].j]=bids[k].a }
	}
	ptot:=make([]float64,nsel); p2tot:=make([]float64,nsel)
	atot:=make([]float64,nsel); a2tot:=make([]float64,nsel)
	for k:=0;k<len(bids);k++ {
		if bids[k].i<nsel { continue } // omit reserve prices
		a:=bids[k].a; p:=bids[k].p; j:=bids[k].j
		atot[j]+=a; a2tot[j]+=a*a
		ptot[j]+=a*p; p2tot[j]+=a*p*p
	}
	pavg:=make([]float64,nsel); pstd:=make([]float64,nsel)
	for j:=0;j<nsel;j++ {
		pstd[j]=0.0
		if atot[j]<1e-7 {
			pavg[j]=0
		} else {
			pavg[j]=ptot[j]/atot[j]
			gamma:=a2tot[j]/atot[j]
			if atot[j]>gamma {
				pstd[j]=math.Sqrt(math.Abs((p2tot[j]-
					pavg[j]*pavg[j]*atot[j])/(atot[j]-gamma)))
			}
		}
	}
	return pavg,pstd,rtot,qres
}

func trybid(i int)bool {
	copy(bidsold,bids)
	uiold:=ui(i); keyold:=buyers[i].key
	qj:=ebest(i); makebid(i,qj)
	uinew:=ui(i)
	if uinew<uiold+epsilon {
		copy(bids,bidsold)
        buyers[i].key=keyold
		return false
	}
	bcount++
	return true
}

func prstat(fp *bufio.Writer){
	pavg,pstd,rtot,qres:=marketstat()
	Fprintf(fp,"\n\n#%s %s %s %s %s\n","j","pavg","pstd","rtot","qres")
	for j:=0;j<nsel;j++ {
		Fprintf(fp,"%d %g %g %g %g\n",j,pavg[j],pstd[j],rtot[j],qres[j])
	}
	fp.Flush()
}

func prconfig(fp *bufio.Writer){
	Fprintf(fp,"# buyseed=%d\n",buyseed)
	Fprintf(fp,"# bidseed=%d\n",bidseed)
	Fprintf(fp,"# nbuy=%d\n",nbuy)
	Fprintf(fp,"# nsel=%d\n",nsel)
	Fprintf(fp,"# nbid=%d\n",nbid)
	Fprintf(fp,"# Qs=%v\n",Qs)
	Fprintf(fp,"# Ps=%v\n",Ps)
	Fprintf(fp,"# epsilon=%g\n",epsilon)
	Fprintf(fp,"# mtype=%s\n",mtype)
	fp.Flush()
}
func main(){
	stdout=bufio.NewWriter(os.Stdout)
	defer stdout.Flush()
	Printf("multauct -- Progressive Second Price Auctions\n"+
		"Version %s Serial %s\n\n",Version,Serial)
	tic()
	cfgread("multauct.conf")
	buyseed=cfgint("buyseed",buyseed)
	bidseed=cfgint("bidseed",bidseed)
	nbuy=cfgint("nbuy",nbuy)
	nsel=cfgint("nsel",nsel)
	nbid=cfgint("nbid",nbid)
	Qs=cfgfloatslc("Qs",Qs)
	Ps=cfgfloatslc("Ps",Ps)
	epsilon=cfgfloat("epsilon",epsilon)
	mtype=cfgstring("mtype",mtype)
	prconfig(stdout)
	mkbuyers(); prbuyers(stdout)
	mksellers(); prsellers(stdout)
	mkgraph(mtype); mkbids()
	getalloc(-1); pralloc(stdout)
	prmarket(stdout)

	Printf("\n\n");
	for m:=0;m<3000;m++ {
		done:=true
		for i:=nsel;i<len(buyers);i++ {
			if trybid(i) { done=false }
		}
		if done { 
			Printf("# Convergence took %d rounds!\n",m+1)
			break
		}
	}
	getalloc(-1); pralloc(stdout)
	prmarket(stdout)
	prstat(stdout)

	t:=toc()
	Printf("\nTotal execution time %g seconds.\n",t)
	os.Exit(0)
}
