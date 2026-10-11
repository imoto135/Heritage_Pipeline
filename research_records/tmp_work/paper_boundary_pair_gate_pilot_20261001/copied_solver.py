def solve_fields(original_lab,reference,active,minimum_pairs):
    assert not np.any(reference&active)
    smooth=reference_lowpass(original_lab,reference)
    fields={m:np.zeros_like(original_lab) for m in MODES}
    used=np.zeros(reference.shape,bool);applied=np.zeros(reference.shape,bool);notes=[]
    n,labels,stats,_=cv2.connectedComponentsWithStats(active.astype(np.uint8),connectivity=4)
    H,W=active.shape
    for k in range(1,n):
        x,y,w,h,area=map(int,stats[k])
        if area<64:
            notes.append(dict(id=k,area=area,applied=False,reason='area below 64'));continue
        component=labels==k;yy,xx=np.nonzero(component);N=len(yy)
        index=np.full(active.shape,-1,np.int32);index[yy,xx]=np.arange(N)
        diagonal=np.zeros(N,np.float64);rhs=np.zeros((N,6),np.float64)
        rows=[];cols=[];values=[];boundary_values=[];local_used=np.zeros_like(used)
        for dy,dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny,nx=yy+dy,xx+dx;valid=(ny>=0)&(ny<H)&(nx>=0)&(nx<W)
            ii=np.flatnonzero(valid);oy,ox=ny[valid],nx[valid]
            inside=component[oy,ox];ids=ii[inside];diagonal[ids]+=1
            rows.extend(ids.tolist());cols.extend(index[oy[inside],ox[inside]].tolist());values.extend([-1.]*len(ids))
            boundary=(~inside)&reference[oy,ox];ids=ii[boundary]
            diagonal[ids]+=1
            vv=np.concatenate([original_lab[oy[boundary],ox[boundary]],smooth[oy[boundary],ox[boundary]]],axis=1)
            rhs[ids]+=vv;boundary_values.append(vv)
            local_used[oy[boundary],ox[boundary]]=True
        pairs=sum(len(v) for v in boundary_values)
        if pairs<minimum_pairs:
            notes.append(dict(id=k,area=area,applied=False,reason=f'fewer than {minimum_pairs} paper boundary pairs',boundary_pairs=pairs));continue
        rows.extend(range(N));cols.extend(range(N));values.extend(diagonal.tolist())
        matrix=sparse.coo_matrix((values,(rows,cols)),shape=(N,N)).tocsr()
        solution=spsolve(matrix,rhs)
        assert np.isfinite(solution).all()
        residual=float(np.abs(matrix@solution-rhs).max())
        refs=np.concatenate(boundary_values).astype(np.float64)
        lower,upper=refs.min(axis=0),refs.max(axis=0)
        assert np.all(solution>=lower-1e-7) and np.all(solution<=upper+1e-7)
        assert residual<1e-6
        for j,mode in enumerate(MODES):fields[mode][yy,xx]=solution[:,j*3:j*3+3].astype(np.float32)
        applied|=component;used|=local_used
        notes.append(dict(id=k,xywh=[x,y,w,h],area=area,applied=True,boundary_pairs=pairs,
            residual_max=residual,reference_min_Lab=lower.tolist(),reference_max_Lab=upper.tolist(),
            fill_min_Lab=solution.min(axis=0).tolist(),fill_max_Lab=solution.max(axis=0).tolist()))
    return fields,applied,used,notes
