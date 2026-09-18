#!/usr/bin/env python3
"""Deterministic Phase-2 hard-case generator and 85-point scorer.

Generate: python tools/hard_cases.py generate --output hard40
Score: python tools/hard_cases.py score --truth hard40/ground_truth.csv --predictions predictions.csv --output hard40/score.json --failures hard40/failures.csv

Original project code: no organizer generator modules are imported.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math
from pathlib import Path
import cv2
import numpy as np

LOC=((1.,1.),(2.,.8),(3.,.6),(5.,.4))
SCALE=((.005,1.),(.01,.8),(.02,.6),(.05,.4))
ROT=((.25,1.),(.5,.8),(1.,.6),(2.,.4))
PRED=("pair_id","x","y","theta","scale","found","score")
TRUTH=("pair_id","present","x","y","theta","scale")

def dump(path, rows, columns):
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=columns); w.writeheader(); w.writerows(rows)

def pattern(rng,size,family):
    a=np.full((size,size),28,np.float32); pitch=(28,34,42)[family%3]; width=(8,11,14)[family%3]
    for x in range(-pitch,size+pitch,pitch): cv2.rectangle(a,(x,0),(x+width,size-1),105+15*family,-1)
    for y in range(pitch//2,size,pitch+11): cv2.rectangle(a,(0,y),(size-1,y+max(3,width//2)),168,-1)
    for _ in range(18):
        x,y=rng.integers(12,size-35,2); w,h=rng.integers(8,31,2)
        cv2.rectangle(a,(int(x),int(y)),(int(x+w),int(y+h)),int(rng.choice((62,138,205,238))),-1)
    for _ in range(24):
        x,y=rng.integers(8,size-8,2); cv2.circle(a,(int(x),int(y)),int(rng.integers(2,5)),225,-1)
    return a

def edge(a):
    a=cv2.GaussianBlur(a.astype(np.float32),(0,0),.7)
    return cv2.Laplacian(a,cv2.CV_32F,ksize=3)

def warp(ref,scale,theta):
    side=max(25,int(round(ref.shape[0]/scale))); c=((ref.shape[1]-1)/2,(ref.shape[0]-1)/2)
    m=cv2.getRotationMatrix2D(c,theta,1/scale); m[:,2]+=np.array([(side-ref.shape[1])/2,(side-ref.shape[0])/2])
    ref=cv2.GaussianBlur(ref,(0,0),max(1.,scale/3.2))
    return cv2.warpAffine(ref,m,(side,side),flags=cv2.INTER_AREA,borderMode=cv2.BORDER_REFLECT_101)

def degrade(a,rng,severity):
    a=a.astype(np.float32)
    if severity: a=cv2.GaussianBlur(a,(0,0),(0,.5,.8,1.1,1.4)[severity])
    if rng.random()<.25+.08*severity: a=255-a
    a=255*np.power(np.clip(a/255,0,1),rng.uniform(.72,1.4))
    yy,xx=np.indices(a.shape); cx,cy=(np.array(a.shape[::-1])-1)/2
    a*=1-(((xx-cx)**2+(yy-cy)**2)/max(cx*cx+cy*cy,1))*(.04+.045*severity)
    # Keep global scan artefacts below the point where their long straight
    # edges become a stronger structural template than the planted device.
    for _ in range(severity//3):
        y=int(rng.integers(0,a.shape[0])); a[max(0,y-1):y+2]+=rng.uniform(12,28)
    a+=rng.normal(0,2.5+1.5*severity,a.shape)+rng.normal(0,.012+.006*severity,a.shape)*a
    return np.clip(np.rint(a),0,255).astype(np.uint8)

def match_audit(tpl,search,x,y):
    # Absolute ZNCC makes the audit invariant to the generator's deliberate
    # contrast-polarity inversion while retaining spatial identifiability.
    r=np.abs(cv2.matchTemplate(search.astype(np.float32),tpl.astype(np.float32),cv2.TM_CCOEFF_NORMED)); _,peak,_,loc=cv2.minMaxLoc(r)
    half=((tpl.shape[1]-1)/2,(tpl.shape[0]-1)/2); err=math.hypot(loc[0]+half[0]-x,loc[1]+half[1]-y)
    masked=r.copy(); radius=max(12,tpl.shape[0]//2); lx,ly=loc
    masked[max(0,ly-radius):ly+radius+1,max(0,lx-radius):lx+radius+1]=-1
    margin=float(peak-masked.max())
    return {"error":float(err),"peak":float(peak),"margin":margin}

def unique(tpl,search,x,y):
    """Certify the labelled match on the final pixels in two feature spaces."""
    intensity=match_audit(tpl,search,x,y)
    structure=match_audit(edge(tpl),edge(search),x,y)
    ok=(intensity["error"]<=2.0 and intensity["peak"]>=.18 and intensity["margin"]>=.03
        and structure["error"]<=5.0 and structure["peak"]>=.08 and structure["margin"]>=.01)
    return ok,intensity,structure

def make_case(seed,index,present,hard):
    rng=np.random.default_rng(seed+index*104729)
    scales=(8.,12.,8.35,11.65) if hard else (8.,12.,9.,10.,11.)
    angles=(-10.,10.,-5.,5.,-2.5,0.,2.5) if hard else (-5.,5.,-2.5,0.,2.5)
    scale,theta,severity=scales[index%len(scales)],angles[index%len(angles)],index%5
    ref=pattern(rng,600,index%3)
    # An asymmetric defect constellation makes the intended crop identifiable
    # even when the surrounding lattice is periodic and heavily degraded.
    cv2.circle(ref,(137,211),43,238,11); cv2.rectangle(ref,(382,96),(451,167),55,10)
    cv2.line(ref,(315,415),(482,523),205,13)
    ref=degrade(ref,rng,min(2,severity//2)); tpl=warp(ref,scale,theta)
    for _ in range(256):
        base=cv2.GaussianBlur(pattern(rng,384,(index+1)%3),(0,0),.7)
        background=cv2.GaussianBlur(rng.normal(90,25,(384,384)).astype(np.float32),(0,0),3.0)
        search=.35*base+.65*background; b=tpl.shape[0]//2+22
        x=float(rng.integers(b,384-b))+float(rng.uniform(-.45,.45)); y=float(rng.integers(b,384-b))+float(rng.uniform(-.45,.45))
        if present:
            tx,ty=x-(tpl.shape[1]-1)/2,y-(tpl.shape[0]-1)/2; m=np.array([[1,0,tx],[0,1,ty]],np.float32)
            layer=cv2.warpAffine(tpl,m,(384,384),flags=cv2.INTER_LINEAR)
            mask=cv2.warpAffine(np.full(tpl.shape,255,np.uint8),m,(384,384),flags=cv2.INTER_LINEAR).astype(np.float32)/255
            search=search*(1-mask)+layer*mask
            final_search=degrade(search,rng,severity)
            ok,intensity_audit,edge_audit=unique(tpl,final_search,x,y)
            if ok: break
        else:
            decoy=cv2.resize(tpl,None,fx=1.10,fy=.88,interpolation=cv2.INTER_AREA)
            cv2.line(decoy,(0,decoy.shape[0]//2),(decoy.shape[1]-1,decoy.shape[0]//2),0,3)
            dx,dy=int(x-decoy.shape[1]/2),int(y-decoy.shape[0]/2); roi=search[dy:dy+decoy.shape[0],dx:dx+decoy.shape[1]]
            roi[:]=.35*roi+.65*decoy
            final_search=degrade(search,rng,severity)
            # No instance of ``tpl`` is planted: the search base and reference
            # were generated from independent RNG draws.  Record both blind
            # peak strengths so hard negatives remain auditable.
            intensity_audit=match_audit(tpl,final_search,0.,0.)
            edge_audit=match_audit(edge(tpl),edge(final_search),0.,0.)
            break
    else: raise RuntimeError(f"failed uniqueness gate for case {index}: intensity={intensity_audit}, edge={edge_audit}")
    gt={"present":int(present),"x":round(x,6) if present else 0.,"y":round(y,6) if present else 0.,
        "theta":theta if present else 0.,"scale":scale if present else 0.,"severity":severity,
        "uniqueness_margin":round(intensity_audit["margin"],6) if present else "",
        "intensity_peak":round(intensity_audit["peak"],6),
        "intensity_error_px":round(intensity_audit["error"],6) if present else "",
        "edge_peak":round(edge_audit["peak"],6),
        "edge_margin":round(edge_audit["margin"],6) if present else "",
        "edge_error_px":round(edge_audit["error"],6) if present else "",
        "no_match_confirmed":int(not present),"seed":seed+index*104729}
    return ref,final_search,gt

def generate(args):
    root=args.output.resolve(); rd=root/"reference"; sd=root/"search"; rd.mkdir(parents=True,exist_ok=True); sd.mkdir(exist_ok=True)
    split_rng=np.random.default_rng(args.seed ^ 0x5EED5EED)
    absent=set(map(int,split_rng.choice(args.count,size=args.absent,replace=False))) if args.absent else set()
    pairs=[]; truth=[]; manifest=[]
    for i in range(args.count):
        pid=f"h{i+1:03d}"; ref,search,gt=make_case(args.seed,i,i not in absent,args.hard)
        cv2.imwrite(str(rd/f"{pid}.png"),ref); cv2.imwrite(str(sd/f"{pid}.png"),search)
        pairs.append({"pair_id":pid,"reference_path":f"reference/{pid}.png","search_path":f"search/{pid}.png"})
        truth.append({"pair_id":pid,**{k:gt[k] for k in ("present","x","y","theta","scale")}})
        manifest.append({"pair_id":pid,**gt,"reference_sha256":hashlib.sha256(ref.tobytes()).hexdigest(),"search_sha256":hashlib.sha256(search.tobytes()).hexdigest()})
    dump(root/"pairs.csv",pairs,("pair_id","reference_path","search_path")); dump(root/"ground_truth.csv",truth,TRUTH); dump(root/"manifest.csv",manifest,tuple(manifest[0]))
    (root/"generator.json").write_text(json.dumps({"version":1,"seed":args.seed,"count":args.count,"absent":args.absent,"hard":args.hard,"pose":{"scale":[8,12],"theta_deg":[-10 if args.hard else -5,10 if args.hard else 5]}},indent=2),encoding="utf-8")
    print(f"generated {args.count} pairs ({args.count-len(absent)} present, {len(absent)} absent) in {root}"); return 0

def credit(error,tiers):
    return next((value for threshold,value in tiers if error<=threshold),0.)

def auc(labels,scores):
    p=[s for y,s in zip(labels,scores) if y]; n=[s for y,s in zip(labels,scores) if not y]
    if not p or not n: return .5
    return sum(1 if a>b else .5 if a==b else 0 for a in p for b in n)/(len(p)*len(n))

def score(args):
    with args.truth.open(newline="",encoding="utf-8-sig") as f:
        reader=csv.DictReader(f); truth=list(reader)
        if tuple(reader.fieldnames or ())!=TRUTH: raise ValueError(f"truth must contain exactly {TRUTH}")
    if not truth: raise ValueError("truth must contain at least one row")
    truth_ids=[r["pair_id"] for r in truth]
    if any(not i for i in truth_ids) or len(set(truth_ids))!=len(truth_ids): raise ValueError("truth pair_id values must be non-empty and unique")
    with args.predictions.open(newline="",encoding="utf-8-sig") as f:
        reader=csv.DictReader(f); predictions_list=list(reader)
        if tuple(reader.fieldnames or ())!=PRED: raise ValueError(f"predictions must contain exactly {PRED}")
    predictions={r["pair_id"]:r for r in predictions_list}
    if len(predictions)!=len(predictions_list): raise ValueError("duplicate pair_id in predictions")
    if any(not i for i in predictions): raise ValueError("prediction pair_id values must be non-empty")
    if set(predictions)!=set(truth_ids): raise ValueError("prediction IDs do not exactly match truth IDs")
    rows=[]
    for gt in truth:
        pred=predictions[gt["pair_id"]]; present=int(gt["present"]); found=int(pred["found"])
        if present not in (0,1): raise ValueError(f"invalid present flag for {gt['pair_id']}")
        if found not in (0,1): raise ValueError(f"invalid found flag for {gt['pair_id']}")
        v={k:float(pred[k]) for k in ("x","y","theta","scale","score")}
        if not all(math.isfinite(x) for x in v.values()): raise ValueError(f"non-finite prediction for {gt['pair_id']}")
        if not 0.0<=v["score"]<=1.0: raise ValueError(f"score outside [0,1] for {gt['pair_id']}")
        if not found and any(v[k]!=0.0 for k in ("x","y","theta","scale")): raise ValueError(f"rejected pose must be zero for {gt['pair_id']}")
        gt_pose={k:float(gt[k]) for k in ("x","y","theta","scale")}
        if not all(math.isfinite(x) for x in gt_pose.values()): raise ValueError(f"non-finite truth for {gt['pair_id']}")
        if (not present and any(gt_pose.values())) or (present and gt_pose["scale"]<=0): raise ValueError(f"invalid truth pose for {gt['pair_id']}")
        err=math.hypot(v["x"]-float(gt["x"]),v["y"]-float(gt["y"])) if present else math.nan
        se=abs(v["scale"]-float(gt["scale"]))/float(gt["scale"]) if present and found else math.inf
        re=abs(v["theta"]-float(gt["theta"])) if present and found else math.inf
        correct=int(present and found and err<=5)
        failure=("false_accept" if not present and found else "false_reject" if present and not found else
                 "wrong_location" if present and err>5 else "wrong_scale" if present and se>.05 else
                 "wrong_rotation" if present and re>2 else "")
        rows.append({"pair_id":gt["pair_id"],"present":present,"found":found,"score":v["score"],
                     "error_px":err,"scale_error_rel":se,"rotation_error_deg":re,"correct":correct,"failure":failure})
    pos=[r for r in rows if r["present"]]; tp=sum(r["found"] for r in pos); fp=sum(r["found"] for r in rows if not r["present"]); fn=len(pos)-tp
    precision=tp/(tp+fp) if tp+fp else 0.; recall=tp/(tp+fn) if tp+fn else 0.; f1=2*precision*recall/(precision+recall) if precision+recall else 0.
    blocks={
        "localization_40":40*sum(credit(r["error_px"],LOC) if r["found"] else 0 for r in pos)/max(len(pos),1),
        "scale_10":10*sum(credit(r["scale_error_rel"],SCALE) for r in pos)/max(len(pos),1),
        "rotation_10":10*sum(credit(r["rotation_error_deg"],ROT) for r in pos)/max(len(pos),1),
        "rejection_15":15*f1,
        "calibration_10":10*auc([r["correct"] for r in rows],[r["score"] for r in rows]),
    }
    report={"measured_total_85":sum(blocks.values()),"blocks":blocks,"counts":{"pairs":len(rows),"present":len(pos),"tp":tp,"fp":fp,"fn":fn},"f1":f1,"correctness_auc":blocks["calibration_10"]/10,"failure_count":sum(bool(r["failure"]) for r in rows)}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    if args.failures:
        clean=[{k:"" if isinstance(v,float) and not math.isfinite(v) else v for k,v in r.items()} for r in rows]
        dump(args.failures,clean,tuple(rows[0]))
    print(json.dumps(report,indent=2)); return 0

def build_parser():
    p=argparse.ArgumentParser(description="Generate and score deterministic hard Phase-2 pairs")
    sub=p.add_subparsers(dest="command",required=True)
    g=sub.add_parser("generate",help="write blind pairs, private truth and audit manifest")
    g.add_argument("--output",type=Path,required=True); g.add_argument("--count",type=int,default=40); g.add_argument("--absent",type=int,default=10); g.add_argument("--seed",type=int,default=20260918)
    g.add_argument("--hard",action=argparse.BooleanOptionalAction,default=True,help="use +/-10 degree red-team range; --no-hard uses judge-like +/-5")
    g.set_defaults(run=generate)
    s=sub.add_parser("score",help="score localization 40, pose 20, rejection 15, calibration 10")
    s.add_argument("--truth",type=Path,required=True); s.add_argument("--predictions",type=Path,required=True); s.add_argument("--output",type=Path,required=True); s.add_argument("--failures",type=Path); s.set_defaults(run=score)
    return p

def main(argv=None):
    args=build_parser().parse_args(argv)
    if args.command=="generate" and (args.count<2 or not 0<=args.absent<args.count): raise ValueError("require count >= 2 and 0 <= absent < count")
    return args.run(args)

if __name__=="__main__": raise SystemExit(main())



