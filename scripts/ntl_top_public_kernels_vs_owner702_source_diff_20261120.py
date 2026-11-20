"""Read-only source diff of owner-scored 0.702 and public 0.704 notebooks.

No model inference, no external data download, no Kaggle submission.
One source may be a public derivative of another. Notebook filename/title are
not evidence that a scoring result belongs to a particular user.
"""
import json,sys,re,hashlib,difflib,ast
from pathlib import Path

def _cell_text(cell):
    s=cell.get('source','')
    return ''.join(s) if isinstance(s,list) else str(s)
def parse(path):
    j=json.loads(Path(path).read_text())
    cells=[_cell_text(c) for c in j.get('cells',[]) if c.get('cell_type')=='code']
    full='\n'.join(cells)
    parts={}
    for i,s in enumerate(cells):
        rows=s.splitlines()
        # typical notebook starts each script with %%writefile and then Python
        m=re.match(r'\s*%%writefile\s+(?:-a\s+)?(?P<file>[\w./-]+)',rows[0]) if rows else None
        if m:
            key=m.group('file')
            parts[key]='\n'.join(rows[1:])
        else:
            # Keep stable cells identifiable in diff
            parts[f'CELL_{i:02d}']='\n'.join(rows)
    return {'path':path,'code_cells':len(cells),
            'code_sha256':hashlib.sha256(full.encode()).hexdigest(),
            'parts':parts,'full':full}
def audit(champion,slug,other):
    A=champion['parts'];B=other['parts']
    shared=set(A)&set(B)
    a_only=sorted(set(A)-set(B))
    b_only=sorted(set(B)-set(A))
    changed=sorted(k for k in shared if A[k].strip()!=B[k].strip())
    print('NTL_PUBLIC_TOP_DIFF_META='+json.dumps({
        'candidate':slug,
        'owner_champion_code_sha':champion['code_sha256'],
        'candidate_code_sha':other['code_sha256'],
        'champion_code_cells':champion['code_cells'],
        'candidate_code_cells':other['code_cells'],
        'common_part_count':len(shared),
        'parts_identical':len(shared)-len(changed),
        'parts_modified':changed[:70],
        'parts_missing_in_champion':b_only[:60],
        'parts_missing_in_candidate':a_only[:60],
        'candidate_code_chars':len(other['full']),
        'champion_code_chars':len(champion['full']),
    },sort_keys=True),flush=True)
    # Keep only concise source deltas for grounded technical review,
    # prioritize newly changed scoring, GLM, Fable, routing & ensemble.
    for key in changed:
        sa=A[key];sb=B[key]
        linesa=sa.splitlines()
        linesb=sb.splitlines()
        seq=difflib.SequenceMatcher(None,linesa,linesb,autojunk=False)
        opcodes=[o for o in seq.get_opcodes() if o[0]!='equal']
        edits=[]
        for tag,i1,i2,j1,j2 in opcodes:
            old='\n'.join(linesa[i1:i2])
            new='\n'.join(linesb[j1:j2])
            keep_terms=('glm','fable','qwen','route','weight','penalty','score','share','evidence',
                        'label','gate','strength','likelihood','mix','family','temperature','prior')
            weight=3 if any(v in (old+' '+new).lower() for v in keep_terms) else 1
            edits.append((weight,tag,i1,j1,old[:700],new[:850]))
        edits=sorted(edits,key=lambda x:-x[0])
        print('NTL_PUBLIC_TOP_FILE_DIFF='+json.dumps({
             'candidate':slug,'file':key,
             'old_lines':len(linesa),'new_lines':len(linesb),
             'line_similarity':round(seq.ratio(),5),
             'change_hunks':len(opcodes),
             'top_hunks':[{'kind':kind,'old_line':i+1,'new_line':j+1,
               'old_excerpt':old,'new_excerpt':new} for _,kind,i,j,old,new in edits[:8]],
        },ensure_ascii=False),flush=True)
    for key in b_only:
        if len(key)>75:continue
        s=B[key]
        print('NTL_PUBLIC_TOP_NEW_PART='+json.dumps({
           'candidate':slug,'file':key,'linecount':len(s.splitlines()),
           'start_excerpt':s[:850],
           'critical_mechanism_keywords':[z for z in
             ('glm','fable','qwen','routing','likelihood','model','sibling','ensemble',
             'logits','inference') if z in s.lower()]
        },ensure_ascii=False),flush=True)

def main():
    owner=parse(sys.argv[1])
    target={'owner_sha':owner['code_sha256'],'code_cells':owner['code_cells']}
    print('NTL_PUBLIC_TOP_OWNER_SOURCE='+json.dumps(target),flush=True)
    for x in sys.argv[2:]:
        slug=Path(x).stem
        ob=parse(x)
        audit(owner,slug,ob)
    print('NTL_PUBLIC_TOP_CODE_DIFF_ONLY_NO_SUBMISSION=COMPLETE',flush=True)
if __name__=='__main__':main()
