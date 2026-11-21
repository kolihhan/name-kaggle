"""Fail-closed alignment of official NTL train.csv to the repo's frozen folds.csv.

No network, no external data, no redistribution and no model training here.
All 18 label indices are derived from the same verified frozen training rows.
"""
from pathlib import Path
import pandas as pd

KNOWN_FAMILIES=('claude','gemini','glm','gpt','kimi','qwen')


def _read(path: Path, needed: set[str]) -> pd.DataFrame:
    data=pd.read_csv(path,dtype={'id':str,'prompt_id':str,'model':str})
    if not needed.issubset(data.columns):
        raise ValueError(f'{path.name} lacks exact columns {sorted(needed-set(data.columns))}')
    if data['id'].isna().any() or data['id'].duplicated().any():
        raise ValueError(f'{path.name} IDs are missing or duplicated')
    if data['prompt_id'].isna().any() or data['model'].isna().any():
        raise ValueError(f'{path.name} has null prompt or model labels')
    return data


def load_official(train_path: Path, fold_path: Path, *, expected_rows: int = 2699,
                  expected_prompts: int = 150):
    """Return (text/label frame, prompt->fold, index->family, sorted labels).

    Only existing official 18-class data may be passed. The returned numeric
    label indices do not claim to match any separately trained checkpoint.
    """
    train=_read(Path(train_path),{'id','prompt_id','model','text'})
    frozen=_read(Path(fold_path),{'id','prompt_id','model','fold_id'})
    if (len(train)!=expected_rows or len(frozen)!=expected_rows or
            train.prompt_id.nunique()!=expected_prompts or
            frozen.prompt_id.nunique()!=expected_prompts):
        raise ValueError('official dataset/frozen fold cohort mismatch')
    if train['text'].isna().any() or not train['text'].astype(str).str.strip().all():
        raise ValueError('missing official passage text')
    if set(train.id)!=set(frozen.id):
        raise ValueError('frozen IDs do not match official train IDs')
    join=train.merge(frozen,on='id',suffixes=('','_fold'),validate='one_to_one',sort=False)
    if (join.prompt_id != join.prompt_id_fold).any() or (join.model != join.model_fold).any():
        raise ValueError('exact ID prompt/model lineage mismatch')
    fold_ids=pd.to_numeric(join.fold_id,errors='raise')
    if not fold_ids.isin(range(5)).all() or set(fold_ids.astype(int))!=set(range(5)):
        raise ValueError('invalid frozen 5-fold assignment')
    by_prompt=join.assign(fold_id=fold_ids.astype(int)).groupby('prompt_id').fold_id
    if by_prompt.nunique().max()!=1:
        raise ValueError('same prompt assigned to multiple folds')
    fold_map={str(k):int(v) for k,v in by_prompt.first().items()}
    labels=sorted(train.model.unique().tolist())
    if len(labels)!=18:
        raise ValueError('need all 18 official model classes')
    def family(model):
        matches=[p for p in KNOWN_FAMILIES if model.startswith(p+'-') or model.startswith(p+'.') or (p=='qwen' and model.startswith('qwen3.'))]
        if len(matches)!=1:
            raise ValueError(f'unknown model family {model}')
        return matches[0]
    families=[family(x) for x in labels]
    fam_ids={x:i for i,x in enumerate(sorted(set(families)))}
    family_map={i:fam_ids[v] for i,v in enumerate(families)}
    class_ids={x:i for i,x in enumerate(labels)}
    output=train[['id','prompt_id','text']].copy()
    output['label']=train.model.map(class_ids).astype(int)
    return output,fold_map,family_map,labels
