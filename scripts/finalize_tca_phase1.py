"""Inference-only replay of frozen phase-1 checkpoint and beam legality."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scripts.audit_tiger_generation import load_model, OFFSETS
from scripts.audit_tca_effect import logs
from genrec.data.amazon import AmazonSeqDataset
from genrec.trainers.tiger_trainer import CollateFn, calculate_pos_index, recall_at_k, ndcg_at_k
from genrec.trainers.tca_loss import sha256_file


def main():
    root = Path('out/tiger/amazon/beauty/rqkmeans')
    folder = root/'tiger_tca_full_vocab_ndcg_select'
    recorded = json.loads((folder/'results.json').read_text())
    metrics = recorded['metrics']
    history = logs(folder)
    assert metrics['best_epoch'] == 111 and metrics['selection_metric'] == 'NDCG@10'
    assert max(history, key=lambda e: history[e]['Valid']['NDCG@10']) == 111
    checkpoint = folder/'best_model.pt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(checkpoint, device)
    sid_path = root/'semantic_ids.pt'
    raw = torch.load(sid_path, map_location='cpu', weights_only=False)['sem_ids'].long()
    known = set(map(tuple, (raw+torch.tensor(OFFSETS)).tolist()))
    result = {'checkpoint': str(checkpoint.resolve()), 'sha256': sha256_file(checkpoint),
        'best_epoch':111, 'selection_metric':'NDCG@10', 'strict_load':True, 'splits':{}}
    for split in ['valid','test']:
        dataset = AmazonSeqDataset(root='dataset/amazon', split='beauty', train_test_split=split,
            max_seq_len=50, add_disambiguation=False, semantic_id_path=str(sid_path), target_n_layers=3)
        assert len(dataset)==22363
        loader = DataLoader(dataset,batch_size=96,shuffle=False,num_workers=0,collate_fn=CollateFn(256,3,3,150))
        scores={f'{m}@{k}':[] for m in ['Recall','NDCG'] for k in [5,10]}
        legal=np.zeros(10,dtype=np.int64)
        recognized=np.zeros(10,dtype=np.int64)
        with torch.no_grad(),torch.amp.autocast('cuda',enabled=device.type=='cuda',dtype=torch.float16):
            for batch in tqdm(loader,desc=split):
                g=model.generate(batch['input_ids'].to(device),batch['attention_mask'].to(device),num_beams=10)
                g=g[:,1:].reshape(-1,10,3).cpu()
                pos=calculate_pos_index(g,batch['labels'],maxk=10)
                for k in [5,10]:
                    scores[f'Recall@{k}'].append(recall_at_k(pos,k))
                    scores[f'NDCG@{k}'].append(ndcg_at_k(pos,k))
                a=g.numpy()
                legal+=((a>=np.array(OFFSETS))&(a<np.array(OFFSETS)+256)).all(-1).sum(0)
                recognized+=np.array([[tuple(s) in known for s in beams] for beams in a]).sum(0)
        replay={k:sum(v)/len(v) for k,v in scores.items()}
        diffs={k:abs(v-metrics[f'best_{split}_{k}']) for k,v in replay.items()}
        if max(diffs.values())>1e-7:
            raise RuntimeError(f'Material replay mismatch: {split}: {replay}, differences={diffs}')
        result['splits'][split]={'samples':len(dataset),'metrics':replay,'absolute_differences':diffs,
            'beam1_level_valid':float(legal[0]/len(dataset)),
            'top10_level_valid':float(legal.sum()/(10*len(dataset))),
            'beam1_known_sid':float(recognized[0]/len(dataset)),
            'top10_known_sid':float(recognized.sum()/(10*len(dataset))),
            'unknown_all':float(1-recognized.sum()/(10*len(dataset))),
            'level_valid_unknown':float((legal-recognized).sum()/(10*len(dataset)))}
    destination=folder/'phase1_final_replay.json'
    destination.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
